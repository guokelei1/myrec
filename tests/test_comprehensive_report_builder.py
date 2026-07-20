from __future__ import annotations

import copy
from pathlib import Path

import pytest

from myrec.mechanism import comprehensive_report_builder as builder_module
from myrec.mechanism.comprehensive_report_builder import (
    CAUSAL_CHAIN_NODES,
    CAUSAL_CHAIN_CLAIM_BOUNDARIES,
    CAUSAL_CHAIN_COMPONENTS,
    COMPREHENSIVE_REPORT_PLAN_IDENTITY,
    COMPONENT_FUNCTIONAL_QUESTIONS,
    DESIGN_NODE_COMPONENTS,
    DESIGN_NODE_CLAIM_ROLES,
    EXECUTION_AXIS_CENSUS,
    FROZEN_OBSERVATION_EVIDENCE_IDENTITIES,
    FROZEN_OBSERVATION_SCOPE_CONTRACT,
    FUNCTIONAL_LOCALIZATION_CONTRACT,
    HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT,
    HYPOTHESIS_IDS,
    LOCALIZATION_TO_DESIGN_BRIDGE,
    NECESSITY_DIRECTION_CLAIM_BOUNDARY,
    PAPER_METHOD_STAGE_REQUIREMENTS,
    REPORT_SECTION_CONTRACT,
    REQUIRED_NARRATIVES,
    SYSTEM_LAYER_IDS,
    SYSTEM_LAYER_COMPONENTS,
    _bind_opportunity_evidence_identities,
    _build_component_evidence_role_coverage,
    _build_reproducibility_ledger,
    _audit_comprehensive_against_formal,
    _audit_design_gate_payload,
    _build_component_bidirectional_gate_matrix,
    _build_opportunity_lineage_matrix,
    _audit_formal_report,
    _audit_report_section_contract,
    _derive_necessity_component_models,
    build_comprehensive_decision_template,
    build_frozen_observation_snapshot,
    build_prior_mechanism_diagnosis_snapshot,
    populate_registered_component_model_coverage,
    render_comprehensive_report_markdown,
    validate_comprehensive_decisions,
)
from myrec.mechanism.deep_dive_evidence_topology import MODEL_IDS
from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_progress import SELECTED_NODES
from myrec.mechanism.frozen_model_architecture_audit import (
    FORWARD_PRIMITIVE_CONTRACTS,
    TRAINING_PRIMITIVE_CONTRACTS,
)
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_DELIVERABLE_MODEL_COVERAGE,
    COMPONENT_IDS,
    OPPORTUNITY_IDS,
)
from myrec.mechanism.postblock_sweep_evaluator import POSTBLOCK_BLOCKS
from myrec.mechanism.supplemental_evidence_registry import EXPECTED_SUPPLEMENT_IDS
from myrec.mechanism.transformer_interface_inventory import (
    build_transformer_interface_coverage,
)


FORMAL = {
    "d1_representation",
    "d2_selected_branches",
    "d3_attention_edges",
    "d5_context",
    "d6_q2_native_readout",
    "d6_q3_native_readout",
}
SUPPLEMENTS = {
    "d1_activation_anisotropy",
    "component_functional_design_gate_synthesis",
    "component_state_reverse_necessity_v2",
}
SUPPLEMENT_METADATA = {
    "d1_activation_anisotropy": {
        "components": ["layerwise_representation", "normalization"],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
    },
    "component_functional_design_gate_synthesis": {
        "components": [
            "attention_output",
            "mlp_output",
            "residual_composition",
            "history_routing",
        ],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
    },
    "component_state_reverse_necessity_v2": {
        "components": [
            "attention_output",
            "mlp_output",
            "residual_composition",
            "history_routing",
        ],
        "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
    },
}
ROOT = Path(__file__).resolve().parents[1]
NO_CAUSAL_ROLE_COMPONENTS = {
    "token_embedding",
    "mlp_feature_formation",
    "optimizer_effective_update",
    "lora_parameterization",
}


def _readiness_component_coverage_fixture() -> dict:
    rows = []
    for component_id in COMPONENT_IDS:
        registered = component_id not in NO_CAUSAL_ROLE_COMPONENTS
        model_scope = list(MODEL_IDS[2:]) if registered else []
        rows.append(
            {
                "component_id": component_id,
                "any_evidence_artifact_completed": True,
                "causal_role_artifact_registered": registered,
                "causal_role_artifact_completed": registered,
                "q2_q3_causal_role_artifacts_registered": registered,
                "q2_q3_causal_role_artifacts_completed": registered,
                "causal_role_model_scope_registered": model_scope,
                "causal_role_model_scope_completed": model_scope,
            }
        )
    return {
        "component_count": len(COMPONENT_IDS),
        "components_with_any_completed_artifact": len(COMPONENT_IDS),
        "components_with_registered_causal_role_artifact": (
            len(COMPONENT_IDS) - len(NO_CAUSAL_ROLE_COMPONENTS)
        ),
        "components_without_registered_causal_role_artifact": sorted(
            NO_CAUSAL_ROLE_COMPONENTS
        ),
        "components_with_completed_causal_role_artifact": (
            len(COMPONENT_IDS) - len(NO_CAUSAL_ROLE_COMPONENTS)
        ),
        "components_with_completed_q2_q3_causal_role_artifacts": (
            len(COMPONENT_IDS) - len(NO_CAUSAL_ROLE_COMPONENTS)
        ),
        "component_artifact_coverage": rows,
    }


def _formal_localization_profiles_fixture() -> dict:
    endpoints = ("target_margin", "ndcg@10")
    shape_rows = [
        {
            "method_id": model_id,
            "endpoint": endpoint,
            "exact_layer_index_is_architecture_evidence": False,
            "layer_scan_alone_authorizes_design": False,
        }
        for model_id in MODEL_IDS[2:]
        for endpoint in endpoints
    ]
    all_rows = [
        {
            "method_id": model_id,
            "endpoint": endpoint,
            "block_zero_based": block,
            "exact_layer_index_is_architecture_evidence": False,
            "used_as_primary_component_attribution": False,
        }
        for model_id in MODEL_IDS[2:]
        for endpoint in endpoints
        for block in POSTBLOCK_BLOCKS
    ]
    adjacent_rows = [
        {
            "method_id": model_id,
            "endpoint": endpoint,
            "block_zero_based": block,
            "exact_layer_index_is_architecture_evidence": False,
            "used_as_primary_component_attribution": False,
        }
        for model_id in MODEL_IDS[2:]
        for endpoint in endpoints
        for block in POSTBLOCK_BLOCKS[1:]
    ]
    transition_rows = [
        {
            "method_id": model_id,
            "endpoint": endpoint,
            "contrast_id": f"adjacent__{left}__to__{right}",
            "literal_hidden_state_sign_reversal_claimed": False,
            "used_as_primary_component_attribution": False,
        }
        for model_id in MODEL_IDS[2:]
        for endpoint in endpoints
        for left, right in zip(SELECTED_NODES[:-1], SELECTED_NODES[1:])
    ]
    return {
        "layerwise_attenuation_profile": {
            "shape_summary": shape_rows,
            "all_layer_rows": all_rows,
            "adjacent_layer_rows": adjacent_rows,
        },
        "attenuation_transition_profile": {"rows": transition_rows},
    }


def _decisions() -> dict:
    finding_id = "F1"
    value = {
        "schema_version": 1,
        "worksheet_status": "final",
        "report_id": "comprehensive_fixture",
        "narratives": {
            key: {
                "text": f"Narrative for {key}.",
                "evidence_level": "U",
                "supporting_findings": [
                    "F_CROSS" if key == "cross_model_boundary" else finding_id
                ],
                "do_not_infer": ["Do not infer evidence beyond the cited findings."],
            }
            for key in REQUIRED_NARRATIVES
        },
        "findings": [
            {
                "finding_id": finding_id,
                "title": "A bounded finding",
                "evidence_level": "D",
                "claim": "The admitted descriptive evidence constrains one hypothesis.",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "dataset_scope": "kuaisearch_dev",
                "supporting_formal_deliverables": ["d1_representation"],
                "supporting_supplements": ["d1_activation_anisotropy"],
                "contradictory_evidence": [],
                "do_not_infer": ["Do not infer component necessity."],
            },
            {
                "finding_id": "F_NEG_BOUNDARY",
                "title": "A stable descriptive design boundary",
                "evidence_level": "D",
                "claim": "Exact internal indices remain descriptive metadata.",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "dataset_scope": "kuaisearch_dev",
                "supporting_formal_deliverables": ["d1_representation"],
                "supporting_supplements": ["d1_activation_anisotropy"],
                "contradictory_evidence": [],
                "do_not_infer": ["Do not infer an exact-layer architecture."],
            },
            {
                "finding_id": "F_CROSS",
                "title": "A bounded cross-model finding",
                "evidence_level": "D",
                "claim": "A descriptive cross-model comparison remains bounded.",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "dataset_scope": "kuaisearch_dev",
                "supporting_formal_deliverables": ["d1_representation"],
                "supporting_supplements": ["d1_activation_anisotropy"],
                "contradictory_evidence": [],
                "do_not_infer": ["Do not infer cross-dataset replication."],
            },
            {
                "finding_id": "F_OPPORTUNITY",
                "title": "A bounded opportunity finding",
                "evidence_level": "D",
                "claim": "Representation evidence supports a diagnostic candidate only.",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "dataset_scope": "kuaisearch_dev",
                "supporting_formal_deliverables": [
                    "d1_representation",
                    "d2_selected_branches",
                    "d5_context",
                ],
                "supporting_supplements": ["d1_activation_anisotropy"],
                "contradictory_evidence": [],
                "do_not_infer": ["Do not infer utility or component necessity."],
            },
        ],
        "component_matrix": {
            component: {
                "status": "unresolved",
                "evidence_level": "U",
                "summary": f"{component} remains bounded.",
                "model_scope": [],
                "supporting_findings": [],
                "remaining_uncertainty": "No design-qualified causal chain yet.",
            }
            for component in COMPONENT_IDS
        },
        "component_model_matrix": {
            component: {
                model: {
                    "status": "unresolved",
                    "evidence_level": "U",
                    "summary": f"{component}/{model} remains bounded.",
                    "supporting_findings": [],
                    "remaining_uncertainty": (
                        "No model-specific causal resolution yet."
                    ),
                }
                for model in MODEL_IDS
            }
            for component in COMPONENT_IDS
        },
        "functional_causal_chain": [
            {
                "node": node,
                "evidence_level": "U",
                "status": "unresolved",
                "model_scope": [],
                "diagnosis": f"{node} is unresolved.",
                "supporting_findings": [],
            }
            for node in CAUSAL_CHAIN_NODES
        ],
        "failure_mode_diagnosis": {
            "primary_mode": "unresolved",
            "evidence_level": "U",
            "summary": "The causal failure mode remains unresolved.",
            "functional_components": [],
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": [finding_id],
            "competing_modes": [
                {
                    "mode": "candidate_transport_failure",
                    "reason_remaining": "Bidirectional component gates are pending.",
                },
                {
                    "mode": "state_present_but_readout_misaligned",
                    "reason_remaining": "Native readout necessity is not established.",
                },
            ],
            "causal_erasure_claim_authorized": False,
            "causal_loss_of_use_claim_authorized": False,
            "exact_layer_index_used_for_design": False,
            "falsification_gate": "Resolve only after the registered S/N/G chain closes.",
        },
        "system_layers": {
            layer: {
                "status": "unresolved",
                "evidence_level": "U",
                "model_scope": [],
                "diagnosis": f"{layer} diagnosis.",
                "supporting_findings": [],
                "remaining_uncertainty": f"{layer} uncertainty.",
            }
            for layer in SYSTEM_LAYER_IDS
        },
        "model_boundaries": {
            model: {
                "summary": f"{model} bounded summary.",
                "supporting_findings": (
                    [finding_id, "F_CROSS"] if model in MODEL_IDS[2:] else []
                ),
                "uncovered_components": [],
                "do_not_generalize": "No scale or dataset generalization.",
            }
            for model in MODEL_IDS
        },
        "cross_model_synthesis": {
            "shared_patterns": [
                {
                    "pattern_id": "shared_bounded_pattern",
                    "evidence_level": "D",
                    "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                    "functional_components": ["layerwise_representation"],
                    "summary": "A bounded functional pattern is shared.",
                    "supporting_findings": ["F_CROSS"],
                    "do_not_generalize": "Do not generalize beyond Q2 and Q3.",
                }
            ],
            "heterogeneous_patterns": [
                {
                    "pattern_id": "heterogeneous_bounded_pattern",
                    "evidence_level": "D",
                    "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                    "functional_components": ["layerwise_representation"],
                    "summary": "A bounded functional response remains heterogeneous.",
                    "supporting_findings": ["F_CROSS"],
                    "do_not_generalize": "Do not infer a universal model response.",
                }
            ],
            "remaining_uncertainty": "Independent model scales remain untested.",
            "absolute_index_alignment_used": False,
        },
        "hypothesis_matrix": {
            hypothesis: {
                "status": "unresolved",
                "evidence_level": "U",
                "summary": f"{hypothesis} remains unresolved.",
                "supporting_findings": ["F_NEG_BOUNDARY"],
                "negative_evidence_basis": ["insufficient_causal_evidence"],
                "contradictory_evidence": ["A registered counter-pattern remains."],
                "remaining_uncertainty": "A bidirectional causal gate is required.",
            }
            for hypothesis in HYPOTHESIS_IDS
        },
        "negative_and_conflicting_results": [
            {
                "result_id": "NEG1",
                "summary": "One registered negative result is retained.",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "endpoint_scope": ["target_margin"],
                "surface_scope": ["strict_transfer"],
                "contrast_scope": ["full_minus_null"],
                "fold_scope": ["fold1"],
                "seed_scope": ["20260714"],
                "supporting_findings": [finding_id, "F_OPPORTUNITY"],
                "interpretation_boundary": "Absence of support is not absence of signal.",
            }
        ],
        "evidence_disposition": {
            evidence_id: {
                "evidence_kind": (
                    "formal_deliverable"
                    if evidence_id in FORMAL
                    else "supplement"
                ),
                "disposition": (
                    "interpreted_in_findings"
                    if evidence_id
                    in {
                        "d1_representation",
                        "d1_activation_anisotropy",
                        "d2_selected_branches",
                        "d5_context",
                    }
                    else "bounded_no_scientific_claim"
                ),
                "supporting_findings": (
                    ["F_OPPORTUNITY"]
                    if evidence_id in {"d2_selected_branches", "d5_context"}
                    else [finding_id]
                    if evidence_id
                    in {"d1_representation", "d1_activation_anisotropy"}
                    else []
                ),
                "summary": f"{evidence_id} receives an explicit bounded disposition.",
                "do_not_infer": [
                    "Do not infer a claim beyond this evidence disposition."
                ],
            }
            for evidence_id in sorted(FORMAL | SUPPLEMENTS)
        },
        "optimization_opportunities": [
            {
                "opportunity_id": "representation_continuity_probe",
                "formal_predecessor_ids": list(OPPORTUNITY_IDS),
                "functional_component": "layerwise_representation",
                "functional_node": None,
                "mechanism_target": "history-conditioned representation continuity",
                "minimum_evidence_level": "N",
                "actual_evidence_level": "D",
                "supporting_formal_deliverables": [
                    "d1_representation",
                    "d2_selected_branches",
                    "d5_context",
                ],
                "supporting_supplements": ["d1_activation_anisotropy"],
                "supporting_findings": ["F_OPPORTUNITY"],
                "contradictory_evidence": ["No component necessity is established."],
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "dataset_scope": "kuaisearch_dev",
                "source_test_opened": False,
                "utility_gain_established": False,
                "design_priority": "candidate_to_test",
                "intervention_polarity": "diagnostic_only",
                "reason": "Descriptive evidence motivates a falsification test only.",
                "expected_benefit": (
                    "A cleaner falsification target for representation continuity."
                ),
                "hypothesized_innovation": (
                    "A functional continuity constraint tied to transferable history state."
                ),
                "training_signal_requirements": [
                    "Train-only different-item preference pairs with query-matched negatives."
                ],
                "key_ablations": [
                    "Remove the continuity constraint while preserving parameter count.",
                    "Replace query-matched negatives with random negatives.",
                ],
                "closest_baseline_families": [
                    "Frozen Q0--Q3 LLM4Rec pathways",
                    "CoPPS-style structural transfer witness",
                ],
                "baseline_differentiation": (
                    "Test whether the functional constraint adds transfer behavior beyond "
                    "input-only and structural-witness controls."
                ),
                "key_risks": [
                    "The descriptive pattern may not survive a causal intervention."
                ],
                "falsification_gate": "Reject if reverse removal and specificity fail.",
                "do_not_infer": ["Do not infer ranking improvement."],
                "diagnostic_patch_promoted_as_method": False,
                "architecture_implemented": False,
            }
        ],
        "not_recommended": [
            {
                "direction": "Exact-layer adapter",
                "formal_predecessor_ids": [],
                "functional_component": "layerwise_representation",
                "reason": "Layer identity is localization metadata.",
                "supporting_findings": ["F_NEG_BOUNDARY"],
                "basis": "descriptive_only",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "dataset_scope": "kuaisearch_dev",
                "source_test_opened": False,
            }
        ],
    }
    for component_id in COMPONENT_IDS:
        for model_id in MODEL_IDS:
            formally_covered = any(
                deliverable in COMPONENT_ALLOWED_DELIVERABLES[component_id]
                and model_id
                in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id].get(
                    deliverable, set()
                )
                for deliverable in FORMAL
            )
            supplement_covered = any(
                component_id in metadata["components"]
                and model_id in metadata["model_scope"]
                for metadata in SUPPLEMENT_METADATA.values()
            )
            if not formally_covered and not supplement_covered:
                value["component_model_matrix"][component_id][model_id].update(
                    {"status": "untested", "evidence_level": "U"}
                )
                value["model_boundaries"][model_id]["uncovered_components"].append(
                    component_id
                )
        if all(
            value["component_model_matrix"][component_id][model_id]["status"]
            == "untested"
            for model_id in MODEL_IDS
        ):
            value["component_matrix"][component_id].update(
                {"status": "untested", "evidence_level": "U"}
            )
    return value


def _validate(
    value: dict,
    *,
    design_nodes: set[str] | None = None,
    necessity_models: dict[str, set[str]] | None = None,
) -> dict:
    value = copy.deepcopy(value)
    # Most tests mutate a finding's evidence to exercise a downstream scientific
    # gate. Keep the fixture-only accounting ledger aligned so those tests fail
    # at the intended gate rather than at evidence-disposition bookkeeping.
    disposition = value.get("evidence_disposition", {})
    if isinstance(disposition, dict):
        for row in disposition.values():
            row["disposition"] = "bounded_no_scientific_claim"
            row["supporting_findings"] = []
        for finding in value.get("findings", []):
            finding_id = finding.get("finding_id")
            for evidence_id in [
                *finding.get("supporting_formal_deliverables", []),
                *finding.get("supporting_supplements", []),
            ]:
                if evidence_id not in disposition:
                    continue
                row = disposition[evidence_id]
                row["disposition"] = "interpreted_in_findings"
                if finding_id not in row["supporting_findings"]:
                    row["supporting_findings"].append(finding_id)
    for component_id, aggregate in value["component_matrix"].items():
        if aggregate["status"] not in {
            "supported",
            "weakened",
            "mechanical_failure",
        }:
            continue
        for model_id in aggregate["model_scope"]:
            value["component_model_matrix"][component_id][model_id].update(
                {
                    "status": aggregate["status"],
                    "evidence_level": aggregate["evidence_level"],
                    "supporting_findings": list(
                        aggregate["supporting_findings"]
                    ),
                }
            )
    return _validate_raw(
        value,
        design_nodes=design_nodes,
        necessity_models=necessity_models,
    )


def _validate_raw(
    value: dict,
    *,
    design_nodes: set[str] | None = None,
    necessity_models: dict[str, set[str]] | None = None,
) -> dict:
    necessity_support = {component_id: set() for component_id in COMPONENT_IDS}
    for component_id, model_scope in (necessity_models or {}).items():
        necessity_support[component_id] = set(model_scope)
    return validate_comprehensive_decisions(
        value,
        completed_formal=set(FORMAL),
        completed_supplements=set(SUPPLEMENTS),
        supplement_metadata=SUPPLEMENT_METADATA,
        design_qualified_nodes=set(design_nodes or set()),
        necessity_supported_component_models=necessity_support,
    )


def _formal_outcomes_fixture() -> dict:
    return {
        "component_evidence_matrix": [
            {
                "component_id": component_id,
                "status": "unresolved",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "evidence_deliverables": ["d1_representation"],
            }
            for component_id in COMPONENT_IDS
        ],
        "hypothesis_status_matrix": [
            {"hypothesis_id": hypothesis_id, "status": "unresolved"}
            for hypothesis_id in HYPOTHESIS_IDS
        ],
    }


def test_valid_decisions_cover_all_required_axes() -> None:
    normalized = _validate(_decisions())
    assert set(normalized["component_matrix"]) == set(COMPONENT_IDS)
    assert set(normalized["hypothesis_matrix"]) == set(HYPOTHESIS_IDS)
    assert [row["node"] for row in normalized["functional_causal_chain"]] == list(
        CAUSAL_CHAIN_NODES
    )
    assert normalized["optimization_opportunities"][0]["design_priority"] == (
        "candidate_to_test"
    )
    assert normalized["optimization_opportunities"][0]["rank"] == 1
    assert set(normalized["formal_opportunity_disposition"]) == set(OPPORTUNITY_IDS)
    assert set(normalized["narratives"]) == set(REQUIRED_NARRATIVES)
    assert all(
        row["supporting_findings"] and row["do_not_infer"]
        for row in normalized["narratives"].values()
    )
    assert set(normalized["evidence_disposition"]) == FORMAL | SUPPLEMENTS
    assert all(
        row["evidence_id"] == evidence_id
        for evidence_id, row in normalized["evidence_disposition"].items()
    )
    assert normalized["component_matrix"]["native_readout"][
        "mechanism_question"
    ] == COMPONENT_FUNCTIONAL_QUESTIONS["native_readout"]
    assert "frozen Q2 yes/no" in normalized["component_matrix"]["native_readout"][
        "claim_boundary"
    ]
    assert "RMSNorm operator necessity" in next(
        row["claim_boundary"]
        for row in normalized["functional_causal_chain"]
        if row["node"] == "final_norm"
    )


def test_free_text_narratives_are_finding_bound_and_cannot_hand_copy_metrics() -> None:
    decisions = _decisions()
    decisions["narratives"]["executive_summary"]["supporting_findings"] = []
    with pytest.raises(ValueError, match="narrative lacks supporting findings"):
        _validate(decisions)

    decisions = _decisions()
    decisions["narratives"]["executive_summary"]["evidence_level"] = "S"
    with pytest.raises(ValueError, match="lacks a matching finding"):
        _validate(decisions)

    decisions = _decisions()
    decisions["narratives"]["executive_summary"]["text"] = "NDCG@10=0.123"
    with pytest.raises(ValueError, match="cannot hand-copy a metric literal"):
        _validate(decisions)

    decisions = _decisions()
    decisions["narratives"]["cross_model_boundary"]["text"] = (
        "The models align at layer 19."
    )
    with pytest.raises(ValueError, match="absolute internal index forbidden"):
        _validate(decisions)

    decisions = _decisions()
    decisions["narratives"]["layer_trajectory_interpretation"]["text"] = (
        "Layer 19 is retained only as trajectory lineage."
    )
    normalized = _validate(decisions)
    assert "Layer 19" in normalized["narratives"][
        "layer_trajectory_interpretation"
    ]["text"]


def test_all_human_interpretation_fields_use_tables_for_metrics_and_no_indices() -> None:
    decisions = _decisions()
    decisions["findings"][0]["claim"] = "The admitted NDCG@10=0.123."
    with pytest.raises(ValueError, match="cannot hand-copy a metric literal"):
        _validate(decisions)

    decisions = _decisions()
    decisions["component_matrix"]["layerwise_representation"]["summary"] = (
        "The design target is layer 19."
    )
    with pytest.raises(
        ValueError, match="allowed only in trajectory lineage"
    ):
        _validate(decisions)

    decisions = _decisions()
    decisions["hypothesis_matrix"]["H0"]["remaining_uncertainty"] = (
        "CI95: 0.01 remains uncertain."
    )
    with pytest.raises(ValueError, match="cannot hand-copy a metric literal"):
        _validate(decisions)


def test_formal_report_admission_preserves_layer_scan_erasure_boundary() -> None:
    payload = {
        "analysis_type": "transformer_deep_dive_mechanism_closeout",
        "status": "completed",
        "evidence_admission": {
            "source_test_content_read_by_closeout_audit": False,
        },
        "execution_census": {"completed_deliverables": 19},
        "primary_loss_attribution": [
            {
                "method_id": model_id,
                "component_erasure_boundary_established": False,
                "history_token_flow_directly_observed_by_layer_scan": False,
                "exact_layer_index_is_architecture_evidence": False,
            }
            for model_id in (MODEL_IDS[2], MODEL_IDS[3])
        ],
        **_formal_localization_profiles_fixture(),
    }
    _audit_formal_report(payload)
    payload["primary_loss_attribution"][0][
        "component_erasure_boundary_established"
    ] = True
    with pytest.raises(ValueError, match="overstates the layer-scan erasure"):
        _audit_formal_report(payload)


def test_formal_report_admission_requires_every_localization_row() -> None:
    payload = {
        "analysis_type": "transformer_deep_dive_mechanism_closeout",
        "status": "completed",
        "evidence_admission": {
            "source_test_content_read_by_closeout_audit": False,
        },
        "execution_census": {"completed_deliverables": 19},
        "primary_loss_attribution": [
            {
                "method_id": model_id,
                "component_erasure_boundary_established": False,
                "history_token_flow_directly_observed_by_layer_scan": False,
                "exact_layer_index_is_architecture_evidence": False,
            }
            for model_id in (MODEL_IDS[2], MODEL_IDS[3])
        ],
        **_formal_localization_profiles_fixture(),
    }

    _audit_formal_report(payload)
    payload["layerwise_attenuation_profile"]["all_layer_rows"].pop()
    with pytest.raises(ValueError, match="all-layer profile coverage differs"):
        _audit_formal_report(payload)

    payload = {
        **payload,
        **_formal_localization_profiles_fixture(),
    }
    payload["attenuation_transition_profile"]["rows"][0][
        "literal_hidden_state_sign_reversal_claimed"
    ] = True
    with pytest.raises(ValueError, match="overstates causality"):
        _audit_formal_report(payload)


def test_comprehensive_s_level_component_cannot_upgrade_formal_outcome() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    decisions["component_matrix"]["attention_query_key_routing"].update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    normalized = _validate(decisions)
    formal = _formal_outcomes_fixture()
    with pytest.raises(ValueError, match="changes the formal outcome"):
        _audit_comprehensive_against_formal(normalized, formal)

    formal_attention = next(
        row
        for row in formal["component_evidence_matrix"]
        if row["component_id"] == "attention_query_key_routing"
    )
    formal_attention.update(
        {"status": "supported", "evidence_deliverables": ["d3_attention_edges"]}
    )
    _audit_comprehensive_against_formal(normalized, formal)

    formal_attention["model_scope"] = [MODEL_IDS[2]]
    with pytest.raises(ValueError, match="exceeds the formal model scope"):
        _audit_comprehensive_against_formal(normalized, formal)


def test_weakened_component_must_match_registered_formal_outcome() -> None:
    decisions = _decisions()
    embedding = decisions["component_matrix"]["token_embedding"]
    embedding.update(
        {
            "status": "weakened",
            "evidence_level": "D",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    normalized = _validate(decisions)
    formal = _formal_outcomes_fixture()
    with pytest.raises(ValueError, match="changes the formal outcome"):
        _audit_comprehensive_against_formal(normalized, formal)

    formal_embedding = next(
        row
        for row in formal["component_evidence_matrix"]
        if row["component_id"] == "token_embedding"
    )
    formal_embedding["status"] = "weakened"
    _audit_comprehensive_against_formal(normalized, formal)

    formal_embedding["model_scope"] = [MODEL_IDS[2]]
    with pytest.raises(ValueError, match="exceeds the formal model scope"):
        _audit_comprehensive_against_formal(normalized, formal)


def test_negative_section_must_cover_every_weakened_or_rejected_outcome() -> None:
    decisions = _decisions()
    decisions["component_matrix"]["token_embedding"].update(
        {
            "status": "weakened",
            "evidence_level": "D",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    decisions["negative_and_conflicting_results"][0][
        "supporting_findings"
    ] = ["F_NEG_BOUNDARY"]
    with pytest.raises(ValueError, match="weakened component is missing"):
        _validate(decisions)

    decisions = _decisions()
    decisions["negative_and_conflicting_results"][0]["supporting_findings"] = []
    with pytest.raises(ValueError, match="lacks supporting findings"):
        _validate(decisions)


def test_supported_aggregate_cannot_hide_a_model_scoped_weakened_cell() -> None:
    decisions = _decisions()
    decisions["component_matrix"]["token_embedding"].update(
        {
            "status": "weakened",
            "evidence_level": "D",
            "model_scope": [MODEL_IDS[2]],
            "supporting_findings": ["F1"],
        }
    )
    decisions["negative_and_conflicting_results"][0]["model_scope"] = [
        MODEL_IDS[3]
    ]
    with pytest.raises(
        ValueError, match="weakened component-model cell is missing"
    ):
        _validate(decisions)

    decisions["negative_and_conflicting_results"][0]["model_scope"] = [
        MODEL_IDS[2]
    ]
    normalized = _validate(decisions)
    assert normalized["component_model_matrix"]["token_embedding"][MODEL_IDS[2]][
        "status"
    ] == "weakened"


def test_negative_rows_retain_registered_model_endpoint_surface_and_seed_scope() -> None:
    decisions = _decisions()
    decisions["negative_and_conflicting_results"][0].pop("endpoint_scope")
    with pytest.raises(ValueError, match="negative endpoint_scope"):
        _validate(decisions)

    decisions = _decisions()
    decisions["negative_and_conflicting_results"][0]["surface_scope"] = [
        "best-looking-slice"
    ]
    with pytest.raises(ValueError, match="unknown values"):
        _validate(decisions)

    decisions = _decisions()
    decisions["negative_and_conflicting_results"][0]["model_scope"] = [
        MODEL_IDS[0]
    ]
    with pytest.raises(ValueError, match="exceeds finding model scope"):
        _validate(decisions)

    decisions = _decisions()
    decisions["negative_and_conflicting_results"][0]["seed_scope"] = [
        "20260714",
        "20260714",
    ]
    with pytest.raises(ValueError, match="seed_scope contains duplicates"):
        _validate(decisions)


def test_hypothesis_negative_evidence_basis_is_structured_and_status_safe() -> None:
    decisions = _decisions()
    decisions["hypothesis_matrix"]["H0"]["negative_evidence_basis"] = [
        "best-looking-explanation"
    ]
    with pytest.raises(ValueError, match="unknown values"):
        _validate(decisions)

    decisions = _decisions()
    decisions["hypothesis_matrix"]["H0"]["negative_evidence_basis"] = [
        "registered_refutation"
    ]
    with pytest.raises(ValueError, match="non-rejected hypothesis"):
        _validate(decisions)


def test_weakened_component_cannot_borrow_unrelated_or_unscoped_evidence() -> None:
    decisions = _decisions()
    attention = decisions["component_matrix"]["attention_output"]
    attention.update(
        {
            "status": "weakened",
            "evidence_level": "D",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="level/component/model-matched"):
        _validate(decisions)

    decisions = _decisions()
    embedding = decisions["component_matrix"]["token_embedding"]
    embedding.update(
        {
            "status": "weakened",
            "evidence_level": "D",
            "model_scope": list(MODEL_IDS),
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="level/component/model-matched"):
        _validate(decisions)


def test_template_is_exhaustive_and_deliberately_non_final() -> None:
    template = build_comprehensive_decision_template(report_id="template_fixture")
    assert template["worksheet_status"] == "incomplete"
    assert set(template["component_matrix"]) == set(COMPONENT_IDS)
    assert set(template["component_model_matrix"]) == set(COMPONENT_IDS)
    assert all(
        set(rows) == set(MODEL_IDS)
        for rows in template["component_model_matrix"].values()
    )
    assert set(template["hypothesis_matrix"]) == set(HYPOTHESIS_IDS)
    assert set(template["system_layers"]) == set(SYSTEM_LAYER_IDS)
    assert set(template["model_boundaries"]) == set(MODEL_IDS)
    assert set(template["cross_model_synthesis"]) == {
        "shared_patterns",
        "heterogeneous_patterns",
        "remaining_uncertainty",
        "absolute_index_alignment_used",
    }
    assert [row["node"] for row in template["functional_causal_chain"]] == list(
        CAUSAL_CHAIN_NODES
    )
    with pytest.raises(ValueError, match="worksheet_status must be final"):
        _validate(template)


def test_registered_coverage_prefill_is_complete_blind_and_non_mutating() -> None:
    template = build_comprehensive_decision_template(report_id="coverage_fixture")
    original = copy.deepcopy(template)

    populated = populate_registered_component_model_coverage(
        template,
        registered_formal=set(EXPECTED_DELIVERABLES),
        registered_supplements=SUPPLEMENT_METADATA,
    )

    assert template == original
    assert populated["template_instructions"][
        "component_model_coverage_prefilled_from_registry"
    ] is True
    assert populated["template_instructions"][
        "scientific_status_inferred_during_prefill"
    ] is False
    for model_id in MODEL_IDS:
        expected_uncovered = [
            component_id
            for component_id in COMPONENT_IDS
            if populated["component_model_matrix"][component_id][model_id][
                "status"
            ]
            == "untested"
        ]
        assert populated["model_boundaries"][model_id][
            "uncovered_components"
        ] == expected_uncovered
    for component_id in COMPONENT_IDS:
        statuses = {
            populated["component_model_matrix"][component_id][model_id][
                "status"
            ]
            for model_id in MODEL_IDS
        }
        assert statuses <= {"unresolved", "untested"}
        expected_aggregate = (
            "untested" if statuses == {"untested"} else "unresolved"
        )
        assert populated["component_matrix"][component_id]["status"] == (
            expected_aggregate
        )


def test_component_model_matrix_requires_all_18_by_4_cells() -> None:
    decisions = _decisions()
    decisions["component_model_matrix"][COMPONENT_IDS[0]].pop(MODEL_IDS[0])
    with pytest.raises(ValueError, match=f"{COMPONENT_IDS[0]} coverage drift"):
        _validate_raw(decisions)


def test_cross_model_patterns_require_functional_multi_model_evidence() -> None:
    decisions = _decisions()
    decisions["model_boundaries"][MODEL_IDS[2]]["supporting_findings"] = ["F1"]
    with pytest.raises(ValueError, match="absent from its model boundary"):
        _validate(decisions)

    decisions = _decisions()
    decisions["cross_model_synthesis"]["shared_patterns"][0][
        "model_scope"
    ] = [MODEL_IDS[2]]
    with pytest.raises(ValueError, match="fewer than two models"):
        _validate(decisions)

    decisions = _decisions()
    decisions["cross_model_synthesis"]["shared_patterns"][0][
        "evidence_level"
    ] = "S"
    with pytest.raises(
        ValueError, match="component/model/level-matched evidence"
    ):
        _validate(decisions)

    decisions = _decisions()
    decisions["cross_model_synthesis"]["shared_patterns"][0][
        "functional_components"
    ] = ["native_readout"]
    with pytest.raises(
        ValueError, match="component/model/level-matched evidence"
    ):
        _validate(decisions)

    decisions = _decisions()
    decisions["cross_model_synthesis"]["heterogeneous_patterns"][0][
        "summary"
    ] = "The behavior differs at block 20."
    with pytest.raises(ValueError, match="absolute internal index forbidden"):
        _validate(decisions)

    decisions = _decisions()
    cross_finding = next(
        row for row in decisions["findings"] if row["finding_id"] == "F_CROSS"
    )
    cross_finding["evidence_level"] = "S"
    decisions["cross_model_synthesis"]["shared_patterns"][0][
        "evidence_level"
    ] = "S"
    with pytest.raises(ValueError, match="conflicts with its component cell"):
        _validate(decisions)


def test_aggregate_component_claim_must_match_each_scoped_model_cell() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    decisions["component_matrix"]["attention_query_key_routing"].update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="status differs from its 18x4 cells"):
        _validate_raw(decisions)

    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    for model_id in MODEL_IDS[2:]:
        decisions["component_model_matrix"]["attention_query_key_routing"][
            model_id
        ].update(
            {
                "status": "supported",
                "evidence_level": "S",
                "supporting_findings": ["F1"],
            }
        )
    with pytest.raises(ValueError, match="status differs from its 18x4 cells"):
        _validate_raw(decisions)


def test_component_model_claim_requires_component_and_model_matched_evidence() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    decisions["component_model_matrix"]["native_readout"][
        MODEL_IDS[2]
    ].update(
        {
            "status": "supported",
            "evidence_level": "S",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="level/component/model-matched"):
        _validate_raw(decisions)


def test_unresolved_component_rows_cannot_cite_out_of_scope_evidence() -> None:
    decisions = _decisions()
    decisions["component_model_matrix"]["native_readout"][MODEL_IDS[2]][
        "supporting_findings"
    ] = ["F1"]
    with pytest.raises(ValueError, match="cell cites evidence outside its scope"):
        _validate_raw(decisions)

    decisions = _decisions()
    decisions["component_matrix"]["native_readout"].update(
        {"model_scope": [MODEL_IDS[0]], "supporting_findings": ["F1"]}
    )
    with pytest.raises(ValueError, match="outside its component/model scope"):
        _validate_raw(decisions)


def test_opportunity_evidence_references_bind_to_audited_file_sha() -> None:
    normalized = _validate(_decisions())
    formal = {
        "evidence_admission": {
            "deliverables": {
                evidence_id: {
                    "status": "completed",
                    "path": f"runs/formal/{evidence_id}/metrics.json",
                    "sha256": "a" * 64,
                }
                for evidence_id in FORMAL
            }
        }
    }
    supplements = {
        evidence_id: {
            "status": "completed",
            "path": f"runs/supplement/{evidence_id}/metrics.json",
            "sha256": "b" * 64,
        }
        for evidence_id in SUPPLEMENTS
    }
    bound = _bind_opportunity_evidence_identities(
        normalized,
        formal=formal,
        completed_supplements=supplements,
    )
    identities = bound["optimization_opportunities"][0][
        "supporting_evidence_identities"
    ]
    assert [(row["evidence_id"], row["sha256"]) for row in identities] == [
        ("d1_representation", "a" * 64),
        ("d2_selected_branches", "a" * 64),
        ("d5_context", "a" * 64),
        ("d1_activation_anisotropy", "b" * 64),
    ]
    assert "supporting_evidence_identities" not in normalized[
        "optimization_opportunities"
    ][0]
    finding_identities = bound["findings"][0]["supporting_evidence_identities"]
    assert [(row["evidence_id"], row["sha256"]) for row in finding_identities] == [
        ("d1_representation", "a" * 64),
        ("d1_activation_anisotropy", "b" * 64),
    ]
    assert "supporting_evidence_identities" not in normalized["findings"][0]
    narrative_identities = bound["narratives"]["executive_summary"][
        "supporting_evidence_identities"
    ]
    assert [(row["evidence_id"], row["sha256"]) for row in narrative_identities] == [
        ("d1_representation", "a" * 64),
        ("d1_activation_anisotropy", "b" * 64),
    ]
    assert "supporting_evidence_identities" not in normalized["narratives"][
        "executive_summary"
    ]
    assert set(bound["evidence_disposition"]) == FORMAL | SUPPLEMENTS
    assert all(
        row["evidence_identity"]["evidence_id"] == evidence_id
        for evidence_id, row in bound["evidence_disposition"].items()
    )

    supplements["d1_activation_anisotropy"]["sha256"] = "forged"
    with pytest.raises(ValueError, match="invalid evidence SHA-256"):
        _bind_opportunity_evidence_identities(
            normalized,
            formal=formal,
            completed_supplements=supplements,
        )


def test_reproducibility_ledger_preserves_run_commands_and_all_file_identities() -> None:
    formal = {
        "evidence_admission": {
            "frozen_assets": {"plan.md": "a" * 64},
            "deliverables": {
                evidence_id: {
                    "status": "completed",
                    "path": f"runs/{evidence_id}/metrics.json",
                    "sha256": "b" * 64,
                }
                for evidence_id in EXPECTED_DELIVERABLES
            },
            "run_declarations": [
                {
                    "run_id": "run-1",
                    "path": "runs/run-1/metadata.json",
                    "sha256": "c" * 64,
                    "analysis_stage": "attention",
                    "method_id": MODEL_IDS[2],
                    "status": "completed",
                    "result_eligible": True,
                    "command": ["python", "score.py", "--fold", "1"],
                }
            ],
            "dev_eval_ledger": {
                "path": "reports/dev_eval_log.jsonl",
                "sha256": "d" * 64,
                "entry_count": 19,
            },
            "source_test_content_read_by_closeout_audit": False,
            "qrels_content_read_by_closeout_audit": False,
        }
    }
    supplements = {
        "supp-1": {
            "status": "completed",
            "path": "runs/supp-1/metrics.json",
            "sha256": "e" * 64,
            "command": ["python", "analyze_supplement.py"],
        }
    }
    ledger = _build_reproducibility_ledger(
        formal=formal, completed_supplements=supplements
    )
    assert len(ledger["formal_deliverables"]) == len(EXPECTED_DELIVERABLES)
    assert any(
        row["path"] == COMPREHENSIVE_REPORT_PLAN_IDENTITY["path"]
        and row["sha256"] == COMPREHENSIVE_REPORT_PLAN_IDENTITY["sha256"]
        for row in ledger["frozen_assets"]
    )
    frozen_paths = {row["path"] for row in ledger["frozen_assets"]}
    assert {
        identity["path"] for identity in FROZEN_OBSERVATION_EVIDENCE_IDENTITIES
    } <= frozen_paths
    assert ledger["run_declarations"][0]["command"] == [
        "python",
        "score.py",
        "--fold",
        "1",
    ]
    assert ledger["run_declarations"][0]["sha256"] == "c" * 64

    formal["evidence_admission"]["run_declarations"][0]["command"] = None
    with pytest.raises(ValueError, match="completed formal run lacks its command"):
        _build_reproducibility_ledger(
            formal=formal, completed_supplements=supplements
        )


def test_reproducibility_ledger_rejects_frozen_observation_byte_drift(
    tmp_path, monkeypatch
) -> None:
    source = tmp_path / "frozen.json"
    source.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        builder_module,
        "FROZEN_OBSERVATION_EVIDENCE_IDENTITIES",
        (
            {
                "evidence_id": "frozen-source",
                "evidence_kind": "frozen_observation_source",
                "path": "frozen.json",
                "sha256": "0" * 64,
            },
        ),
    )
    formal = {
        "evidence_admission": {
            "frozen_assets": {"plan.md": "a" * 64},
            "deliverables": {
                evidence_id: {
                    "status": "completed",
                    "path": f"runs/{evidence_id}/metrics.json",
                    "sha256": "b" * 64,
                }
                for evidence_id in EXPECTED_DELIVERABLES
            },
            "run_declarations": [
                {
                    "run_id": "run-1",
                    "path": "runs/run-1/metadata.json",
                    "sha256": "c" * 64,
                    "analysis_stage": "attention",
                    "method_id": MODEL_IDS[2],
                    "status": "completed",
                    "result_eligible": True,
                    "command": ["python", "score.py"],
                }
            ],
            "dev_eval_ledger": {
                "path": "reports/dev_eval_log.jsonl",
                "sha256": "d" * 64,
            },
        }
    }
    supplements = {
        "supp-1": {
            "status": "completed",
            "path": "runs/supp-1/metrics.json",
            "sha256": "e" * 64,
            "command": ["python", "analyze.py"],
        }
    }
    monkeypatch.setattr(
        builder_module,
        "COMPREHENSIVE_REPORT_PLAN_IDENTITY",
        {"path": "frozen.json", "sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="repository evidence SHA-256 drift"):
        _build_reproducibility_ledger(
            formal=formal,
            completed_supplements=supplements,
            root=tmp_path,
        )


def test_frozen_observation_snapshot_is_machine_derived_and_bounded() -> None:
    snapshot = build_frozen_observation_snapshot(ROOT)
    assert [row["method_id"] for row in snapshot["methods"]] == list(MODEL_IDS)
    assert snapshot["surface_counts"]["all"] == 4000
    assert snapshot["pilot_seed"] == 20260714
    assert snapshot["second_seed_run"] is False
    assert snapshot["source_test_opened"] is False
    assert snapshot["qrels_or_score_bundles_opened_by_this_snapshot"] is False
    assert all(snapshot["claim_checks"].values())
    assert all(
        row["evaluator_evidence"]["pre_qrels_audit_passed"] is True
        for row in snapshot["methods"]
    )
    assert all(
        len(row["evaluator_evidence"]["shared_evaluator_artifacts"]) == 4
        and row["evaluator_evidence"]["pre_qrels_score_bundle_audit"]["sha256"]
        and row["evaluator_evidence"]["qrels_hash_lock"]["sha256"]
        and row["evaluator_evidence"]["confirmation_records"]["sha256"]
        for row in snapshot["methods"]
    )


def test_prior_mechanism_snapshot_retains_h0_h5_and_all_artifact_bytes() -> None:
    snapshot = build_prior_mechanism_diagnosis_snapshot(ROOT)
    assert [
        (row["hypothesis_id"], row["status"])
        for row in snapshot["hypothesis_status_matrix"]
    ] == [
        ("H0", "unresolved"),
        ("H1", "weakened"),
        ("H2", "weakened"),
        ("H3", "weakened"),
        ("H4", "weakened"),
        ("H5", "unresolved"),
    ]
    assert len(snapshot["artifact_registry"]) == 18
    assert len(snapshot["evidence_index"]) == 11
    assert {
        artifact_id
        for row in snapshot["evidence_index"]
        for artifact_id in row["artifact_ids"]
    } == {row["evidence_id"] for row in snapshot["artifact_registry"]}
    assert len(snapshot["contradictions"]) == 10
    assert [
        row["opportunity_id"] for row in snapshot["architecture_opportunity_matrix"]
    ] == list(builder_module.OPPORTUNITY_IDS)
    assert all(
        row["implementation_status"] == "not_started_not_authorized"
        and row["evaluation_contract_unchanged"] is True
        and row["necessary_modules"]
        and row["training_signals"]
        and row["key_ablations"]
        and row["falsifiable_predictions"]
        and set(row["prior_work_differentiation"])
        == {"CoPPS", "BATA", "HMPPS", "MemRerank"}
        for row in snapshot["architecture_opportunity_matrix"]
    )
    assert snapshot["scientific_effect_values_recomputed"] is False
    assert snapshot["source_test_opened"] is False
    assert snapshot["architecture_implemented"] is False


def test_model_uncovered_list_is_derived_from_the_18_by_4_matrix() -> None:
    decisions = _decisions()
    assert decisions["model_boundaries"][MODEL_IDS[0]]["uncovered_components"]
    decisions["model_boundaries"][MODEL_IDS[0]]["uncovered_components"].pop()
    with pytest.raises(ValueError, match="uncovered_components differs"):
        _validate_raw(decisions)


def test_untested_status_is_derived_from_registered_component_model_coverage() -> None:
    decisions = _decisions()
    uncovered_component = next(
        component_id
        for component_id in COMPONENT_IDS
        if decisions["component_model_matrix"][component_id][MODEL_IDS[0]][
            "status"
        ]
        == "untested"
    )
    decisions["component_model_matrix"][uncovered_component][MODEL_IDS[0]][
        "status"
    ] = "unresolved"
    with pytest.raises(ValueError, match="uncovered component-model cell must be untested"):
        _validate_raw(decisions)

    decisions = _decisions()
    covered_component = next(
        component_id
        for component_id in COMPONENT_IDS
        if decisions["component_model_matrix"][component_id][MODEL_IDS[2]][
            "status"
        ]
        != "untested"
    )
    decisions["component_model_matrix"][covered_component][MODEL_IDS[2]][
        "status"
    ] = "untested"
    with pytest.raises(ValueError, match="covered component-model cell cannot be untested"):
        _validate_raw(decisions)


def test_model_boundary_cannot_borrow_a_different_models_finding() -> None:
    decisions = _decisions()
    decisions["model_boundaries"][MODEL_IDS[0]]["supporting_findings"] = ["F1"]
    with pytest.raises(ValueError, match="outside its model scope"):
        _validate_raw(decisions)

    decisions = _decisions()
    decisions["model_boundaries"][MODEL_IDS[2]]["supporting_findings"] = []
    with pytest.raises(ValueError, match="completed direct evidence lacks findings"):
        _validate_raw(decisions)


def test_component_functional_questions_cover_every_transformer_axis() -> None:
    assert set(COMPONENT_FUNCTIONAL_QUESTIONS) == set(COMPONENT_IDS)
    assert "attention mass" in COMPONENT_FUNCTIONAL_QUESTIONS[
        "attention_value_transport"
    ]
    assert "candidate-common" in COMPONENT_FUNCTIONAL_QUESTIONS[
        "candidate_conditioned_interaction"
    ]
    assert "effective update" in COMPONENT_FUNCTIONAL_QUESTIONS[
        "optimizer_effective_update"
    ]
    assert set(SYSTEM_LAYER_COMPONENTS) == set(SYSTEM_LAYER_IDS)
    assert set().union(*SYSTEM_LAYER_COMPONENTS.values()) == set(COMPONENT_IDS)


def test_history_signal_scope_contract_distinguishes_spans_endpoints_and_blind_spots() -> None:
    rows = {
        row["scope_id"]: row for row in HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT
    }
    assert len(rows) == len(HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT) == 8
    assert "exact contiguous token span" in rows[
        "serialized_history_content_span"
    ]["sequence_or_token_scope"]
    assert "not a tokenwise trace" in rows[
        "layer_state_trajectory_endpoints"
    ]["not_observed"]
    assert "not a per-history-event MLP trajectory" in rows[
        "swiglu_feature_formation_endpoints"
    ]["not_observed"]
    assert set(rows["native_score_readout"]["model_scope"]) == set(MODEL_IDS)
    assert set().union(*(set(row["evidence_ids"]) for row in rows.values())).issubset(
        set(EXPECTED_DELIVERABLES) | set(EXPECTED_SUPPLEMENT_IDS)
    )


def test_five_system_layers_cannot_borrow_evidence_from_each_other() -> None:
    decisions = _decisions()
    decisions["system_layers"]["routing"]["supporting_findings"] = ["F1"]
    with pytest.raises(ValueError, match="evidence from a different layer"):
        _validate_raw(decisions)


def test_resolved_system_layer_requires_matching_component_cells() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    decisions["component_matrix"]["attention_query_key_routing"].update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    decisions["system_layers"]["routing"].update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    normalized = _validate(decisions)
    assert normalized["system_layers"]["routing"]["status"] == "supported"

    decisions["system_layers"]["routing"]["status"] = "weakened"
    with pytest.raises(ValueError, match="status/level/component/model-matched"):
        _validate(decisions)


def test_functional_localization_contract_forbids_layer_number_design() -> None:
    assert "pre/post interfaces" in FUNCTIONAL_LOCALIZATION_CONTRACT["authorized_role"]
    assert "never an absolute block index" in FUNCTIONAL_LOCALIZATION_CONTRACT[
        "cross_model_alignment_unit"
    ]
    assert any(
        "architecture target" in boundary
        for boundary in FUNCTIONAL_LOCALIZATION_CONTRACT["not_authorized"]
    )
    assert [row["stage"] for row in LOCALIZATION_TO_DESIGN_BRIDGE] == [
        "state_localization",
        "component_disambiguation",
        "bidirectional_causal_mediation",
        "cross_model_functional_replication",
    ]
    assert [row["design_authority"] for row in LOCALIZATION_TO_DESIGN_BRIDGE] == [
        False,
        False,
        False,
        True,
    ]
    assert "absolute index" in LOCALIZATION_TO_DESIGN_BRIDGE[-1][
        "authorized_consequence"
    ]


def test_incoming_state_gate_cannot_qualify_selected_block_residual_composition() -> None:
    assert DESIGN_NODE_COMPONENTS["block_input_residual"] == {"history_routing"}
    assert "residual_composition" not in DESIGN_NODE_COMPONENTS[
        "block_input_residual"
    ]
    assert DESIGN_NODE_COMPONENTS["block_output_residual"] == set()
    assert CAUSAL_CHAIN_COMPONENTS["incoming_state"] == {
        "history_routing",
        "layerwise_representation",
    }
    assert "residual_composition" not in CAUSAL_CHAIN_COMPONENTS["incoming_state"]
    assert "current block" in CAUSAL_CHAIN_CLAIM_BOUNDARIES["incoming_state"]
    assert "RMSNorm operator necessity" in CAUSAL_CHAIN_CLAIM_BOUNDARIES[
        "final_norm"
    ]


def test_residual_component_cannot_backfill_incoming_state_chain() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d2_selected_branches"],
            "supporting_supplements": [],
        }
    )
    decisions["component_matrix"]["residual_composition"].update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    incoming = next(
        row
        for row in decisions["functional_causal_chain"]
        if row["node"] == "incoming_state"
    )
    incoming.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="level-matched supported component"):
        _validate(decisions)


def test_design_gate_admission_rederives_cross_model_nodes_from_rows() -> None:
    rows = [
        {
            "method_id": model_id,
            "node": node,
            "endpoint_gates": {
                "target_margin": {
                    "position_preserving_removal_gate_passed": False,
                    "parent_same_request_sufficiency_gate_passed": False,
                    "parent_history_specificity_gate_passed": False,
                    "parent_cross_request_stress_gate_passed": False,
                    "parent_direction_scale_controls_passed": False,
                    "registered_component_state_gate_passed": False,
                    "functional_node_design_target_eligible": (
                        node != "block_output_residual"
                    ),
                    "robust_design_prioritization_gate_passed": False,
                }
            },
            "primary_target_margin_component_state_gate_passed": False,
            "primary_target_margin_design_gate_passed": False,
            "claim_role": DESIGN_NODE_CLAIM_ROLES[node],
        }
        for model_id in MODEL_IDS[2:]
        for node in DESIGN_NODE_COMPONENTS
    ]
    payload = {
        "analysis_type": "transformer_component_design_gate_synthesis",
        "status": "completed",
        "models": list(MODEL_IDS[2:]),
        "nodes": list(DESIGN_NODE_COMPONENTS),
        "primary_endpoint": "target_margin",
        "shared_parent_lineage": {
            model_id: {
                "shared_parent_bytes_verified": True,
                "exact_layer_index_is_architecture_evidence": False,
            }
            for model_id in MODEL_IDS[2:]
        },
        "rows": rows,
        "cross_model_functional_support": {
            "component_state_supported_nodes": [],
            "design_prioritized_nodes": [],
            "any_shared_component_state_node": False,
            "any_shared_design_prioritized_node": False,
            "component_path_design_ranking_eligible": False,
        },
        "claim_boundary": {
            "exact_layer_index_is_architecture_evidence": False,
            "operator_necessity_authorized": False,
            "block_output_state_ceiling_authorizes_residual_operator_claim": False,
            "registered_behavior": "harmful_full_history_target_margin_response",
            "positive_neutral_removal_means_harm_reduction": True,
            "component_is_beneficial_for_transfer_authorized": False,
            "strengthen_or_preserve_component_authorized": False,
        },
    }
    assert _audit_design_gate_payload(payload) == set()

    payload["claim_boundary"][
        "component_is_beneficial_for_transfer_authorized"
    ] = True
    with pytest.raises(ValueError, match="claim boundary differs"):
        _audit_design_gate_payload(payload)
    payload["claim_boundary"][
        "component_is_beneficial_for_transfer_authorized"
    ] = False

    payload["cross_model_functional_support"]["design_prioritized_nodes"] = [
        "attention_o_projection"
    ]
    with pytest.raises(ValueError, match="not row-derived"):
        _audit_design_gate_payload(payload)

    for row in rows:
        if row["node"] == "attention_o_projection":
            gate = row["endpoint_gates"]["target_margin"]
            gate.update(
                {
                    "position_preserving_removal_gate_passed": True,
                    "parent_same_request_sufficiency_gate_passed": True,
                    "parent_history_specificity_gate_passed": True,
                    "parent_cross_request_stress_gate_passed": True,
                    "parent_direction_scale_controls_passed": True,
                    "registered_component_state_gate_passed": True,
                    "robust_design_prioritization_gate_passed": True,
                }
            )
            row["primary_target_margin_component_state_gate_passed"] = True
            row["primary_target_margin_design_gate_passed"] = True
    payload["cross_model_functional_support"].update(
        {
            "component_state_supported_nodes": ["attention_o_projection"],
            "design_prioritized_nodes": ["attention_o_projection"],
            "any_shared_component_state_node": True,
            "any_shared_design_prioritized_node": True,
            "component_path_design_ranking_eligible": True,
        }
    )
    assert _audit_design_gate_payload(payload) == {"attention_o_projection"}
    necessity_support = _derive_necessity_component_models(payload)
    assert necessity_support["attention_output"] == set(MODEL_IDS[2:])
    assert necessity_support["mlp_output"] == set()
    gate_matrix = _build_component_bidirectional_gate_matrix(
        payload,
        evidence_identity={
            "path": "runs/component_design/metrics.json",
            "sha256": "9" * 64,
        },
    )
    assert len(gate_matrix["rows"]) == 8
    assert gate_matrix["cross_model"]["design_prioritized_nodes"] == [
        "attention_o_projection"
    ]
    attention_rows = [
        row
        for row in gate_matrix["rows"]
        if row["functional_node"] == "attention_o_projection"
    ]
    assert all(
        row["sufficiency_S_same_request"]
        and row["necessity_N_position_preserving_removal"]
        and row["history_specificity_same_minus_wrong"]
        and row["cross_request_stress_control"]
        and row["norm_direction_random_controls"]
        and row["design_G_gate"]
        for row in attention_rows
    )
    assert all(
        row["functional_node_design_target_eligible"] is False
        for row in gate_matrix["rows"]
        if row["functional_node"] == "block_output_residual"
    )

    rows[0]["claim_role"] = "forged_role"
    with pytest.raises(ValueError, match="claim role differs"):
        _audit_design_gate_payload(payload)
    rows[0]["claim_role"] = DESIGN_NODE_CLAIM_ROLES[rows[0]["node"]]
    target_gate = rows[0]["endpoint_gates"]["target_margin"]
    target_gate["functional_node_design_target_eligible"] = not target_gate[
        "functional_node_design_target_eligible"
    ]
    with pytest.raises(ValueError, match="target eligibility differs"):
        _audit_design_gate_payload(payload)
    target_gate["functional_node_design_target_eligible"] = not target_gate[
        "functional_node_design_target_eligible"
    ]

    block_output = next(
        row
        for row in rows
        if row["method_id"] == MODEL_IDS[2]
        and row["node"] == "block_output_residual"
    )
    block_output["primary_target_margin_component_state_gate_passed"] = True
    block_output["primary_target_margin_design_gate_passed"] = True
    with pytest.raises(ValueError, match="state ceiling cannot receive design"):
        _audit_design_gate_payload(payload)
    block_output["primary_target_margin_component_state_gate_passed"] = False
    block_output["primary_target_margin_design_gate_passed"] = False

    rows[0]["primary_target_margin_design_gate_passed"] = True
    with pytest.raises(ValueError, match="not primitive-gate-derived"):
        _audit_design_gate_payload(payload)


def test_absolute_layer_index_cannot_enter_design_opportunity() -> None:
    decisions = _decisions()
    decisions["optimization_opportunities"][0]["mechanism_target"] = (
        "insert a gate at layer 19"
    )
    with pytest.raises(ValueError, match="absolute internal index"):
        _validate(decisions)


def test_design_qualified_requires_cross_model_node_gate() -> None:
    decisions = _decisions()
    opportunity = decisions["optimization_opportunities"][0]
    opportunity.update(
        {
            "functional_component": "mlp_output",
            "functional_node": "mlp_down_projection",
            "minimum_evidence_level": "G",
            "actual_evidence_level": "G",
            "design_priority": "design_qualified",
            "intervention_polarity": "suppress_harmful_state",
            "supporting_supplements": [
                "component_functional_design_gate_synthesis"
            ],
            "supporting_findings": ["F1", "F_OPPORTUNITY"],
        }
    )
    with pytest.raises(ValueError, match="did not pass cross-model"):
        _validate(decisions)
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "G",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_formal_deliverables": [],
            "supporting_supplements": [
                "component_functional_design_gate_synthesis"
            ],
        }
    )
    decisions["component_matrix"]["mlp_output"].update(
        {
            "status": "supported",
            "evidence_level": "G",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    for model_id in MODEL_IDS[2:]:
        decisions["component_model_matrix"]["mlp_output"][model_id].update(
            {
                "status": "supported",
                "evidence_level": "G",
                "supporting_findings": ["F1"],
            }
        )
    normalized = _validate(decisions, design_nodes={"mlp_down_projection"})
    assert normalized["optimization_opportunities"][0]["actual_evidence_level"] == "G"

    opportunity["model_scope"] = [MODEL_IDS[2]]
    with pytest.raises(ValueError, match="exactly the two primary design models"):
        _validate(decisions, design_nodes={"mlp_down_projection"})


def test_g_finding_cannot_expand_or_shrink_primary_model_scope() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "G",
            "model_scope": list(MODEL_IDS),
            "supporting_formal_deliverables": [],
            "supporting_supplements": [
                "component_functional_design_gate_synthesis"
            ],
        }
    )
    with pytest.raises(ValueError, match="cross-model design gate"):
        _validate(decisions)

    finding["model_scope"] = [MODEL_IDS[2]]
    with pytest.raises(ValueError, match="cross-model design gate"):
        _validate(decisions)


def test_descriptive_evidence_cannot_be_marked_g_without_design_priority() -> None:
    decisions = _decisions()
    decisions["optimization_opportunities"][0]["actual_evidence_level"] = "G"
    with pytest.raises(ValueError, match="reserved for design-qualified"):
        _validate(decisions)


def test_opportunity_requires_component_matched_evidence_and_model_scope() -> None:
    decisions = _decisions()
    opportunity = decisions["optimization_opportunities"][0]
    opportunity["functional_component"] = "native_readout"
    with pytest.raises(ValueError, match="component-matched evidence"):
        _validate(decisions)

    decisions = _decisions()
    opportunity = decisions["optimization_opportunities"][0]
    opportunity["model_scope"] = list(MODEL_IDS)
    with pytest.raises(ValueError, match="model scope exceeds"):
        _validate(decisions)


def test_opportunity_evidence_must_be_interpreted_by_level_matched_findings() -> None:
    decisions = _decisions()
    decisions["optimization_opportunities"][0]["supporting_findings"] = []
    with pytest.raises(ValueError, match="lacks supporting findings"):
        _validate_raw(decisions)

    decisions = _decisions()
    decisions["optimization_opportunities"][0][
        "supporting_formal_deliverables"
    ].append("d6_q2_native_readout")
    with pytest.raises(ValueError, match="not interpreted by its supporting findings"):
        _validate_raw(decisions)

    decisions = _decisions()
    decisions["optimization_opportunities"][0].update(
        {
            "actual_evidence_level": "U",
            "design_priority": "deprioritized",
        }
    )
    with pytest.raises(ValueError, match="evidence-level/component/model-matched"):
        _validate_raw(decisions)

    decisions = _decisions()
    decisions["optimization_opportunities"][0]["actual_evidence_level"] = "U"
    with pytest.raises(ValueError, match="candidate-to-test opportunity requires"):
        _validate_raw(decisions)


def test_diagnostic_opportunity_cannot_claim_established_utility_gain() -> None:
    decisions = _decisions()
    decisions["optimization_opportunities"][0]["utility_gain_established"] = True
    with pytest.raises(ValueError, match="cannot establish opportunity utility"):
        _validate(decisions)


def test_opportunity_requires_explicit_expected_benefit_and_risks() -> None:
    decisions = _decisions()
    decisions["optimization_opportunities"][0].pop("expected_benefit")
    with pytest.raises(ValueError, match="expected_benefit"):
        _validate(decisions)


@pytest.mark.parametrize(
    ("field", "error"),
    [
        ("hypothesized_innovation", "hypothesized_innovation"),
        ("training_signal_requirements", "training_signal_requirements"),
        ("key_ablations", "key_ablations"),
        ("closest_baseline_families", "closest_baseline_families"),
        ("baseline_differentiation", "baseline_differentiation"),
    ],
)
def test_each_opportunity_retains_its_method_stage_falsification_contract(
    field: str, error: str
) -> None:
    decisions = _decisions()
    decisions["optimization_opportunities"][0].pop(field)
    with pytest.raises(ValueError, match=error):
        _validate(decisions)


def test_opportunity_cannot_claim_the_architecture_was_implemented() -> None:
    decisions = _decisions()
    decisions["optimization_opportunities"][0]["architecture_implemented"] = True
    with pytest.raises(ValueError, match="implemented architecture"):
        _validate(decisions)


def test_not_recommended_reason_uses_a_registered_basis_category() -> None:
    decisions = _decisions()
    decisions["not_recommended"][0]["basis"] = "seems weak"
    with pytest.raises(ValueError, match="invalid not_recommended basis"):
        _validate(decisions)

    decisions = _decisions()
    decisions["not_recommended"][0]["basis"] = "mechanical_non_result"
    with pytest.raises(ValueError, match="lacks a matching evidence level"):
        _validate(decisions)

    decisions = _decisions()
    negative_finding = next(
        row for row in decisions["findings"] if row["finding_id"] == "F_NEG_BOUNDARY"
    )
    negative_finding["evidence_level"] = "S"
    decisions["not_recommended"][0]["basis"] = "registered_refutation"
    with pytest.raises(ValueError, match="lacks a weakened component-model outcome"):
        _validate(decisions)

    decisions["component_matrix"]["layerwise_representation"].update(
        {
            "status": "weakened",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F_NEG_BOUNDARY"],
        }
    )
    decisions["negative_and_conflicting_results"][0][
        "supporting_findings"
    ].append("F_NEG_BOUNDARY")
    normalized = _validate(decisions)
    assert normalized["not_recommended"][0]["basis"] == "registered_refutation"

    decisions = _decisions()
    negative_finding = next(
        row for row in decisions["findings"] if row["finding_id"] == "F_NEG_BOUNDARY"
    )
    negative_finding["evidence_level"] = "M"
    decisions["not_recommended"][0]["basis"] = "mechanical_non_result"
    with pytest.raises(ValueError, match="lacks a mechanical-failure"):
        _validate(decisions)

    decisions = _decisions()
    decisions["not_recommended"][0]["functional_component"] = "native_readout"
    with pytest.raises(ValueError, match="component/model-matched evidence"):
        _validate(decisions)

    decisions = _decisions()
    decisions["not_recommended"][0]["model_scope"] = list(MODEL_IDS)
    with pytest.raises(ValueError, match="component/model-matched evidence"):
        _validate(decisions)

    decisions = _decisions()
    decisions["not_recommended"][0]["source_test_opened"] = True
    with pytest.raises(ValueError, match="preserve source_test_opened=false"):
        _validate(decisions)

    decisions = _decisions()
    decisions["not_recommended"][0]["supporting_findings"] = []
    with pytest.raises(ValueError, match="lacks a matching evidence level"):
        _validate(decisions)

    decisions = _decisions()
    decisions["optimization_opportunities"][0]["key_risks"] = []
    with pytest.raises(ValueError, match="key_risks"):
        _validate(decisions)

    decisions = _decisions()
    decisions["optimization_opportunities"][0]["key_risks"] = [
        "Overfit the intervention to layer 19."
    ]
    with pytest.raises(ValueError, match="absolute internal index"):
        _validate(decisions)


def test_optimization_opportunities_are_priority_ordered() -> None:
    decisions = _decisions()
    lower = decisions["optimization_opportunities"][0]
    lower["design_priority"] = "deprioritized"
    higher = copy.deepcopy(lower)
    higher["opportunity_id"] = "higher_priority_candidate"
    higher["design_priority"] = "candidate_to_test"
    higher["formal_predecessor_ids"] = []
    decisions["optimization_opportunities"] = [lower, higher]
    with pytest.raises(ValueError, match="must be ordered design-qualified"):
        _validate(decisions)

    decisions["optimization_opportunities"] = [higher, lower]
    normalized = _validate(decisions)
    assert [row["rank"] for row in normalized["optimization_opportunities"]] == [
        1,
        2,
    ]

    decisions = _decisions()
    wide = decisions["optimization_opportunities"][0]
    narrow = copy.deepcopy(wide)
    narrow["opportunity_id"] = "single_model_candidate"
    narrow["model_scope"] = [MODEL_IDS[2]]
    narrow["formal_predecessor_ids"] = []
    decisions["optimization_opportunities"] = [narrow, wide]
    with pytest.raises(ValueError, match="evidence strength, then model coverage"):
        _validate(decisions)

    decisions["optimization_opportunities"] = [wide, narrow]
    normalized = _validate(decisions)
    assert normalized["optimization_opportunities"][0][
        "evidence_strength_tier"
    ] == 2


def test_every_formal_opportunity_has_one_comprehensive_disposition() -> None:
    decisions = _decisions()
    omitted = "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK"
    decisions["optimization_opportunities"][0][
        "formal_predecessor_ids"
    ].remove(omitted)
    with pytest.raises(ValueError, match="disposition coverage differs"):
        _validate(decisions)

    decisions["not_recommended"][0]["formal_predecessor_ids"] = [omitted]
    normalized = _validate(decisions)
    assert normalized["formal_opportunity_disposition"][omitted][
        "disposition"
    ] == "mapped_to_not_recommended"

    decisions["optimization_opportunities"][0][
        "formal_predecessor_ids"
    ].append(omitted)
    with pytest.raises(ValueError, match="mapped more than once"):
        _validate(decisions)

    decisions = _decisions()
    decisions["optimization_opportunities"][0][
        "supporting_formal_deliverables"
    ].remove("d5_context")
    with pytest.raises(ValueError, match="lacks mechanism-family evidence"):
        _validate(decisions)

    decisions = _decisions()
    routing_id = "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER"
    decisions["optimization_opportunities"][0][
        "formal_predecessor_ids"
    ].remove(routing_id)
    decisions["not_recommended"][0]["formal_predecessor_ids"] = [routing_id]
    with pytest.raises(ValueError, match="not-recommended target"):
        _validate(decisions)


def test_opportunity_lineage_joins_prior_formal_and_final_disposition_exactly_once() -> None:
    normalized = _validate(_decisions())
    formal = {
        "architecture_opportunity_ranking": [
            {
                "opportunity_id": opportunity_id,
                "rank": rank,
                "status": "deprioritized",
                "evidence_deliverables": ["d1_representation"],
            }
            for rank, opportunity_id in enumerate(builder_module.OPPORTUNITY_IDS, 1)
        ]
    }
    lineage = _build_opportunity_lineage_matrix(
        prior=build_prior_mechanism_diagnosis_snapshot(ROOT),
        formal=formal,
        comprehensive=normalized,
    )
    assert lineage["prior_opportunity_count"] == 5
    assert lineage["formal_opportunity_count"] == 5
    assert lineage["disposition_count"] == 5
    assert lineage["all_predecessors_disposed_exactly_once"] is True
    assert lineage["utility_gain_established"] is False
    assert lineage["architecture_implemented"] is False
    assert [row["formal_rank"] for row in lineage["rows"]] == [1, 2, 3, 4, 5]
    assert {row["opportunity_id"] for row in lineage["rows"]} == set(
        builder_module.OPPORTUNITY_IDS
    )

    broken = copy.deepcopy(normalized)
    first_id = builder_module.OPPORTUNITY_IDS[0]
    broken["formal_opportunity_disposition"][first_id]["target_id"] = "missing"
    with pytest.raises(ValueError, match="lineage target is missing"):
        _build_opportunity_lineage_matrix(
            prior=build_prior_mechanism_diagnosis_snapshot(ROOT),
            formal=formal,
            comprehensive=broken,
        )


def test_opportunity_evidence_level_cannot_upgrade_descriptive_component_evidence() -> None:
    decisions = _decisions()
    opportunity = decisions["optimization_opportunities"][0]
    opportunity["actual_evidence_level"] = "S"
    with pytest.raises(ValueError, match="component causal evidence"):
        _validate(decisions)

    decisions = _decisions()
    opportunity = decisions["optimization_opportunities"][0]
    opportunity["functional_component"] = "normalization"
    opportunity["actual_evidence_level"] = "N"
    with pytest.raises(ValueError, match="component necessity evidence"):
        _validate(decisions)


def test_necessity_opportunity_scope_comes_from_passing_model_rows() -> None:
    decisions = _decisions()
    opportunity = decisions["optimization_opportunities"][0]
    opportunity.update(
        {
            "functional_component": "attention_output",
            "minimum_evidence_level": "N",
            "actual_evidence_level": "N",
            "intervention_polarity": "reroute_history_state",
            "supporting_supplements": ["component_state_reverse_necessity_v2"],
            "supporting_findings": ["F1", "F_OPPORTUNITY"],
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        }
    )
    with pytest.raises(ValueError, match="component necessity evidence"):
        _validate(
            decisions,
            necessity_models={"attention_output": {MODEL_IDS[2]}},
        )
    with pytest.raises(
        ValueError, match="conflicts with the component evidence matrix"
    ):
        _validate(
            decisions,
            necessity_models={
                "attention_output": {MODEL_IDS[2], MODEL_IDS[3]},
            },
        )

    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "N",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_formal_deliverables": [],
            "supporting_supplements": ["component_state_reverse_necessity_v2"],
        }
    )
    decisions["component_matrix"]["attention_output"].update(
        {
            "status": "supported",
            "evidence_level": "N",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    for model_id in MODEL_IDS[2:]:
        decisions["component_model_matrix"]["attention_output"][model_id].update(
            {
                "status": "supported",
                "evidence_level": "N",
                "supporting_findings": ["F1"],
            }
        )

    normalized = _validate(
        decisions,
        necessity_models={
            "attention_output": {MODEL_IDS[2], MODEL_IDS[3]},
        },
    )
    assert normalized["optimization_opportunities"][0][
        "actual_evidence_level"
    ] == "N"

    opportunity["intervention_polarity"] = "preserve_or_strengthen_beneficial_state"
    with pytest.raises(ValueError, match="cannot preserve or strengthen"):
        _validate(
            decisions,
            necessity_models={
                "attention_output": {MODEL_IDS[2], MODEL_IDS[3]},
            },
        )

    opportunity["intervention_polarity"] = "diagnostic_only"
    with pytest.raises(ValueError, match="must suppress, reroute, or recalibrate"):
        _validate(
            decisions,
            necessity_models={
                "attention_output": {MODEL_IDS[2], MODEL_IDS[3]},
            },
        )


def test_mechanical_evidence_cannot_rank_an_optimization_candidate() -> None:
    decisions = _decisions()
    finding = next(
        row for row in decisions["findings"] if row["finding_id"] == "F_OPPORTUNITY"
    )
    finding["evidence_level"] = "M"
    opportunity = decisions["optimization_opportunities"][0]
    opportunity["actual_evidence_level"] = "M"
    with pytest.raises(ValueError, match="mechanical evidence cannot rank"):
        _validate(decisions)


def test_finding_cannot_upgrade_descriptive_supplement() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "N"
    finding["supporting_formal_deliverables"] = []
    with pytest.raises(ValueError, match="lacks reverse-necessity"):
        _validate(decisions)


def test_supported_component_and_hypothesis_require_causal_level() -> None:
    decisions = _decisions()
    decisions["component_matrix"][COMPONENT_IDS[0]]["status"] = "supported"
    decisions["component_matrix"][COMPONENT_IDS[0]]["evidence_level"] = "D"
    with pytest.raises(ValueError, match="supported component lacks causal"):
        _validate(decisions)

    decisions = _decisions()
    decisions["hypothesis_matrix"]["H0"]["status"] = "rejected"
    decisions["hypothesis_matrix"]["H0"]["evidence_level"] = "D"
    with pytest.raises(ValueError, match="decisive hypothesis status"):
        _validate(decisions)


def test_supported_hypothesis_requires_its_own_registered_evidence_groups() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "S"
    finding["supporting_formal_deliverables"] = ["d2_selected_branches"]
    finding["supporting_supplements"] = []
    hypothesis = decisions["hypothesis_matrix"]["H4"]
    hypothesis.update(
        {"status": "supported", "evidence_level": "S", "supporting_findings": ["F1"]}
    )
    with pytest.raises(ValueError, match="hypothesis-matched"):
        _validate(decisions)


def test_supported_h1_requires_both_groups_and_required_components() -> None:
    decisions = _decisions()
    first = decisions["findings"][0]
    first.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    second = {
        **copy.deepcopy(first),
        "finding_id": "F2",
        "title": "History-specific context control",
        "supporting_formal_deliverables": ["d5_context"],
    }
    decisions["findings"].append(second)
    for component_id in ("attention_query_key_routing", "history_routing"):
        decisions["component_matrix"][component_id].update(
            {
                "status": "supported",
                "evidence_level": "S",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "supporting_findings": ["F1"],
            }
        )
    h1 = decisions["hypothesis_matrix"]["H1"]
    h1.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "supporting_findings": ["F1", "F2"],
        }
    )
    normalized = _validate(decisions)
    assert normalized["hypothesis_matrix"]["H1"]["status"] == "supported"

    h1["supporting_findings"] = ["F1"]
    with pytest.raises(ValueError, match="preregistered evidence group"):
        _validate(decisions)
    h1["supporting_findings"] = ["F1", "F2"]

    decisions["component_matrix"]["history_routing"]["status"] = "unresolved"
    decisions["component_matrix"]["history_routing"]["evidence_level"] = "U"
    with pytest.raises(ValueError, match="required supported components"):
        _validate(decisions)


def test_h5_cannot_be_decisive_without_registered_independent_seed() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "S"
    finding["supporting_formal_deliverables"] = ["d2_selected_branches"]
    finding["supporting_supplements"] = []
    decisions["hypothesis_matrix"]["H5"].update(
        {"status": "rejected", "evidence_level": "S", "supporting_findings": ["F1"]}
    )
    with pytest.raises(ValueError, match="unregistered independent seed"):
        _validate(decisions)


def test_reverse_component_necessity_cannot_decide_a_whole_hypothesis() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "N",
            "supporting_formal_deliverables": [],
            "supporting_supplements": ["component_state_reverse_necessity_v2"],
        }
    )
    decisions["hypothesis_matrix"]["H0"].update(
        {
            "status": "rejected",
            "evidence_level": "N",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="cannot by itself decide H0--H5"):
        _validate(decisions)


def test_supported_component_requires_component_matched_sufficiency() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "S"
    finding["supporting_formal_deliverables"] = ["d2_selected_branches"]
    finding["supporting_supplements"] = []

    embedding = decisions["component_matrix"]["token_embedding"]
    embedding.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="component-matched sufficiency"):
        _validate(decisions)

    decisions["component_matrix"]["token_embedding"] = {
        **embedding,
        "status": "unresolved",
        "evidence_level": "U",
        "supporting_findings": [],
    }
    attention = decisions["component_matrix"]["attention_output"]
    attention.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "supporting_findings": ["F1"],
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        }
    )
    normalized = _validate(decisions)
    assert normalized["component_matrix"]["attention_output"]["status"] == (
        "supported"
    )


def test_supported_component_requires_matching_design_qualified_node() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "G"
    finding["supporting_formal_deliverables"] = []
    finding["supporting_supplements"] = [
        "component_functional_design_gate_synthesis"
    ]
    embedding = decisions["component_matrix"]["token_embedding"]
    embedding.update(
        {
            "status": "supported",
            "evidence_level": "G",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="component-matched cross-model"):
        _validate(decisions, design_nodes={"mlp_down_projection"})

    embedding.update(
        {"status": "unresolved", "evidence_level": "U", "supporting_findings": []}
    )
    mlp = decisions["component_matrix"]["mlp_output"]
    mlp.update(
        {
            "status": "supported",
            "evidence_level": "G",
            "supporting_findings": ["F1"],
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        }
    )
    normalized = _validate(decisions, design_nodes={"mlp_down_projection"})
    assert normalized["component_matrix"]["mlp_output"]["evidence_level"] == "G"

    mlp["model_scope"] = [MODEL_IDS[2]]
    with pytest.raises(ValueError, match="component-matched cross-model"):
        _validate(decisions, design_nodes={"mlp_down_projection"})


def test_supported_component_requires_component_eligible_necessity() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "N"
    finding["supporting_formal_deliverables"] = []
    finding["supporting_supplements"] = ["component_state_reverse_necessity_v2"]
    normalization = decisions["component_matrix"]["normalization"]
    normalization.update(
        {
            "status": "supported",
            "evidence_level": "N",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="component/model-matched necessity"):
        _validate(decisions)

    normalization.update(
        {"status": "unresolved", "evidence_level": "U", "supporting_findings": []}
    )
    attention = decisions["component_matrix"]["attention_output"]
    attention.update(
        {
            "status": "supported",
            "evidence_level": "N",
            "supporting_findings": ["F1"],
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
        }
    )
    normalized = _validate(
        decisions,
        necessity_models={
            "attention_output": {MODEL_IDS[2], MODEL_IDS[3]},
        },
    )
    assert normalized["component_matrix"]["attention_output"]["evidence_level"] == "N"

    with pytest.raises(ValueError, match="component/model-matched necessity"):
        _validate(
            decisions,
            necessity_models={"attention_output": {MODEL_IDS[2]}},
        )


def test_supported_component_cannot_expand_beyond_causal_finding_models() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "S"
    finding["supporting_formal_deliverables"] = ["d2_selected_branches"]
    finding["supporting_supplements"] = []
    attention = decisions["component_matrix"]["attention_output"]
    attention.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "supporting_findings": ["F1"],
            "model_scope": list(MODEL_IDS),
        }
    )
    with pytest.raises(ValueError, match="model scope exceeds"):
        _validate(decisions)


def test_functional_causal_chain_requires_causal_level_and_status_vocabulary() -> None:
    decisions = _decisions()
    attention = decisions["functional_causal_chain"][1]
    attention.update(
        {"status": "supported", "evidence_level": "D", "supporting_findings": ["F1"]}
    )
    with pytest.raises(ValueError, match="lacks causal level"):
        _validate(decisions)

    decisions = _decisions()
    decisions["functional_causal_chain"][1]["evidence_level"] = "D"
    with pytest.raises(ValueError, match="unresolved causal-chain node"):
        _validate(decisions)

    decisions = _decisions()
    decisions["functional_causal_chain"][1]["status"] = "primary_bottleneck"
    with pytest.raises(ValueError, match="invalid functional causal-chain status"):
        _validate(decisions)


def test_supported_functional_chain_requires_level_matched_finding_and_component() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    component = decisions["component_matrix"]["attention_query_key_routing"]
    component.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    attention = decisions["functional_causal_chain"][1]
    attention.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    normalized = _validate(decisions)
    assert normalized["functional_causal_chain"][1]["status"] == "supported"

    component["status"] = "unresolved"
    component["evidence_level"] = "U"
    with pytest.raises(ValueError, match="level-matched supported component"):
        _validate(decisions)


def test_supported_functional_chain_cannot_overstate_component_model_scope() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    component = decisions["component_matrix"]["attention_query_key_routing"]
    component.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2]],
            "supporting_findings": ["F1"],
        }
    )
    attention = decisions["functional_causal_chain"][1]
    attention.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="model scope exceeds"):
        _validate(decisions)


def test_weakened_functional_chain_requires_matched_negative_component() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d3_attention_edges"],
            "supporting_supplements": [],
        }
    )
    component = decisions["component_matrix"]["attention_query_key_routing"]
    component.update(
        {
            "status": "weakened",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    attention = decisions["functional_causal_chain"][1]
    attention.update(
        {
            "status": "weakened",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    normalized = _validate(decisions)
    assert normalized["functional_causal_chain"][1]["status"] == "weakened"

    component["status"] = "unresolved"
    component["evidence_level"] = "U"
    component["model_scope"] = []
    component["supporting_findings"] = []
    with pytest.raises(ValueError, match="level-matched weakened component"):
        _validate(decisions)


def test_g_functional_chain_requires_both_primary_models() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "G",
            "supporting_formal_deliverables": [],
            "supporting_supplements": [
                "component_functional_design_gate_synthesis"
            ],
        }
    )
    component = decisions["component_matrix"]["mlp_output"]
    component.update(
        {
            "status": "supported",
            "evidence_level": "G",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F1"],
        }
    )
    mlp_chain = decisions["functional_causal_chain"][2]
    mlp_chain.update(
        {
            "status": "supported",
            "evidence_level": "G",
            "model_scope": [MODEL_IDS[2]],
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="exactly the two primary design models"):
        _validate(decisions, design_nodes={"mlp_down_projection"})

    mlp_chain["model_scope"] = [MODEL_IDS[2], MODEL_IDS[3]]
    normalized = _validate(decisions, design_nodes={"mlp_down_projection"})
    assert normalized["functional_causal_chain"][2]["evidence_level"] == "G"


def test_weakened_hypothesis_requires_matching_formal_negative_outcome() -> None:
    decisions = _decisions()
    hypothesis = decisions["hypothesis_matrix"]["H0"]
    hypothesis.update(
        {
            "status": "weakened",
            "evidence_level": "D",
            "supporting_findings": ["F1"],
            "negative_evidence_basis": ["registered_weakening"],
        }
    )
    normalized = _validate(decisions)
    formal = _formal_outcomes_fixture()
    with pytest.raises(ValueError, match="weakened hypothesis changes"):
        _audit_comprehensive_against_formal(normalized, formal)
    formal["hypothesis_status_matrix"][0]["status"] = "weakened"
    _audit_comprehensive_against_formal(normalized, formal)

    decisions["findings"][0]["supporting_formal_deliverables"] = [
        "d3_attention_edges"
    ]
    with pytest.raises(ValueError, match="hypothesis-matched registered evidence"):
        _validate(decisions)


def test_design_gate_cannot_refute_or_replace_hypothesis_evidence() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "G",
            "supporting_formal_deliverables": [],
            "supporting_supplements": [
                "component_functional_design_gate_synthesis"
            ],
        }
    )
    hypothesis = decisions["hypothesis_matrix"]["H0"]
    hypothesis.update(
        {
            "status": "rejected",
            "evidence_level": "G",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="not hypothesis refutation"):
        _validate(decisions)

    hypothesis["status"] = "supported"
    with pytest.raises(ValueError, match="hypothesis-matched evidence"):
        _validate(decisions)


def test_rejected_hypothesis_must_match_admitted_formal_status() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "S",
            "supporting_formal_deliverables": ["d1_representation"],
            "supporting_supplements": [],
        }
    )
    hypothesis = decisions["hypothesis_matrix"]["H0"]
    hypothesis.update(
        {
            "status": "rejected",
            "evidence_level": "S",
            "supporting_findings": ["F1"],
            "negative_evidence_basis": ["registered_refutation"],
        }
    )
    normalized = _validate(decisions)
    formal = _formal_outcomes_fixture()
    with pytest.raises(ValueError, match="rejected hypothesis changes"):
        _audit_comprehensive_against_formal(normalized, formal)
    formal["hypothesis_status_matrix"][0]["status"] = "rejected"
    _audit_comprehensive_against_formal(normalized, formal)


def test_descriptive_failure_mode_cannot_authorize_erasure() -> None:
    decisions = _decisions()
    diagnosis = decisions["failure_mode_diagnosis"]
    diagnosis["primary_mode"] = "localized_state_attenuation"
    diagnosis["evidence_level"] = "D"
    diagnosis["causal_erasure_claim_authorized"] = True
    diagnosis["competing_modes"][0]["mode"] = "distributed_state_attenuation"
    with pytest.raises(ValueError, match="requires bidirectional level G"):
        _validate(decisions)


def test_even_bidirectional_component_mediation_cannot_claim_signal_erasure() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding.update(
        {
            "evidence_level": "G",
            "supporting_formal_deliverables": ["d2_selected_branches"],
            "supporting_supplements": [
                "component_functional_design_gate_synthesis"
            ],
        }
    )
    diagnosis = decisions["failure_mode_diagnosis"]
    diagnosis.update(
        {
            "primary_mode": "localized_state_attenuation",
            "evidence_level": "G",
            "causal_erasure_claim_authorized": True,
            "competing_modes": [
                {
                    "mode": "distributed_state_attenuation",
                    "reason_remaining": "A distributed pattern remains possible.",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="no registered signal-erasure experiment"):
        _validate(decisions)


def test_failure_mode_cannot_use_exact_layer_for_design() -> None:
    decisions = _decisions()
    decisions["failure_mode_diagnosis"]["exact_layer_index_used_for_design"] = True
    with pytest.raises(ValueError, match="exact layer index"):
        _validate(decisions)


def test_loss_of_use_claim_requires_model_scoped_native_readouts() -> None:
    decisions = _decisions()
    finding = decisions["findings"][0]
    finding["evidence_level"] = "G"
    finding["supporting_supplements"] = [
        "component_functional_design_gate_synthesis"
    ]
    diagnosis = decisions["failure_mode_diagnosis"]
    diagnosis["primary_mode"] = "state_present_but_readout_misaligned"
    diagnosis["evidence_level"] = "G"
    diagnosis["causal_loss_of_use_claim_authorized"] = True
    diagnosis["competing_modes"] = [
        {
            "mode": "candidate_transport_failure",
            "reason_remaining": "Transport remains a competing explanation.",
        }
    ]
    with pytest.raises(ValueError, match="lacks model-scoped native readout"):
        _validate(decisions)


def test_loss_of_use_claim_requires_supported_native_score_chain() -> None:
    decisions = _decisions()
    design_finding = decisions["findings"][0]
    design_finding.update(
        {
            "evidence_level": "G",
            "supporting_formal_deliverables": [
                "d6_q2_native_readout",
                "d6_q3_native_readout",
            ],
            "supporting_supplements": [
                "component_functional_design_gate_synthesis"
            ],
        }
    )
    readout_finding = {
        **copy.deepcopy(design_finding),
        "finding_id": "F2",
        "title": "Model-scoped native readout intervention",
        "evidence_level": "S",
        "supporting_supplements": [],
    }
    decisions["findings"].append(readout_finding)
    decisions["component_matrix"]["native_readout"].update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F2"],
        }
    )
    diagnosis = decisions["failure_mode_diagnosis"]
    diagnosis.update(
        {
            "primary_mode": "state_present_but_readout_misaligned",
            "evidence_level": "G",
            "functional_components": ["attention_output", "native_readout"],
            "supporting_findings": ["F1", "F2"],
            "causal_loss_of_use_claim_authorized": True,
            "competing_modes": [
                {
                    "mode": "candidate_transport_failure",
                    "reason_remaining": "Transport remains a competing explanation.",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="supported native-score chain"):
        _validate(decisions)

    native_score = next(
        row
        for row in decisions["functional_causal_chain"]
        if row["node"] == "native_score"
    )
    native_score.update(
        {
            "status": "supported",
            "evidence_level": "S",
            "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
            "supporting_findings": ["F2"],
        }
    )
    normalized = _validate(decisions)
    assert normalized["failure_mode_diagnosis"][
        "causal_loss_of_use_claim_authorized"
    ] is True
    assert normalized["failure_mode_diagnosis"]["claim_boundary"][
        "registered_signal_erasure_experiment_exists"
    ] is False


def test_resolved_failure_mode_requires_mode_component_model_binding() -> None:
    decisions = _decisions()
    diagnosis = decisions["failure_mode_diagnosis"]
    diagnosis.update(
        {
            "primary_mode": "objective_update_mismatch",
            "evidence_level": "D",
            "functional_components": [],
            "competing_modes": [
                {
                    "mode": "candidate_transport_failure",
                    "reason_remaining": "Transport remains a competing explanation.",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="requires functional components"):
        _validate(decisions)

    diagnosis["functional_components"] = ["layerwise_representation"]
    with pytest.raises(ValueError, match="mode/component/model-matched"):
        _validate(decisions)

    diagnosis["functional_components"] = ["optimizer_effective_update"]
    with pytest.raises(ValueError, match="without component/model-matched"):
        _validate(decisions)


def test_mechanical_evidence_cannot_name_a_scientific_failure_mode() -> None:
    decisions = _decisions()
    diagnosis = decisions["failure_mode_diagnosis"]
    diagnosis.update(
        {
            "primary_mode": "localized_state_attenuation",
            "evidence_level": "M",
            "functional_components": ["layerwise_representation"],
            "competing_modes": [
                {
                    "mode": "distributed_state_attenuation",
                    "reason_remaining": "The trajectory shape remains unresolved.",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="mechanical evidence cannot name"):
        _validate(decisions)


def test_multiple_bottlenecks_require_two_evidenced_system_layers_per_model() -> None:
    decisions = _decisions()
    diagnosis = decisions["failure_mode_diagnosis"]
    diagnosis.update(
        {
            "primary_mode": "multiple_bottlenecks",
            "evidence_level": "D",
            "functional_components": [
                "layerwise_representation",
                "normalization",
            ],
            "competing_modes": [
                {
                    "mode": "localized_state_attenuation",
                    "reason_remaining": "One representation bottleneck remains possible.",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="two independently evidenced system layers"):
        _validate(decisions)


def test_missing_component_or_hypothesis_fails_closed() -> None:
    decisions = _decisions()
    decisions["component_matrix"].pop(COMPONENT_IDS[0])
    with pytest.raises(ValueError, match="component_matrix coverage drift"):
        _validate(decisions)

    decisions = _decisions()
    decisions["hypothesis_matrix"].pop("H5")
    with pytest.raises(ValueError, match="hypothesis_matrix coverage drift"):
        _validate(decisions)


def test_every_admitted_evidence_item_requires_an_explicit_disposition() -> None:
    decisions = _decisions()
    decisions["evidence_disposition"].pop("d5_context")
    with pytest.raises(ValueError, match="evidence_disposition coverage drift"):
        _validate_raw(decisions)

    template = build_comprehensive_decision_template()
    assert set(template["evidence_disposition"]) == set(EXPECTED_DELIVERABLES) | set(
        EXPECTED_SUPPLEMENT_IDS
    )
    assert len(template["evidence_disposition"]) == 40


def test_evidence_disposition_cannot_use_an_unrelated_finding_or_hide_a_claim() -> None:
    decisions = _decisions()
    decisions["evidence_disposition"]["d5_context"].update(
        {
            "disposition": "interpreted_in_findings",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(ValueError, match="does not cite the same evidence"):
        _validate_raw(decisions)

    decisions = _decisions()
    decisions["evidence_disposition"]["d5_context"].update(
        {
            "disposition": "bounded_no_scientific_claim",
            "supporting_findings": ["F1"],
        }
    )
    with pytest.raises(
        ValueError, match="bounded evidence disposition cannot cite"
    ):
        _validate_raw(decisions)


def test_negative_evidence_disposition_must_enter_negative_table() -> None:
    decisions = _decisions()
    cross_finding = next(
        row for row in decisions["findings"] if row["finding_id"] == "F_CROSS"
    )
    cross_finding["supporting_formal_deliverables"].append("d5_context")
    decisions["evidence_disposition"]["d5_context"].update(
        {
            "disposition": "negative_or_conflicting",
            "supporting_findings": ["F_CROSS"],
        }
    )
    with pytest.raises(ValueError, match="absent from the registered negative"):
        _validate_raw(decisions)

    decisions["negative_and_conflicting_results"][0][
        "supporting_findings"
    ].append("F_CROSS")
    normalized = _validate_raw(decisions)
    assert normalized["evidence_disposition"]["d5_context"][
        "disposition"
    ] == "negative_or_conflicting"


def test_markdown_contains_all_thirteen_required_sections() -> None:
    decisions = _validate(_decisions())
    for row in decisions["findings"]:
        row["supporting_evidence_identities"] = [
            *(
                {
                    "evidence_id": evidence_id,
                    "evidence_kind": "formal_deliverable",
                    "path": f"runs/{evidence_id}/metrics.json",
                    "sha256": "a" * 64,
                }
                for evidence_id in row["supporting_formal_deliverables"]
            ),
            *(
                {
                    "evidence_id": evidence_id,
                    "evidence_kind": "supplement",
                    "path": f"runs/{evidence_id}/metrics.json",
                    "sha256": "b" * 64,
                }
                for evidence_id in row["supporting_supplements"]
            ),
        ]
    for row in decisions["optimization_opportunities"]:
        row["supporting_evidence_identities"] = [
            {
                "evidence_id": "d1_representation",
                "evidence_kind": "formal_deliverable",
                "path": "runs/formal/metrics.json",
                "sha256": "a" * 64,
            }
        ]
    for row in decisions["narratives"].values():
        row["supporting_evidence_identities"] = [
            {
                "evidence_id": "d1_representation",
                "evidence_kind": "formal_deliverable",
                "path": "runs/formal/metrics.json",
                "sha256": "a" * 64,
            }
        ]
    for evidence_id, row in decisions["evidence_disposition"].items():
        row["evidence_identity"] = {
            "evidence_id": evidence_id,
            "evidence_kind": row["evidence_kind"],
            "path": f"runs/{evidence_id}/metrics.json",
            "sha256": "f" * 64,
        }
    payload = {
        "status": "completed",
        **decisions,
        "localization_to_design_bridge": list(LOCALIZATION_TO_DESIGN_BRIDGE),
        "component_bidirectional_gate_matrix": {
            "source": {
                "path": "runs/component_design/metrics.json",
                "sha256": "8" * 64,
            },
            "primary_endpoint": "target_margin",
            "rows": [
                {
                    "method_id": model_id,
                    "functional_node": node,
                    "claim_role": DESIGN_NODE_CLAIM_ROLES[node],
                    "sufficiency_S_same_request": False,
                    "history_specificity_same_minus_wrong": False,
                    "necessity_N_position_preserving_removal": False,
                    "cross_request_stress_control": False,
                    "norm_direction_random_controls": False,
                    "combined_component_state_gate": False,
                    "functional_node_design_target_eligible": (
                        node != "block_output_residual"
                    ),
                    "design_G_gate": False,
                }
                for model_id in MODEL_IDS[2:]
                for node in DESIGN_NODE_COMPONENTS
            ],
            "cross_model": {
                "component_state_supported_nodes": [],
                "design_prioritized_nodes": [],
                "any_shared_component_state_node": False,
                "any_shared_design_prioritized_node": False,
                "component_path_design_ranking_eligible": False,
            },
            "interpretation_boundary": "S, N, specificity, controls, and G remain separate.",
            "scientific_effect_values_recomputed": False,
        },
        "necessity_direction_claim_boundary": dict(
            NECESSITY_DIRECTION_CLAIM_BOUNDARY
        ),
        "component_functional_questions": COMPONENT_FUNCTIONAL_QUESTIONS,
        "component_evidence_role_coverage": (
            _build_component_evidence_role_coverage(
                _readiness_component_coverage_fixture()
            )
        ),
        "transformer_internal_interface_coverage": (
            build_transformer_interface_coverage(
                completed_formal=set(EXPECTED_DELIVERABLES),
                completed_supplements=set(EXPECTED_SUPPLEMENT_IDS),
                supplement_model_scopes={
                    evidence_id: set(MODEL_IDS)
                    for evidence_id in EXPECTED_SUPPLEMENT_IDS
                },
                supplement_component_scopes={
                    evidence_id: set(COMPONENT_IDS)
                    for evidence_id in EXPECTED_SUPPLEMENT_IDS
                },
            )
        ),
        "history_signal_observation_scope_contract": list(
            HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT
        ),
        "frozen_model_architecture_audit": {
            "status": "completed",
            "failures": [],
            "frozen_topology": {
                "num_hidden_layers": 28,
                "hidden_size": 1024,
                "intermediate_size": 3072,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "rope_theta": 1_000_000,
                "rms_norm_eps": 1e-6,
                "tie_word_embeddings": True,
            },
            "model_pathways": {
                "q0_qwen3_reranker_06b": {
                    "base_artifact": "qwen3_reranker_06b",
                    "adaptation": "full_parameter",
                    "optimizer_steps": 967,
                    "objective": "pointwise_binary_cross_entropy_on_yes_no_logits",
                },
                "q1_instructrec_generalqwen": {
                    "base_artifact": "qwen3_general_06b",
                    "adaptation": "full_parameter",
                    "optimizer_steps": 967,
                    "objective": "output_only_normalized_candidate_response_nll",
                },
                "q2_recranker_generalqwen": {
                    "base_artifact": "qwen3_general_06b",
                    "adaptation": "full_parameter",
                    "optimizer_steps": 967,
                    "objective": "0.5_ranknet_plus_0.5_tie_aware_listnet",
                },
                "q3_tallrec_generalqwen": {
                    "base_artifact": "qwen3_general_06b",
                    "adaptation": "lora",
                    "lora_rank": 8,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05,
                    "lora_targets": ["q_proj", "v_proj"],
                    "optimizer_steps": 967,
                    "objective": "output_only_yes_no_recommendation_alignment_nll_with_lora",
                },
            },
            "frozen_model_pathway_count": 4,
            "frozen_base_artifact_count": 2,
            "config_backed_interface_count": 53,
            "exact_interface_inventory_count": 63,
            "config_backed_interfaces_present_in_inventory": True,
            "dynamic_runtime_or_source_backed_interface_count": 10,
            "implementation_provenance_covered_interface_count": 63,
            "all_exact_interfaces_have_config_or_runtime_source_provenance": True,
            "forward_primitive_count": 40,
            "forward_primitives": [
                {
                    "primitive_id": row["primitive_id"],
                    "execution_order": row["execution_order"],
                    "implementation_step": row["implementation_step"],
                    "interface_ids": list(row["interface_ids"]),
                    "model_scope": list(row["model_scope"]),
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                }
                for row in FORWARD_PRIMITIVE_CONTRACTS
            ],
            "forward_inference_interface_count": 43,
            "forward_mapped_interface_count": 43,
            "forward_missing_interface_ids": [],
            "forward_extraneous_interface_ids": [],
            "forward_training_interface_count": 20,
            "forward_training_interfaces_excluded_by_design": [
                f"training_interface_{index}" for index in range(20)
            ],
            "forward_primitive_interface_coverage_complete": True,
            "forward_source_binding_count": 11,
            "forward_source_bindings": [
                {
                    "source_id": f"forward_source_{index}",
                    "object": f"ForwardObject{index}",
                    "package_relative_path": f"transformers/forward_{index}.py",
                    "source_file_sha256": "a" * 64,
                    "object_source_sha256": "b" * 64,
                    "required_fragment_count": 2,
                    "missing_fragments": [],
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                }
                for index in range(11)
            ],
            "transformers_version": "5.12.1",
            "forward_peft_version": "0.19.1",
            "forward_frozen_python_executable": "/frozen/bin/python",
            "forward_source_environment_is_frozen_checkpoint_environment": True,
            "inactive_architecture_path_count": 9,
            "inactive_architecture_paths": [
                {
                    "path_id": f"inactive_path_{index}",
                    "evidence_kind": "fixture",
                    "observed": {"value": False},
                    "expected": False,
                    "inactive_verified": True,
                    "explanation": "Frozen inactive path fixture.",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                }
                for index in range(9)
            ],
            "all_inactive_architecture_paths_verified": True,
            "forward_graph_failures": [],
            "forward_coverage_is_semantic_primitive_census": True,
            "forward_coverage_is_kernel_instruction_census": False,
            "operator_attribution_inferred_from_forward_coverage": False,
            "training_primitive_count": 27,
            "training_primitives": [
                {
                    "primitive_id": row["primitive_id"],
                    "execution_order": row["execution_order"],
                    "implementation_step": row["implementation_step"],
                    "interface_ids": list(row["interface_ids"]),
                    "model_scope": list(row["model_scope"]),
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                }
                for row in TRAINING_PRIMITIVE_CONTRACTS
            ],
            "training_exact_interface_count": 20,
            "training_mapped_interface_count": 20,
            "training_missing_interface_ids": [],
            "training_extraneous_interface_ids": [],
            "training_nontraining_interface_count": 43,
            "training_nontraining_interfaces_excluded_by_design": [
                f"nontraining_interface_{index}" for index in range(43)
            ],
            "training_primitive_interface_coverage_complete": True,
            "training_source_binding_count": 12,
            "training_source_bindings": [
                {
                    "source_id": f"training_source_{index}",
                    "object": f"TrainingObject{index}",
                    "package_relative_path": f"myrec/training_{index}.py",
                    "source_file_sha256": "1" * 64,
                    "object_source_sha256": "2" * 64,
                    "required_fragment_count": 2,
                    "missing_fragments": [],
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                }
                for index in range(12)
            ],
            "training_artifact_binding_count": 3,
            "training_artifact_bindings": [
                {
                    "binding_id": "q3_peft_lora_adapter_config",
                    "binding_kind": "frozen_checkpoint_adapter_config",
                    "path": "artifacts/q3/adapter_config.json",
                    "sha256": "3" * 64,
                    "dropout_executes_before_a_down_projection": True,
                    "dropout_identity_at_evaluation": True,
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                },
                {
                    "binding_id": "frozen_training_project_source_identity",
                    "binding_kind": "training_metadata_implementation_identity",
                    "path": "artifacts/q3/training_metadata.json",
                    "sha256": "4" * 64,
                    "current_project_source_matches_frozen_training_identity": True,
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                },
                {
                    "binding_id": "frozen_checkpoint_tensor_dtype_identity",
                    "binding_kind": "safetensors_header_dtype_and_shape_census",
                    "path": "artifacts/checkpoints",
                    "sha256": "5" * 64,
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                },
            ],
            "training_torch_version": "2.6.0+cu124",
            "training_transformers_version": "5.12.1",
            "training_peft_version": "0.19.1",
            "frozen_training_hyperparameters": {},
            "training_update_graph_failures": [],
            "training_coverage_is_single_step_semantic_primitive_census": True,
            "training_coverage_is_multiseed_causal_attribution": False,
            "operator_attribution_inferred_from_training_coverage": False,
            "inactive_training_path_count": 7,
            "inactive_training_paths": [
                {
                    "path_id": f"inactive_training_path_{index}",
                    "evidence_kind": "fixture",
                    "observed": {"value": False},
                    "expected": False,
                    "inactive_verified": True,
                    "explanation": "Frozen inactive training path fixture.",
                    "failures": [],
                    "scientific_support_inferred": False,
                    "operator_attribution_inferred": False,
                }
                for index in range(7)
            ],
            "all_inactive_training_paths_verified": True,
            "dynamic_runtime_or_source_backed_interfaces": [
                {
                    "interface_id": interface_id,
                    "binding_kind": "fixture_dynamic_contract",
                    "source_identities": [
                        {
                            "path": f"src/{interface_id}.py",
                            "sha256": "c" * 64,
                        }
                    ],
                    "runtime_identity_node": (
                        interface_id
                        if "residual" in interface_id
                        else (
                            "q_post_norm_pre_rope"
                            if interface_id.startswith("q_head_rmsnorm")
                            else (
                                "k_post_norm_pre_rope"
                                if interface_id.startswith("k_head_rmsnorm")
                                else None
                            )
                        )
                    ),
                    "runtime_identity_models": (
                        list(MODEL_IDS)
                        if "residual" in interface_id
                        or "head_rmsnorm" in interface_id
                        else []
                    ),
                    "runtime_algebra_key": (
                        "fixture_recomposition"
                        if interface_id
                        in {"attention_residual_addition", "mlp_residual_addition"}
                        else None
                    ),
                    "runtime_algebra_models": (
                        list(MODEL_IDS)
                        if interface_id
                        in {"attention_residual_addition", "mlp_residual_addition"}
                        else []
                    ),
                    "status": "completed",
                    "failures": [],
                    "scientific_support_inferred": False,
                }
                for interface_id in (
                    "serialization_tokenization",
                    "kv_cache_phase_boundary",
                    "q_head_rmsnorm_variance_rescale_and_gain",
                    "k_head_rmsnorm_variance_rescale_and_gain",
                    "block_input_residual",
                    "post_attention_residual",
                    "attention_residual_addition",
                    "block_output_residual",
                    "mlp_residual_addition",
                    "candidate_readout_positions",
                )
            ],
            "runtime_identity_smoke_count": 4,
            "runtime_hook_node_count": 18,
            "runtime_attention_backend": "sdpa",
            "runtime_identity_and_recomposition_validated": True,
            "superseded_runtime_lineage_count": 1,
            "superseded_runtime_lineage_retained": True,
            "files": {
                "base_config": {
                    "path": "models/huggingface/Qwen3-0.6B/config.json",
                    "sha256": "f" * 64,
                }
            },
            "runtime_identity_smokes": [
                {
                    "method_id": model_id,
                    "path": f"runs/{model_id}_identity/metadata.json",
                    "sha256": "e" * 64,
                    "attention_backend": "sdpa",
                    "hook_nodes_validated": 18,
                    "maximum_identity_error": 0.0,
                    "algebra_recomposition_passed": True,
                }
                for model_id in MODEL_IDS
            ],
            "superseded_runtime_lineage": [
                {
                    "run_id": "q1_identity_smoke_v2",
                    "method_id": MODEL_IDS[1],
                    "path": "runs/q1_identity_smoke_v2/metadata.json",
                    "sha256": "d" * 64,
                    "status": "failed_identity",
                    "failure_kind": "native_attention_wrapper_changed_q1_execution",
                    "maximum_identity_error": 14.86,
                    "identity_tolerance": 1e-5,
                    "canonical_replacement": "runs/q1_identity_smoke_v3/metadata.json",
                    "scientific_result_eligible": False,
                }
            ],
            "scientific_effect_values_read": False,
            "scientific_support_inferred": False,
        },
        "execution_axis_census": EXECUTION_AXIS_CENSUS,
        "frozen_observation_scope_contract": list(
            FROZEN_OBSERVATION_SCOPE_CONTRACT
        ),
        "frozen_observation_evidence": [
            dict(identity) for identity in FROZEN_OBSERVATION_EVIDENCE_IDENTITIES
        ],
        "frozen_observation_machine_snapshot": build_frozen_observation_snapshot(
            ROOT
        ),
        "prior_mechanism_diagnosis_snapshot": (
            build_prior_mechanism_diagnosis_snapshot(ROOT)
        ),
        "paper_method_stage_requirements": list(PAPER_METHOD_STAGE_REQUIREMENTS),
        "formal_layerwise_attenuation_profile": {
            "source": {
                "path": "runs/postblock/metrics.json",
                "sha256": "1" * 64,
            },
            "interpretation_boundary": "Layer profile localization boundary.",
            "shape_summary": [
                {
                    "method_id": MODEL_IDS[2],
                    "endpoint": "target_margin",
                    "shape": "localized_single_attenuation_step",
                    "significant_attenuation_steps": 1,
                    "significant_amplification_steps": 0,
                    "distributed_attenuation_pattern_established": False,
                    "registered_followup": "registered_component_decomposition",
                }
            ],
            "all_layer_rows": [
                {
                    "method_id": MODEL_IDS[2],
                    "endpoint": "target_margin",
                    "block_zero_based": 0,
                    "mean": 0.25,
                    "ci95": [0.2, 0.3],
                    "bh_q": 0.01,
                    "directional_description": "significant_sufficiency",
                }
            ],
            "adjacent_layer_rows": [
                {
                    "method_id": MODEL_IDS[2],
                    "endpoint": "target_margin",
                    "transition": "block_0_to_block_1",
                    "mean": -0.1,
                    "ci95": [-0.15, -0.05],
                    "bh_q": 0.02,
                    "directional_description": "significant_attenuation",
                }
            ],
        },
        "formal_attenuation_transition_profile": {
            "source": {
                "path": "runs/selected/metrics.json",
                "sha256": "2" * 64,
            },
            "interpretation_boundary": "Functional-node transition boundary.",
            "rows": [
                {
                    "method_id": MODEL_IDS[2],
                    "endpoint": "target_margin",
                    "transition": "attention__to__mlp",
                    "mean": -0.05,
                    "ci95": [-0.08, -0.02],
                    "bh_q": 0.03,
                    "directional_description": "significant_attenuation",
                    "evidence_role": "registered_confirmatory_branch_localization",
                }
            ],
        },
        "formal_architecture_opportunity_ranking": [
            {
                "rank": index,
                "opportunity_id": opportunity_id,
                "status": "deprioritized",
                "model_scope": [MODEL_IDS[2], MODEL_IDS[3]],
                "evidence_deliverables": ["d1_representation"],
                "rationale": "Formal bounded opportunity rationale.",
                "falsification_gate": "Formal bounded falsification gate.",
            }
            for index, opportunity_id in enumerate(
                (
                    "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER",
                    "OP_H2_ID_FREE_FACTORIZED_PREFERENCE_BOTTLENECK",
                    "OP_H3_CANDIDATE_CONDITIONED_SIGNED_PREFERENCE_RESIDUAL",
                    "OP_H2_H3_FACTORIZED_SIGNED_PREFERENCE_PATH",
                    "OP_H4_SURFACE_AWARE_GRADIENT_BUDGET",
                ),
                start=1,
            )
        ],
        "formal_execution_census": {
            "run_declaration_count": 84,
            "run_status_counts": {"completed": 80, "mechanical_failure": 4},
            "completed_result_eligible_runs": 76,
        },
        "reproducibility_ledger": {
            "frozen_assets": [
                {
                    "evidence_id": "plan.md",
                    "path": "plan.md",
                    "sha256": "a" * 64,
                }
            ],
            "formal_deliverables": [
                {
                    "evidence_id": "d1_representation",
                    "path": "runs/formal/metrics.json",
                    "sha256": "b" * 64,
                }
            ],
            "supplements": [
                {
                    "evidence_id": "d1_activation_anisotropy",
                    "path": "runs/supplement/metrics.json",
                    "sha256": "c" * 64,
                    "command": ["python", "analyze_supplement.py"],
                }
            ],
            "run_declarations": [
                {
                    "evidence_id": "run-1",
                    "analysis_stage": "attention",
                    "method_id": MODEL_IDS[2],
                    "status": "completed",
                    "result_eligible": True,
                    "path": "runs/run-1/metadata.json",
                    "sha256": "d" * 64,
                    "command": ["python", "score.py"],
                }
            ],
            "dev_eval_ledger": {"entry_count": 19, "sha256": "e" * 64},
            "commands_are_copied_from_audited_run_metadata": True,
        },
        "evidence_admission": {
            "readiness": {
                "formal": {"completed": 19, "registered": 19},
                "supplements": {"completed": 21, "registered": 21},
                "d2_causal_core": {"fixed_completed": 60, "fixed_registered": 60},
                **_readiness_component_coverage_fixture(),
                "component_coverage_interpretation": (
                    "Any-artifact coverage includes descriptive geometry; causal-role "
                    "completion does not infer scientific support."
                ),
                "mechanical_nonresults": {
                    "retained": 7,
                    "run_ids": [f"failure-{index}" for index in range(7)],
                    "interpretation": (
                        "Bound mechanical failures are neither scientific null results "
                        "nor evidence against a mechanism."
                    ),
                },
            },
            "source_test_opened": False,
            "formal_report": {"path": "formal.json", "sha256": "a" * 64},
            "supplement_registry": {"path": "registry.yaml", "sha256": "b" * 64},
            "supplement_registry_manifest": {
                "path": "manifest.yaml",
                "sha256": "c" * 64,
            },
        },
        "claim_invariants": {
            "source_test_opened": False,
            "diagnostic_patch_promoted_as_method": False,
            "architecture_implemented": False,
            "opportunity_utility_gain_established": False,
        },
    }
    payload["opportunity_lineage_matrix"] = _build_opportunity_lineage_matrix(
        prior=payload["prior_mechanism_diagnosis_snapshot"],
        formal={
            "architecture_opportunity_ranking": payload[
                "formal_architecture_opportunity_ranking"
            ]
        },
        comprehensive=payload,
    )
    markdown = render_comprehensive_report_markdown(payload)
    contract = _audit_report_section_contract(payload)
    assert contract["registered_sections"] == contract["covered_sections"] == 13
    assert len(contract["sections"]) == len(REPORT_SECTION_CONTRACT) == 13
    assert contract["scientific_effect_values_read_for_coverage"] is False
    assert "Covered：`13/13`" in markdown
    assert "19+21 项证据逐项处置账本" in markdown
    assert "S / N / specificity / controls / G 原始门矩阵" in markdown
    assert "Norm/direction/random" in markdown
    assert "optimization_opportunity_ranking" in markdown
    assert "正式深挖报告冻结的五项机会" in markdown
    assert "OP_H1_QUERY_CONDITIONED_SPARSE_ROUTER" in markdown
    assert "组件门控后的综合机会排序" in markdown
    assert "层扫描只用于界定后续因果分解" in markdown
    assert "strict_transfer_surface" in markdown
    assert "full_null_wrong_user" in markdown
    assert "Frozen shared-evaluator observation snapshot" in markdown
    assert "0.25671034382115254" in markdown
    assert "first_round_machine_summary" in markdown
    assert "Prior M0--M3 hypothesis state" in markdown
    assert "C_M3_MATCHED_CONTROL_ENDPOINT_REVERSAL" in markdown
    assert "首轮M0--M3原始产物字节：`18`项" in markdown
    assert "Prior logical-evidence to artifact mapping" in markdown
    assert "m3.q2.matched_did_statistics" in markdown
    assert "全部 post-block sufficiency 格" in markdown
    assert "block_0_to_block_1" in markdown
    assert "全部相邻功能节点变化" in markdown
    assert "attention__to__mlp" in markdown
    assert "runs/postblock/metrics.json" in markdown
    assert "runs/selected/metrics.json" in markdown
    assert "Finding evidence ledger" in markdown
    assert "F1: A bounded finding" in markdown
    assert "Do not infer component necessity" in markdown
    assert "d1_representation@" + "a" * 64 in markdown
    assert "Narrative evidence" in markdown
    assert "evidence bytes: d1_representation@" + "a" * 64 in markdown
    assert "Untested components" in markdown
    assert "保留的机械 non-results" in markdown
    assert "strict_transfer" in markdown
    assert "full_minus_null" in markdown
    assert "20260714" in markdown
    assert "insufficient_causal_evidence" in markdown
    assert "`failure-0`" in markdown
    assert "固定 claim invariants" in markdown
    assert "`diagnostic_patch_promoted_as_method`：`False`" in markdown
    assert "`architecture_implemented`：`False`" in markdown
    assert "进入论文方法阶段仍需通过的门槛" in markdown
    assert "尚无注册 operator-level 因果检验的精确接口" in markdown
    assert "结构化因果债务：`38`" in markdown
    assert '"inference_operator": 18' in markdown
    assert '"training_mechanism": 20' in markdown
    assert "本账本授权新增实验 family：`False`" in markdown
    assert "preregistered_swiglu_stage_intervention" in markdown
    assert "preregistered_rmsnorm_operator_intervention" in markdown
    assert "preregistered_q_head_rmsnorm_operator_intervention" in markdown
    assert "preregistered_k_head_rmsnorm_operator_intervention" in markdown
    assert "preregistered_attention_residual_composition_intervention" in markdown
    assert "preregistered_mlp_residual_composition_intervention" in markdown
    assert "preregistered_multiseed_training_control" in markdown
    assert "preregistered_multiseed_adapter_dropout_control" in markdown
    assert "preregistered_precision_equivalence_control" in markdown
    assert "preregistered_checkpoint_rng_equivalence_control" in markdown
    assert "preregistered_q3_q_adapter_branch_intervention" in markdown
    assert "preregistered_q3_v_adapter_branch_intervention" in markdown
    assert "unresolved_no_operator_causal_claim" in markdown
    assert "gqa_query_to_kv_grouping" in markdown
    assert "kv_cache_phase_boundary" in markdown
    assert "functional_causal_target" in markdown
    assert "replication_and_generalization" in markdown
    assert "跨模型共享功能现象" in markdown
    assert "shared_bounded_pattern" in markdown
    assert "跨模型异质功能现象" in markdown
    assert "heterogeneous_bounded_pattern" in markdown
    assert "使用绝对层号对齐：`False`" in markdown
    assert "跨模型对齐单位是功能节点" in markdown
    assert "从层扫描到设计证据的桥接" in markdown
    assert "cross_model_functional_replication" in markdown
    assert "Does attention route the relevant query-conditioned history" in markdown
    assert "Components with a registered causal-role artifact" in markdown
    assert "Components without a registered causal-role artifact" in markdown
    assert "`mlp_feature_formation`" in markdown
    assert "Causal role registered" in markdown
    assert "Causal role completed" in markdown
    assert "completed causal-role artifacts in both Q2/Q3" in markdown
    assert "精确 Transformer 实现接口清单" in markdown
    assert "跨接口或模型范围证据" in markdown
    assert "cross_interface_or_scope_gate" in markdown
    assert "全部 `40` 项正式与补充证据" in markdown
    assert "attention_scaled_qk_logits" in markdown
    assert "attention_softmax_edge_weights" in markdown
    assert "gqa_query_to_kv_grouping" in markdown
    assert "mlp_silu_gate" in markdown
    assert "kv_cache_phase_boundary" in markdown
    assert "Exact Transformer interfaces with any completed evidence" in markdown
    assert "Operator attribution inferred from artifact availability：`0/63`" in markdown
    assert (
        "operator attribution remains unresolved from artifact availability：`63/63`"
        in markdown
    )
    assert "仍未由产物可用性解决：`63`" in markdown
    assert "Frozen architecture audit：`completed`; failures=`0`" in markdown
    assert "Q heads/KV heads=`16/8`" in markdown
    assert "Frozen model pathways/base artifacts：`4/2`" in markdown
    assert "Q0–Q3 adaptation boundary" in markdown
    assert "q0_qwen3_reranker_06b=`qwen3_reranker_06b/full_parameter`" in markdown
    assert "q3_tallrec_generalqwen=`qwen3_general_06b/lora`" in markdown
    assert (
        "Q3 LoRA details：rank=`8` alpha=`16` training dropout=`0.05` "
        "targets=`q_proj,v_proj`" in markdown
    )
    assert "Config-backed exact interfaces：`53/63`; all present=`True`" in markdown
    assert "Dynamic runtime/source-backed exact interfaces：`10/63`" in markdown
    assert "Exact-interface implementation provenance：`63/63`; exhaustive=`True`" in markdown
    assert (
        "Frozen forward primitive coverage：primitives=`40`, inference interfaces=`43/43`, missing=`0`, exhaustive=`True`"
        in markdown
    )
    assert (
        "Installed forward-source bindings：`11` @ frozen Transformers `5.12.1` "
        "and PEFT `0.19.1`; checkpoint environment=`True`" in markdown
    )
    assert (
        "Frozen training-update primitive coverage：primitives=`27`, training "
        "interfaces=`20/20`, missing=`0`, exhaustive=`True`" in markdown
    )
    assert (
        "Installed training-source/artifact bindings：`12/3` @ Torch `2.6.0+cu124`, "
        "Transformers `5.12.1`, PEFT `0.19.1`" in markdown
    )
    assert "Frozen inactive training paths：`7/7`; exhaustive=`True`" in markdown
    assert "Frozen inactive architecture paths：`9/9`; exhaustive=`True`" in markdown
    assert (
        "semantic primitive census=`True`, kernel-instruction census=`False`, operator attribution inferred=`False`"
        in markdown
    )
    assert "Frozen inference forward primitive map" in markdown
    assert "Frozen training-update primitive map" in markdown
    assert "Installed training-update source bindings" in markdown
    assert "Frozen training artifact bindings" in markdown
    assert "Frozen inactive training paths" in markdown
    assert "inactive_training_path_0" in markdown
    assert "q3_lora_input_dropout" in markdown
    assert "q3_query_lora_scaled_adapter_injection" in markdown
    assert "q3_value_lora_scaled_adapter_injection" in markdown
    assert "current source matches frozen training identity=`True`" in markdown
    assert "project_input_serialization" in markdown
    assert "mlp_silu_nonlinearity" in markdown
    assert "q3_native_score_formula" in markdown
    assert "Installed Transformer forward source bindings" in markdown
    assert "forward_source_0" in markdown
    assert "Frozen inactive architecture paths" in markdown
    assert "inactive_path_0" in markdown
    assert "Dynamic interface implementation provenance" in markdown
    assert "project_owned_q1_cache_phase_source" not in markdown
    assert "fixture_dynamic_contract" in markdown
    assert "attention_residual_addition" in markdown
    assert "mlp_residual_addition" in markdown
    assert "algebra=`fixture_recomposition`" in markdown
    assert "五层精确接口因果角色覆盖" in markdown
    assert "Lower-bound causal debt" in markdown
    assert "Registered artifact claim ceilings" in markdown
    assert "Completed artifact claim ceilings" in markdown
    assert "Artifact claim ceiling registered / completed" in markdown
    assert '"G": 3' in markdown
    assert '"D": 35' in markdown
    assert '"none": 3' in markdown
    assert "gradient_accumulation_and_global_clip" in markdown
    assert "bfloat16_autocast_training_forward" in markdown
    assert "nonreentrant_gradient_checkpoint_recomputation" in markdown
    assert "q3_input_activation_requires_grad_bridge" in markdown
    assert "preregistered_q3_gradient_bridge_equivalence_control" in markdown
    assert "q3_fp32_lora_bf16_base_cast_boundary" in markdown
    assert "preregistered_q3_adapter_dtype_boundary_control" in markdown
    assert "frozen_checkpoint_tensor_dtype_identity" in markdown
    assert "adam_moment_preconditioned_direction" in markdown
    assert "decoupled_weight_decay_term" in markdown
    assert "learning_rate_scaled_effective_parameter_delta" in markdown
    assert '"S": 21' in markdown
    assert "autoregressive_causal_attention_mask" in markdown
    assert "q0_next_token_yes_no_logit_difference" in markdown
    assert "q1_candidate_response_mean_log_likelihood" in markdown
    assert "q2_next_token_yes_no_logit_difference" in markdown
    assert "q3_two_path_mean_log_likelihood_difference" in markdown
    assert "lora_q_low_rank_a_factor" in markdown
    assert "lora_q_low_rank_b_factor" in markdown
    assert "lora_q_effective_delta_weight" in markdown
    assert "lora_v_low_rank_a_factor" in markdown
    assert "lora_v_low_rank_b_factor" in markdown
    assert "lora_v_effective_delta_weight" in markdown
    assert '"none": 3' in markdown
    assert "q0_pointwise_bce_loss" in markdown
    assert "q1_normalized_response_nll" in markdown
    assert "不在表内的接口也不能据此自动获得 operator attribution" in markdown
    assert "Runtime instrumentation identity：models=`4`, hook nodes/model=`18`" in markdown
    assert "exact identity and BF16-bounded recomposition=`True`" in markdown
    assert "Frozen model/config identities" in markdown
    assert "models/huggingface/Qwen3-0.6B/config.json" in markdown
    assert "Runtime instrumentation identity smokes" in markdown
    assert "q3_tallrec_generalqwen_identity/metadata.json" in markdown
    assert "Superseded runtime mechanical lineage" in markdown
    assert "q1_identity_smoke_v2" in markdown
    assert "scientific result eligible=`False`" in markdown
    assert "descriptive geometry" in markdown
    assert "Retained mechanical non-results：`7`" in markdown
    assert "Declared runs：`84`" in markdown
    assert "primary_candidate_relative_mechanism_endpoint" in markdown
    assert "Hypothesized innovation" in markdown
    assert "Train-only different-item preference pairs" in markdown
    assert "Remove the continuity constraint" in markdown
    assert "CoPPS-style structural transfer witness" in markdown
    assert "Architecture implemented" in markdown
    assert "fixed_transition_confirmation" in markdown
    assert "Audited run metadata and commands" in markdown
    assert "runs/run-1/metadata.json" in markdown
    assert "python score.py" in markdown
    assert "python analyze_supplement.py" in markdown
    assert "neither scientific null" in markdown
    assert "registered harmful full-history response" in markdown
    assert "should be strengthened" in markdown
    assert "diagnostic_only" in markdown
    assert "Utility gain established" in markdown
    component_section = markdown.split(
        "### Component-model cells", 1
    )[1].split("## 5. 功能因果链", 1)[0]
    component_rows = [
        line
        for line in component_section.splitlines()
        if line.startswith("| ") and not line.startswith("|---")
    ]
    assert len(component_rows) == 1 + len(COMPONENT_IDS) * len(MODEL_IDS)
    for index in range(1, 14):
        assert f"## {index}." in markdown


def test_component_evidence_role_coverage_is_exact_and_count_bound() -> None:
    readiness = _readiness_component_coverage_fixture()
    normalized = _build_component_evidence_role_coverage(readiness)
    assert set(normalized) == set(COMPONENT_IDS)
    assert normalized["token_embedding"]["causal_role_artifact_registered"] is False
    assert normalized["attention_output"][
        "q2_q3_causal_role_artifacts_completed"
    ] is True
    assert normalized["attention_output"][
        "scientific_support_inferred_from_completion"
    ] is False

    broken = copy.deepcopy(readiness)
    broken["component_artifact_coverage"][0][
        "causal_role_model_scope_completed"
    ] = [MODEL_IDS[0]]
    with pytest.raises(ValueError, match="exceeds registration"):
        _build_component_evidence_role_coverage(broken)

    broken = copy.deepcopy(readiness)
    broken["components_with_registered_causal_role_artifact"] -= 1
    with pytest.raises(ValueError, match="registered count differs"):
        _build_component_evidence_role_coverage(broken)

    broken = copy.deepcopy(readiness)
    broken["component_artifact_coverage"][0][
        "q2_q3_causal_role_artifacts_completed"
    ] = False
    with pytest.raises(ValueError, match="Q2/Q3 causal-role completion disagrees"):
        _build_component_evidence_role_coverage(broken)


def test_thirteen_section_contract_fails_when_a_required_section_is_empty() -> None:
    payload = {
        path.split(".")[0]: {"placeholder": True}
        for row in REPORT_SECTION_CONTRACT
        for path in row["required_payload_paths"]
    }
    payload["formal_execution_census"] = {}
    with pytest.raises(ValueError, match="execution_and_evidence"):
        _audit_report_section_contract(payload)


def test_decisions_are_not_mutated() -> None:
    decisions = _decisions()
    original = copy.deepcopy(decisions)
    _validate(decisions)
    assert decisions == original
