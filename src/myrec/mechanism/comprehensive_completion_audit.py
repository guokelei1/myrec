"""Requirement-by-requirement completion audit for the Transformer deep dive."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from myrec.mechanism.comprehensive_readiness import build_comprehensive_readiness
from myrec.mechanism.comprehensive_report_builder import (
    ANALYSIS_TYPE,
    CLAIM_INVARIANTS,
    COMPREHENSIVE_REPORT_PLAN_IDENTITY,
    COMPONENT_FUNCTIONAL_QUESTIONS,
    DESIGN_GATE_SUPPLEMENT,
    EXECUTION_AXIS_CENSUS,
    FROZEN_OBSERVATION_EVIDENCE_IDENTITIES,
    FROZEN_OBSERVATION_SCOPE_CONTRACT,
    FUNCTIONAL_LOCALIZATION_CONTRACT,
    HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT,
    LOCALIZATION_TO_DESIGN_BRIDGE,
    NECESSITY_DIRECTION_CLAIM_BOUNDARY,
    PAPER_METHOD_STAGE_REQUIREMENTS,
    REPORT_SECTION_CONTRACT,
    _audit_comprehensive_against_formal,
    _audit_formal_report,
    _audit_report_section_contract,
    _bind_opportunity_evidence_identities,
    _build_component_bidirectional_gate_matrix,
    _build_component_evidence_role_coverage,
    _build_opportunity_lineage_matrix,
    _build_reproducibility_ledger,
    build_frozen_observation_snapshot,
    build_prior_mechanism_diagnosis_snapshot,
    _derive_necessity_component_models,
    render_comprehensive_report_markdown,
    validate_comprehensive_decisions,
)
from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_evidence_topology import MODEL_IDS
from myrec.mechanism.deep_dive_report_contract import COMPONENT_IDS
from myrec.mechanism.deep_dive_report_builder import REPORT_ANALYSIS_TYPE
from myrec.mechanism.supplemental_evidence_registry import (
    EXPECTED_SUPPLEMENT_IDS,
    audit_supplemental_evidence_registry,
)
from myrec.mechanism.transformer_interface_inventory import (
    build_transformer_interface_coverage,
)
from myrec.utils.hashing import sha256_file


FINAL_PATHS = {
    "decision_worksheet": Path(
        "experiments/motivation/transformer_comprehensive_decisions.json"
    ),
    "formal_report": Path("reports/motivation_transformer_deep_dive.json"),
    "comprehensive_json": Path("reports/motivation_transformer_comprehensive.json"),
    "comprehensive_markdown": Path("reports/motivation_transformer_comprehensive.md"),
}


def audit_comprehensive_completion(root: str | Path = ".") -> dict[str, Any]:
    """Prove terminal completion without recomputing scientific effects."""

    root_path = Path(root).resolve()
    readiness = build_comprehensive_readiness(root_path)
    interfaces = readiness["transformer_internal_interface_coverage"]
    architecture = readiness["frozen_model_architecture_audit"]
    requirements = [
        _requirement(
            "formal_19_closeout",
            readiness["formal"]["completed"] == readiness["formal"]["registered"] == 19,
            {
                "completed": readiness["formal"]["completed"],
                "registered": readiness["formal"]["registered"],
                "pending": readiness["formal"]["pending_deliverables"],
            },
        ),
        _requirement(
            "supplement_21_closeout",
            readiness["supplements"]["completed"]
            == readiness["supplements"]["registered"]
            == 21,
            {
                "completed": readiness["supplements"]["completed"],
                "registered": readiness["supplements"]["registered"],
                "pending": readiness["supplements"]["pending_evidence"],
            },
        ),
        _requirement(
            "d2_fixed_and_conditional_terminal",
            readiness["readiness_gates"][
                "d2_all_fixed_and_conditional_units_terminal"
            ],
            dict(readiness["d2_causal_core"]),
        ),
        _requirement(
            "component_18_coverage",
            readiness["component_count"] == 18
            and readiness["components_with_any_completed_artifact"] == 18,
            {
                "registered": readiness["component_count"],
                "with_any_completed_artifact": readiness[
                    "components_with_any_completed_artifact"
                ],
                "scientific_support_inferred": readiness[
                    "scientific_support_inferred_from_completion"
                ],
            },
        ),
        _requirement(
            "exact_interface_and_evidence_disposition",
            interfaces["interface_count"] == 63
            and interfaces[
                "all_registered_evidence_has_exact_or_cross_interface_disposition"
            ]
            is True
            and interfaces["direct_interface_evidence_count"]
            + interfaces["cross_interface_evidence_count"]
            == interfaces["registered_evidence_count"]
            == 40
            and interfaces["operator_causal_debt_count"] == 38
            and interfaces["operator_causal_debt_class_counts"]
            == {"inference_operator": 18, "training_mechanism": 20}
            and {
                row["interface_id"] for row in interfaces["operator_causal_debt"]
            }
            == set(interfaces["interfaces_without_registered_causal_role_evidence"])
            and all(
                row["active_experiment_authorized"] is False
                and row["can_rank_architecture_from_current_evidence"] is False
                for row in interfaces["operator_causal_debt"]
            )
            and interfaces["new_experiment_family_authorized_by_debt_ledger"]
            is False
            and interfaces["operator_causal_debt_is_lower_bound"] is True
            and interfaces["operator_attribution_inferred_for_other_interfaces"]
            is False
            and interfaces[
                "operator_attribution_inferred_from_artifact_availability_count"
            ]
            == 0
            and interfaces[
                "operator_attribution_unresolved_from_artifact_availability_count"
            ]
            == 63
            and len(
                interfaces[
                    "interfaces_with_functional_causal_role_but_no_operator_attribution_inferred"
                ]
            )
            == 25
            and interfaces["functional_causal_role_is_operator_attribution"] is False
            and all(
                row[
                    "operator_attribution_inferred_from_artifact_availability"
                ]
                is False
                for row in interfaces["interfaces"]
            )
            and set(interfaces["system_layer_coverage"])
            == {"input", "representation", "routing", "readout", "training"}
            and sum(
                row["interface_count"]
                for row in interfaces["system_layer_coverage"].values()
            )
            == 63
            and sum(
                row["operator_causal_debt_count"]
                for row in interfaces["system_layer_coverage"].values()
            )
            == 38
            and interfaces["registered_claim_ceiling_counts"]
            == {"none": 3, "M": 0, "D": 35, "S": 21, "N": 1, "G": 3}
            and interfaces["claim_ceilings_are_artifact_availability_only"] is True
            and interfaces["actual_scientific_evidence_levels_inferred"] is False
            and architecture["status"] == "completed"
            and not architecture["failures"]
            and architecture["config_backed_interfaces_present_in_inventory"]
            is True
            and architecture["config_backed_interface_count"] == 53
            and architecture[
                "dynamic_runtime_or_source_backed_interface_count"
            ]
            == 10
            and architecture["implementation_provenance_covered_interface_count"]
            == 63
            and architecture[
                "all_exact_interfaces_have_config_or_runtime_source_provenance"
            ]
            is True
            and architecture["forward_primitive_count"] == 40
            and architecture["forward_inference_interface_count"] == 43
            and architecture["forward_mapped_interface_count"] == 43
            and architecture["forward_missing_interface_ids"] == []
            and architecture["forward_extraneous_interface_ids"] == []
            and architecture["forward_training_interface_count"] == 20
            and len(
                architecture["forward_training_interfaces_excluded_by_design"]
            )
            == 20
            and architecture["forward_primitive_interface_coverage_complete"]
            is True
            and architecture["forward_source_binding_count"] == 11
            and all(
                row["status"] == "completed" and not row["failures"]
                for row in architecture["forward_source_bindings"]
            )
            and architecture["inactive_architecture_path_count"] == 9
            and architecture["all_inactive_architecture_paths_verified"] is True
            and all(
                row["inactive_verified"] is True
                and row["scientific_support_inferred"] is False
                and row["operator_attribution_inferred"] is False
                for row in architecture["inactive_architecture_paths"]
            )
            and architecture["forward_graph_failures"] == []
            and architecture["forward_coverage_is_semantic_primitive_census"]
            is True
            and architecture["forward_coverage_is_kernel_instruction_census"]
            is False
            and architecture["operator_attribution_inferred_from_forward_coverage"]
            is False
            and architecture["forward_peft_version"] == "0.19.1"
            and architecture[
                "forward_source_environment_is_frozen_checkpoint_environment"
            ]
            is True
            and architecture["training_primitive_count"] == 27
            and architecture["training_exact_interface_count"] == 20
            and architecture["training_mapped_interface_count"] == 20
            and architecture["training_missing_interface_ids"] == []
            and architecture["training_extraneous_interface_ids"] == []
            and architecture["training_nontraining_interface_count"] == 43
            and len(
                architecture[
                    "training_nontraining_interfaces_excluded_by_design"
                ]
            )
            == 43
            and architecture["training_primitive_interface_coverage_complete"]
            is True
            and architecture["training_source_binding_count"] == 12
            and architecture["training_artifact_binding_count"] == 3
            and all(
                row["status"] == "completed" and not row["failures"]
                for row in architecture["training_source_bindings"]
            )
            and all(
                row["status"] == "completed" and not row["failures"]
                for row in architecture["training_artifact_bindings"]
            )
            and architecture["training_peft_version"] == "0.19.1"
            and architecture["training_update_graph_failures"] == []
            and architecture[
                "training_coverage_is_single_step_semantic_primitive_census"
            ]
            is True
            and architecture[
                "training_coverage_is_multiseed_causal_attribution"
            ]
            is False
            and architecture[
                "operator_attribution_inferred_from_training_coverage"
            ]
            is False
            and architecture["inactive_training_path_count"] == 7
            and architecture["all_inactive_training_paths_verified"] is True
            and all(
                row["inactive_verified"] is True
                and row["scientific_support_inferred"] is False
                and row["operator_attribution_inferred"] is False
                for row in architecture["inactive_training_paths"]
            )
            and not any(
                row["failures"]
                for row in architecture[
                    "dynamic_runtime_or_source_backed_interfaces"
                ]
            )
            and architecture["runtime_identity_smoke_count"] == 4
            and architecture["runtime_hook_node_count"] == 18
            and architecture["runtime_identity_and_recomposition_validated"]
            is True
            and architecture["superseded_runtime_lineage_count"] == 1
            and architecture["superseded_runtime_lineage_retained"] is True
            and architecture["frozen_base_artifact_count"] == 2
            and architecture["frozen_model_pathway_count"] == 4
            and set(architecture["model_pathways"])
            == {
                "q0_qwen3_reranker_06b",
                "q1_instructrec_generalqwen",
                "q2_recranker_generalqwen",
                "q3_tallrec_generalqwen",
            }
            and {
                row["optimizer_steps"]
                for row in architecture["model_pathways"].values()
            }
            == {967},
            {
                "interfaces": interfaces["interface_count"],
                "direct_evidence": interfaces["direct_interface_evidence_count"],
                "cross_interface_evidence": interfaces[
                    "cross_interface_evidence_count"
                ],
                "registered_evidence": interfaces["registered_evidence_count"],
                "operator_causal_debts": interfaces[
                    "operator_causal_debt_count"
                ],
                "operator_causal_debt_class_counts": interfaces[
                    "operator_causal_debt_class_counts"
                ],
                "new_experiment_family_authorized": interfaces[
                    "new_experiment_family_authorized_by_debt_ledger"
                ],
                "operator_causal_debt_is_lower_bound": interfaces[
                    "operator_causal_debt_is_lower_bound"
                ],
                "operator_attribution_inferred_for_other_interfaces": interfaces[
                    "operator_attribution_inferred_for_other_interfaces"
                ],
                "operator_attribution_inferred_from_artifact_availability_count": interfaces[
                    "operator_attribution_inferred_from_artifact_availability_count"
                ],
                "operator_attribution_unresolved_from_artifact_availability_count": interfaces[
                    "operator_attribution_unresolved_from_artifact_availability_count"
                ],
                "functional_causal_role_is_operator_attribution": interfaces[
                    "functional_causal_role_is_operator_attribution"
                ],
                "system_layer_coverage": interfaces["system_layer_coverage"],
                "registered_claim_ceiling_counts": interfaces[
                    "registered_claim_ceiling_counts"
                ],
                "completed_artifact_claim_ceiling_counts": interfaces[
                    "completed_artifact_claim_ceiling_counts"
                ],
                "claim_ceilings_are_artifact_availability_only": interfaces[
                    "claim_ceilings_are_artifact_availability_only"
                ],
                "actual_scientific_evidence_levels_inferred": interfaces[
                    "actual_scientific_evidence_levels_inferred"
                ],
                "frozen_model_architecture_audit_status": architecture["status"],
                "config_backed_interfaces": architecture[
                    "config_backed_interface_count"
                ],
                "config_backed_interfaces_present": architecture[
                    "config_backed_interfaces_present_in_inventory"
                ],
                "dynamic_runtime_or_source_backed_interfaces": architecture[
                    "dynamic_runtime_or_source_backed_interface_count"
                ],
                "implementation_provenance_covered_interfaces": architecture[
                    "implementation_provenance_covered_interface_count"
                ],
                "all_exact_interfaces_have_config_or_runtime_source_provenance": architecture[
                    "all_exact_interfaces_have_config_or_runtime_source_provenance"
                ],
                "forward_primitives": architecture["forward_primitive_count"],
                "forward_inference_interfaces": architecture[
                    "forward_inference_interface_count"
                ],
                "forward_mapped_interfaces": architecture[
                    "forward_mapped_interface_count"
                ],
                "forward_missing_interfaces": architecture[
                    "forward_missing_interface_ids"
                ],
                "forward_source_bindings": architecture[
                    "forward_source_binding_count"
                ],
                "forward_peft_version": architecture["forward_peft_version"],
                "forward_source_environment_is_frozen_checkpoint_environment": architecture[
                    "forward_source_environment_is_frozen_checkpoint_environment"
                ],
                "inactive_architecture_paths": architecture[
                    "inactive_architecture_path_count"
                ],
                "forward_primitive_interface_coverage_complete": architecture[
                    "forward_primitive_interface_coverage_complete"
                ],
                "all_inactive_architecture_paths_verified": architecture[
                    "all_inactive_architecture_paths_verified"
                ],
                "operator_attribution_inferred_from_forward_coverage": architecture[
                    "operator_attribution_inferred_from_forward_coverage"
                ],
                "training_primitives": architecture["training_primitive_count"],
                "training_exact_interfaces": architecture[
                    "training_exact_interface_count"
                ],
                "training_mapped_interfaces": architecture[
                    "training_mapped_interface_count"
                ],
                "training_missing_interfaces": architecture[
                    "training_missing_interface_ids"
                ],
                "training_source_bindings": architecture[
                    "training_source_binding_count"
                ],
                "training_artifact_bindings": architecture[
                    "training_artifact_binding_count"
                ],
                "training_primitive_interface_coverage_complete": architecture[
                    "training_primitive_interface_coverage_complete"
                ],
                "training_peft_version": architecture["training_peft_version"],
                "training_coverage_is_multiseed_causal_attribution": architecture[
                    "training_coverage_is_multiseed_causal_attribution"
                ],
                "operator_attribution_inferred_from_training_coverage": architecture[
                    "operator_attribution_inferred_from_training_coverage"
                ],
                "inactive_training_paths": architecture[
                    "inactive_training_path_count"
                ],
                "all_inactive_training_paths_verified": architecture[
                    "all_inactive_training_paths_verified"
                ],
                "runtime_identity_smokes": architecture[
                    "runtime_identity_smoke_count"
                ],
                "runtime_hook_nodes": architecture["runtime_hook_node_count"],
                "runtime_identity_and_recomposition_validated": architecture[
                    "runtime_identity_and_recomposition_validated"
                ],
                "superseded_runtime_lineage": architecture[
                    "superseded_runtime_lineage_count"
                ],
                "superseded_runtime_lineage_retained": architecture[
                    "superseded_runtime_lineage_retained"
                ],
                "frozen_base_artifacts": architecture[
                    "frozen_base_artifact_count"
                ],
                "frozen_model_pathways": architecture[
                    "frozen_model_pathway_count"
                ],
                "model_pathways": architecture["model_pathways"],
                "scientific_support_inferred": interfaces[
                    "scientific_support_inferred_from_completion"
                ],
            },
        ),
        _requirement(
            "producer_topology",
            readiness["readiness_gates"]["producer_topology_complete"]
            and not readiness["producer_topology"]["failures"],
            dict(readiness["producer_topology"]),
        ),
        _requirement(
            "source_test_closed",
            readiness["readiness_gates"]["source_test_unopened"]
            and readiness["source_test_opened"] is False,
            {
                "source_test_opened": readiness["source_test_opened"],
                "qrels_files_opened_by_readiness": readiness[
                    "qrels_files_opened_by_this_readiness"
                ],
            },
        ),
        _requirement(
            "no_readiness_audit_failures",
            readiness["status"] != "failed"
            and readiness["readiness_gates"]["no_closeout_audit_failures"]
            and readiness["readiness_gates"]["no_supplement_registry_failures"],
            {"readiness_status": readiness["status"]},
        ),
    ]
    requirements.extend(_audit_final_files(root_path))
    complete = all(row["status"] == "completed" for row in requirements)
    return {
        "schema_version": 1,
        "analysis_type": "transformer_comprehensive_completion_audit",
        "status": "completed" if complete else "pending",
        "requirements_completed": sum(
            row["status"] == "completed" for row in requirements
        ),
        "requirements_registered": len(requirements),
        "pending_requirement_ids": [
            row["requirement_id"]
            for row in requirements
            if row["status"] != "completed"
        ],
        "requirements": requirements,
        "scientific_effect_values_recomputed": False,
        "scientific_effect_values_used_for_completion": False,
        "qrels_files_opened": False,
        "source_test_opened": False,
    }


def _audit_final_files(root: Path) -> list[dict[str, Any]]:
    worksheet = _load_optional_json(root / FINAL_PATHS["decision_worksheet"])
    formal = _load_optional_json(root / FINAL_PATHS["formal_report"])
    comprehensive = _load_optional_json(root / FINAL_PATHS["comprehensive_json"])
    markdown_path = root / FINAL_PATHS["comprehensive_markdown"]

    formal_validation_errors: list[str] = []
    if formal:
        try:
            _audit_formal_report(formal)
            _audit_formal_input_identities(root, formal)
        except (KeyError, OSError, TypeError, ValueError) as exc:
            formal_validation_errors.append(f"{type(exc).__name__}: {exc}")

    chain_validation_errors: list[str] = []
    chain_validation_attempted = bool(
        worksheet and formal and comprehensive and markdown_path.is_file()
    )
    if chain_validation_attempted:
        try:
            _audit_terminal_report_chain(
                root,
                worksheet=worksheet,
                formal=formal,
                comprehensive=comprehensive,
                markdown_text=markdown_path.read_text(encoding="utf-8"),
            )
        except (KeyError, OSError, TypeError, ValueError) as exc:
            chain_validation_errors.append(f"{type(exc).__name__}: {exc}")
    chain_ok = not chain_validation_attempted or not chain_validation_errors

    worksheet_ok = bool(
        worksheet
        and worksheet.get("worksheet_status") == "final"
        and worksheet.get("schema_version") == 1
        and chain_ok
    )
    formal_ok = bool(
        formal
        and formal.get("status") == "completed"
        and formal.get("analysis_type") == REPORT_ANALYSIS_TYPE
        and not formal_validation_errors
        and chain_ok
    )
    section_contract = comprehensive.get("report_section_contract", {}) if comprehensive else {}
    component_matrix = comprehensive.get("component_matrix", {}) if comprehensive else {}
    component_model_matrix = (
        comprehensive.get("component_model_matrix", {}) if comprehensive else {}
    )
    exact_component_model_matrix = bool(
        isinstance(component_matrix, Mapping)
        and set(component_matrix) == set(COMPONENT_IDS)
        and isinstance(component_model_matrix, Mapping)
        and set(component_model_matrix) == set(COMPONENT_IDS)
        and all(
            isinstance(component_model_matrix[component_id], Mapping)
            and set(component_model_matrix[component_id]) == set(MODEL_IDS)
            for component_id in COMPONENT_IDS
        )
    )
    comprehensive_ok = bool(
        comprehensive
        and comprehensive.get("status") == "completed"
        and comprehensive.get("analysis_type") == ANALYSIS_TYPE
        and section_contract.get("covered_sections") == len(REPORT_SECTION_CONTRACT)
        and section_contract.get("registered_sections") == len(REPORT_SECTION_CONTRACT)
        and comprehensive.get("frozen_model_architecture_audit", {}).get("status")
        == "completed"
        and not comprehensive.get("frozen_model_architecture_audit", {}).get(
            "failures"
        )
        and comprehensive.get("claim_invariants", {}).get("source_test_opened")
        is False
        and exact_component_model_matrix
        and chain_ok
    )
    markdown_ok = (
        markdown_path.is_file()
        and markdown_path.stat().st_size > 0
        and chain_ok
    )
    return [
        _requirement(
            "final_human_interpretation_worksheet",
            worksheet_ok,
            {
                **_file_evidence(root, FINAL_PATHS["decision_worksheet"]),
                "terminal_chain_validation_attempted": chain_validation_attempted,
                "terminal_chain_validation_errors": chain_validation_errors,
            },
        ),
        _requirement(
            "formal_deep_dive_report",
            formal_ok,
            {
                **_file_evidence(root, FINAL_PATHS["formal_report"]),
                "formal_semantic_validation_errors": formal_validation_errors,
                "terminal_chain_validation_errors": chain_validation_errors,
            },
        ),
        _requirement(
            "comprehensive_json_and_13_section_contract",
            comprehensive_ok,
            {
                **_file_evidence(root, FINAL_PATHS["comprehensive_json"]),
                "terminal_chain_validation_attempted": chain_validation_attempted,
                "terminal_chain_validation_errors": chain_validation_errors,
            },
        ),
        _requirement(
            "comprehensive_markdown_report",
            markdown_ok,
            {
                **_file_evidence(root, FINAL_PATHS["comprehensive_markdown"]),
                "rendered_from_current_json_exactly": bool(
                    chain_validation_attempted and not chain_validation_errors
                ),
                "terminal_chain_validation_errors": chain_validation_errors,
            },
        ),
    ]


def _audit_terminal_report_chain(
    root: Path,
    *,
    worksheet: Mapping[str, Any],
    formal: Mapping[str, Any],
    comprehensive: Mapping[str, Any],
    markdown_text: str,
) -> None:
    """Revalidate terminal prose, gates, and every admitted byte identity."""

    plan_path = root / COMPREHENSIVE_REPORT_PLAN_IDENTITY["path"]
    if sha256_file(plan_path) != COMPREHENSIVE_REPORT_PLAN_IDENTITY["sha256"]:
        raise ValueError("comprehensive report plan hash drift")

    _audit_formal_report(formal)
    _audit_formal_input_identities(root, formal)

    readiness = build_comprehensive_readiness(root)
    if readiness.get("status") != "completed" or readiness.get(
        "final_comprehensive_report_ready"
    ) is not True:
        raise ValueError("current readiness is not terminal")
    component_evidence_role_coverage = _build_component_evidence_role_coverage(
        readiness
    )

    supplements = audit_supplemental_evidence_registry(root)
    if supplements.get("status") != "completed":
        raise ValueError("current supplemental evidence registry is not terminal")
    completed_supplements = {
        str(row["evidence_id"]): row for row in supplements.get("entries", [])
    }
    if set(completed_supplements) != set(EXPECTED_SUPPLEMENT_IDS) or any(
        row.get("status") != "completed"
        for row in completed_supplements.values()
    ):
        raise ValueError("current supplemental evidence coverage is incomplete")

    interface_coverage = build_transformer_interface_coverage(
        completed_formal=set(EXPECTED_DELIVERABLES),
        completed_supplements=set(completed_supplements),
        supplement_model_scopes={
            evidence_id: set(row["model_scope"])
            for evidence_id, row in completed_supplements.items()
        },
        supplement_component_scopes={
            evidence_id: set(row["components"])
            for evidence_id, row in completed_supplements.items()
        },
    )
    if readiness.get("transformer_internal_interface_coverage") != interface_coverage:
        raise ValueError("current Transformer interface coverage differs")

    design_identity = completed_supplements[DESIGN_GATE_SUPPLEMENT]
    design_payload = _load_identity_json(
        root, design_identity, label=DESIGN_GATE_SUPPLEMENT
    )
    component_gate_matrix = _build_component_bidirectional_gate_matrix(
        design_payload,
        evidence_identity=design_identity,
    )
    design_nodes = set(
        component_gate_matrix["cross_model"]["design_prioritized_nodes"]
    )
    necessity_component_models = _derive_necessity_component_models(
        design_payload
    )
    normalized = validate_comprehensive_decisions(
        worksheet,
        completed_formal=set(EXPECTED_DELIVERABLES),
        completed_supplements=set(completed_supplements),
        supplement_metadata=completed_supplements,
        design_qualified_nodes=design_nodes,
        necessity_supported_component_models=necessity_component_models,
    )
    _audit_comprehensive_against_formal(normalized, formal)
    normalized = _bind_opportunity_evidence_identities(
        normalized,
        formal=formal,
        completed_supplements=completed_supplements,
    )
    reproducibility_ledger = _build_reproducibility_ledger(
        formal=formal,
        completed_supplements=completed_supplements,
        root=root,
    )
    frozen_observation_snapshot = build_frozen_observation_snapshot(root)
    prior_mechanism_snapshot = build_prior_mechanism_diagnosis_snapshot(root)
    opportunity_lineage_matrix = _build_opportunity_lineage_matrix(
        prior=prior_mechanism_snapshot,
        formal=formal,
        comprehensive=normalized,
    )

    formal_path = root / FINAL_PATHS["formal_report"]
    expected_admission = {
        "readiness": readiness,
        "formal_report": {
            "path": FINAL_PATHS["formal_report"].as_posix(),
            "sha256": sha256_file(formal_path),
            "analysis_type": formal["analysis_type"],
        },
        "supplement_registry": supplements["registry"],
        "supplement_registry_manifest": supplements["registry_manifest"],
        "supplements": list(completed_supplements.values()),
        "component_design_gate": {
            "path": design_identity["path"],
            "sha256": design_identity["sha256"],
            "cross_model_design_qualified_nodes": sorted(design_nodes),
        },
        "source_test_opened": False,
        "qrels_or_score_bundles_opened_by_this_builder": False,
    }
    expected_generated = {
        "comprehensive_report_plan": dict(COMPREHENSIVE_REPORT_PLAN_IDENTITY),
        "evidence_admission": expected_admission,
        "formal_execution_census": formal.get("execution_census", {}),
        "formal_layerwise_attenuation_profile": formal.get(
            "layerwise_attenuation_profile", {}
        ),
        "formal_attenuation_transition_profile": formal.get(
            "attenuation_transition_profile", {}
        ),
        "formal_architecture_opportunity_ranking": formal.get(
            "architecture_opportunity_ranking", []
        ),
        "execution_axis_census": EXECUTION_AXIS_CENSUS,
        "frozen_observation_scope_contract": list(
            FROZEN_OBSERVATION_SCOPE_CONTRACT
        ),
        "frozen_observation_evidence": [
            dict(identity) for identity in FROZEN_OBSERVATION_EVIDENCE_IDENTITIES
        ],
        "frozen_observation_machine_snapshot": frozen_observation_snapshot,
        "prior_mechanism_diagnosis_snapshot": prior_mechanism_snapshot,
        "opportunity_lineage_matrix": opportunity_lineage_matrix,
        "paper_method_stage_requirements": list(PAPER_METHOD_STAGE_REQUIREMENTS),
        "reproducibility_ledger": reproducibility_ledger,
        "functional_localization_contract": FUNCTIONAL_LOCALIZATION_CONTRACT,
        "localization_to_design_bridge": list(LOCALIZATION_TO_DESIGN_BRIDGE),
        "component_bidirectional_gate_matrix": component_gate_matrix,
        "necessity_direction_claim_boundary": dict(
            NECESSITY_DIRECTION_CLAIM_BOUNDARY
        ),
        "component_functional_questions": COMPONENT_FUNCTIONAL_QUESTIONS,
        "component_evidence_role_coverage": component_evidence_role_coverage,
        "transformer_internal_interface_coverage": interface_coverage,
        "history_signal_observation_scope_contract": list(
            HISTORY_SIGNAL_OBSERVATION_SCOPE_CONTRACT
        ),
        "frozen_model_architecture_audit": readiness[
            "frozen_model_architecture_audit"
        ],
        "claim_invariants": dict(CLAIM_INVARIANTS),
    }
    if (
        comprehensive.get("schema_version") != 1
        or comprehensive.get("analysis_type") != ANALYSIS_TYPE
        or comprehensive.get("status") != "completed"
        or comprehensive.get("report_id") != normalized["report_id"]
    ):
        raise ValueError("comprehensive report header differs from current worksheet")
    for key, expected in {**expected_generated, **normalized}.items():
        if comprehensive.get(key) != expected:
            raise ValueError(f"comprehensive report field is stale or altered: {key}")

    expected_section_contract = _audit_report_section_contract(comprehensive)
    if comprehensive.get("report_section_contract") != expected_section_contract:
        raise ValueError("comprehensive report section contract differs")
    expected_keys = {
        "schema_version",
        "analysis_type",
        "status",
        "report_section_contract",
        *expected_generated,
        *normalized,
    }
    if set(comprehensive) != expected_keys:
        raise ValueError("comprehensive report top-level field coverage differs")

    rendered = render_comprehensive_report_markdown(comprehensive)
    if markdown_text != rendered:
        raise ValueError("comprehensive Markdown is stale or not rendered from JSON")


def _audit_formal_input_identities(
    root: Path, formal: Mapping[str, Any]
) -> None:
    admission = formal.get("evidence_admission")
    if not isinstance(admission, Mapping):
        raise ValueError("formal report evidence admission is missing")
    if (
        admission.get("source_test_content_read_by_closeout_audit") is not False
        or admission.get("qrels_content_read_by_closeout_audit") is not False
    ):
        raise ValueError("formal report test/qrels audit boundary differs")

    frozen_assets = admission.get("frozen_assets")
    if not isinstance(frozen_assets, Mapping) or not frozen_assets:
        raise ValueError("formal report frozen-asset identities are missing")
    for path, digest in frozen_assets.items():
        _validate_file_identity(
            root,
            {"path": path, "sha256": digest},
            label=f"formal frozen asset {path}",
        )

    deliverables = admission.get("deliverables")
    if not isinstance(deliverables, Mapping) or set(deliverables) != set(
        EXPECTED_DELIVERABLES
    ):
        raise ValueError("formal report deliverable identity coverage differs")
    for evidence_id, identity in deliverables.items():
        if not isinstance(identity, Mapping) or identity.get("status") != "completed":
            raise ValueError(f"formal deliverable is not completed: {evidence_id}")
        _validate_file_identity(
            root, identity, label=f"formal deliverable {evidence_id}"
        )

    declarations = admission.get("run_declarations")
    if not isinstance(declarations, list) or not declarations:
        raise ValueError("formal report run declarations are missing")
    for index, identity in enumerate(declarations):
        if not isinstance(identity, Mapping):
            raise ValueError("formal run declaration is not an object")
        _validate_file_identity(
            root, identity, label=f"formal run declaration {index}"
        )

    dev_eval = admission.get("dev_eval_ledger")
    if not isinstance(dev_eval, Mapping):
        raise ValueError("formal dev-eval ledger identity is missing")
    _validate_file_identity(root, dev_eval, label="formal dev-eval ledger")


def _load_identity_json(
    root: Path, identity: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    path = _validate_file_identity(root, identity, label=label)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"identity JSON is not an object: {label}")
    return value


def _validate_file_identity(
    root: Path, identity: Mapping[str, Any], *, label: str
) -> Path:
    raw_path = identity.get("path")
    digest = identity.get("sha256")
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError(f"file identity path is missing: {label}")
    relative = Path(raw_path)
    if relative.is_absolute():
        raise ValueError(f"file identity path must be repository-relative: {label}")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"file identity escapes repository root: {label}") from exc
    if not path.is_file():
        raise ValueError(f"file identity target is missing: {label}")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"file identity SHA-256 is invalid: {label}")
    if sha256_file(path) != digest:
        raise ValueError(f"file identity SHA-256 drift: {label}")
    return path


def _requirement(
    requirement_id: str, completed: bool, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "requirement_id": requirement_id,
        "status": "completed" if completed else "pending",
        "evidence": dict(evidence),
    }


def _load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _file_evidence(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "present": path.is_file(),
        "sha256": sha256_file(path) if path.is_file() else None,
    }
