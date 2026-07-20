"""Build the final Transformer deep-dive report after fail-closed admission."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_DELIVERABLE_MODEL_COVERAGE,
    COMPONENT_IDS,
    COMPONENT_PROBE_CLAIM_BOUNDARIES,
    COMPONENT_SUPPORT_MECHANICAL_DEPENDENCIES,
    COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE,
    HYPOTHESIS_SUPPORTED_COMPONENT_REQUIREMENTS,
    HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS,
    OPPORTUNITY_ALLOWED_MODEL_SCOPE,
    OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS,
    OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS,
    PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE,
    PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE,
    PRIMARY_ATTRIBUTION_FOLD_SCOPE,
    PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE,
    PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE,
    REQUIRED_ASSERTIONS,
    RESULT_LEVEL_ANALYSIS_TYPES,
    RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS,
    RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES,
    RESULT_LEVEL_SUPPORTED_COMPONENT_GATE_DESCRIPTIONS,
    RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES,
    TRANSFER_EXPLANATION_LADDER_SCOPE,
    TRANSFER_EXPLANATION_LEVELS,
    component_result_practical_equivalence,
    component_result_support,
    derive_cross_model_primary_attribution,
    derive_primary_attribution_evidence,
    validate_deep_dive_report_against_closeout,
)
from myrec.mechanism.deep_dive_evidence_topology import (
    DELIVERABLE_MODEL_COVERAGE,
    MODEL_IDS,
)
from myrec.mechanism.deep_dive_progress import SELECTED_NODES, audit_deep_dive_progress
from myrec.mechanism.postblock_sweep_evaluator import POSTBLOCK_BLOCKS
from myrec.mechanism.deep_dive_opportunity_catalog import (
    OPPORTUNITY_DESIGN_CATALOG_SHA256,
    OPPORTUNITY_IDS,
    OPPORTUNITY_STAGE_BOUNDARY,
)
from myrec.mechanism.deep_dive_overview import component_model_coverage_debt
from myrec.utils.hashing import sha256_file


REPORT_ANALYSIS_TYPE = "transformer_deep_dive_mechanism_closeout"

# These are result-structure counts, not outcome-selected summaries.  Some
# deliverables intentionally expose both a family container and its flattened
# rows, so counts must not be added across structures as independent tests.
DELIVERABLE_RESULT_STRUCTURE_SPECS = {
    "d1_representation": (
        ("cells", "list", 96, "confirmatory_family"),
    ),
    "d2_q3_native_gate": (
        ("block_results", "mapping", 2, "confirmatory_gate_blocks"),
    ),
    "d2_postblock": (
        ("all_layer.*", "list", 60, "confirmatory_families"),
        (
            "adjacent_transition.*",
            "list",
            56,
            "confirmatory_families",
        ),
    ),
    "d2_selected_branches": (
        ("families", "mapping", 10, "confirmatory_family_containers"),
        ("rows", "list", 192, "confirmatory_flattened_rows"),
    ),
    "d3_attention_edges": (
        ("family_rows", "list", 36, "confirmatory_family"),
    ),
    "d3_attention_heads": (
        ("results.*", "mapping", 6, "descriptive_model_blocks"),
    ),
    "d3_attention_groups": (
        ("results.*.*.groups", "list", 48, "descriptive_gqa_groups"),
    ),
    "d4_mlp_groups": (
        ("results.*.*.groups", "list", 96, "descriptive_mlp_groups"),
    ),
    "d5_context": (
        ("family_rows", "list", 8, "confirmatory_family"),
    ),
    "d5_rope": (
        ("family_rows", "list", 36, "confirmatory_family"),
    ),
    "d6_q2_native_readout": (
        ("family_rows", "list", 12, "confirmatory_family"),
    ),
    "d6_q3_native_readout": (
        ("family_rows", "list", 24, "confirmatory_family"),
    ),
    "d6_q0_trajectory": (
        ("geometry_rows", "list", None, "registered_descriptive_geometry"),
        ("region_rows", "list", None, "registered_descriptive_regions"),
    ),
    "d6_q1_trajectory": (
        ("geometry_rows", "list", None, "registered_descriptive_geometry"),
        ("region_rows", "list", None, "registered_descriptive_regions"),
    ),
    "d6_q0_q1_branches": (
        ("family_rows", "list", 96, "confirmatory_family"),
    ),
    "d6_q0_q1_readouts": (
        ("rows", "list", 16, "registered_descriptive_rows"),
    ),
    "d7_q2_objective": (
        ("family_rows", "list", 12, "confirmatory_family"),
    ),
    "d7_q3_lora_path": (
        ("states", "mapping_or_list", 3, "registered_descriptive_states"),
        ("comparisons", "list", 168, "registered_descriptive_comparisons"),
    ),
    "d7_optimizer_replay": (
        (
            "q2.results.*.*.objectives",
            "mapping",
            18,
            "descriptive_objective_cells",
        ),
        (
            "q3.results.*.*.modes",
            "mapping",
            18,
            "descriptive_coordinate_cells",
        ),
    ),
}


def build_deep_dive_report(
    root: str | Path,
    decisions: Mapping[str, Any],
    *,
    json_output: str | Path,
    markdown_output: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate all evidence first, then atomically write JSON and Markdown."""

    root_path = Path(root).resolve()
    closeout = validate_deep_dive_report_against_closeout(root_path, decisions)
    json_path = _resolve_output(root_path, json_output)
    markdown_path = _resolve_output(root_path, markdown_output)
    if json_path == markdown_path:
        raise ValueError("JSON and Markdown outputs must differ")
    if not overwrite:
        existing = [str(path) for path in (json_path, markdown_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite report outputs: {existing}")

    d2_progress = audit_deep_dive_progress(root_path)
    if (
        d2_progress.get("status") != "ok"
        or d2_progress["fixed"]["remaining_mandatory_bundles"] != 0
        or d2_progress["maximum_remaining_scientific_bundles"] != 0
    ):
        raise ValueError("D2 scientific-bundle progress is not terminal")

    payload = {
        "schema_version": 1,
        "analysis_type": REPORT_ANALYSIS_TYPE,
        "status": "completed",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "evidence_admission": {
            "closeout_status": closeout["status"],
            "frozen_assets": closeout["frozen_assets"],
            "deliverables": closeout["deliverables"],
            "mechanical_failure_records": closeout[
                "mechanical_failure_records"
            ],
            "run_declarations": closeout["run_declarations"],
            "formal_completed_integrity_checked": closeout[
                "formal_completed_integrity_checked"
            ],
            "dev_eval_ledger": closeout["dev_eval_ledger"],
            "source_test_content_read_by_closeout_audit": closeout[
                "source_test_content_read_by_this_audit"
            ],
            "qrels_content_read_by_closeout_audit": closeout[
                "qrels_content_read_by_this_audit"
            ],
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
            "allowed_model_scope": {
                opportunity_id: sorted(OPPORTUNITY_ALLOWED_MODEL_SCOPE[opportunity_id])
                for opportunity_id in OPPORTUNITY_IDS
            },
        },
        "execution_census": _execution_census(closeout),
        "registered_result_structure_census": (
            _registered_result_structure_census(root_path, closeout)
        ),
        "d2_causal_core_progress": d2_progress,
        "layerwise_attenuation_profile": _layerwise_attenuation_profile(
            root_path, closeout
        ),
        "attenuation_transition_profile": _attenuation_transition_profile(
            root_path, closeout
        ),
        "component_probe_coverage": _component_probe_coverage(),
        "component_model_coverage_debt": component_model_coverage_debt(),
        "component_result_gate_catalog": _component_result_gate_catalog(),
        "component_practical_equivalence_gate_catalog": (
            _component_practical_equivalence_gate_catalog()
        ),
        "component_practical_equivalence_gate_outcomes": (
            _component_practical_equivalence_gate_outcomes(root_path, closeout)
        ),
        "component_result_gate_outcomes": _component_result_gate_outcomes(
            root_path, closeout
        ),
        "decision_gate_catalog": _decision_gate_catalog(),
        "primary_attribution_evidence_census": (
            _primary_attribution_evidence_census(root_path, closeout)
        ),
        **dict(decisions),
    }
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown_text = render_deep_dive_report_markdown(payload)
    _atomic_write_pair(json_path, json_text, markdown_path, markdown_text)
    return payload


def render_deep_dive_report_markdown(payload: Mapping[str, Any]) -> str:
    """Render a complete, compact human-readable report from validated data."""

    narratives = payload["narratives"]
    lines = [
        "# Transformer Deep-Dive Mechanism Closeout",
        "",
        f"Status: `{payload['status']}`  ",
        f"Generated at: `{payload['generated_at']}`",
        f"Opportunity catalog: `{payload['opportunity_design_catalog']['sha256']}`",
        "",
        "## Execution and evidence census",
        "",
        f"- Run declarations: `{payload['execution_census']['run_declaration_count']}`",
        f"- Run status counts: `{json.dumps(payload['execution_census']['run_status_counts'], sort_keys=True)}`",
        f"- Run stage counts: `{json.dumps(payload['execution_census']['run_stage_counts'], sort_keys=True)}`",
        f"- Formal run stage counts: `{json.dumps(payload['execution_census']['formal_run_stage_counts'], sort_keys=True)}`",
        f"- Result-eligible completed runs: `{payload['execution_census']['completed_result_eligible_runs']}`",
        f"- Completed registered deliverables: `{payload['execution_census']['completed_deliverables']}/19`",
        f"- Bound mechanical failure records: `{payload['execution_census']['mechanical_failure_records']}`",
        f"- D2 completed scientific bundles: `{payload['d2_causal_core_progress']['completed_scientific_bundles']}/{payload['d2_causal_core_progress']['maximum_total_scientific_bundles']}`",
        f"- D2 gate-stopped conditional branches: `{payload['d2_causal_core_progress']['conditional_selected_branches']['gate_stopped_models']}`",
        "",
        "### Registered result structures",
        "",
        (
            "Counts below are audited from SHA-bound deliverables. They describe "
            "registered result structure, not selected scientific effects; overlapping "
            "containers and flattened rows are not added together."
        ),
        "",
        "| Deliverable | Analysis type | Audited structures | Multiplicity metadata | Source |",
        "|---|---|---|---|---|",
    ]
    for row in payload["registered_result_structure_census"]:
        structures = "; ".join(
            f"{item['path']}={item['unit_count']} ({item['role']})"
            for item in row["structures"]
        )
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["deliverable"],
                    row["analysis_type"],
                    structures,
                    json.dumps(
                        row["multiplicity_metadata"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    f"{row['source']['path']} @ {row['source']['sha256']}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Executive summary",
            "",
            str(narratives["executive_summary"]),
            "",
            "## Primary mechanism diagnosis",
            "",
            str(narratives["primary_mechanism_diagnosis"]),
            "",
            "## Is the intermediate signal erased?",
            "",
            str(narratives["signal_attenuation_answer"]),
            "",
            "## Cross-model boundary",
            "",
            str(narratives["cross_model_boundary"]),
            "",
            "## Registered Transformer component probe coverage",
            "",
            "| Component | Q0 registered | Q1 registered | Q2 registered | Q3 registered | Causal-support-capable deliverables | Coverage boundary |",
            "|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["component_probe_coverage"]:
        model_coverage = row["model_registered_deliverables"]
        model_causal = row["model_causal_support_deliverables"]
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["component_id"],
                    _coverage_cell(
                        model_coverage[MODEL_IDS[0]], model_causal[MODEL_IDS[0]]
                    ),
                    _coverage_cell(
                        model_coverage[MODEL_IDS[1]], model_causal[MODEL_IDS[1]]
                    ),
                    _coverage_cell(
                        model_coverage[MODEL_IDS[2]], model_causal[MODEL_IDS[2]]
                    ),
                    _coverage_cell(
                        model_coverage[MODEL_IDS[3]], model_causal[MODEL_IDS[3]]
                    ),
                    ", ".join(row["causal_support_deliverables"]),
                    row["coverage_boundary"],
                )
            )
            + " |"
        )
    debt = payload["component_model_coverage_debt"]
    lines.extend(
        [
            "",
            "## Per-model Transformer coverage debt",
            "",
            (
                "Coverage is outcome-independent. Registered or causal-capable "
                "does not mean that a scientific support gate passed."
            ),
            (
                f"18x4 cells: `{debt['component_model_cell_count']}` total; "
                f"`{debt['registered_component_model_cell_count']}` registered; "
                f"`{debt['causal_support_capable_component_model_cell_count']}` "
                "causal-support-capable; "
                f"`{debt['registered_but_descriptive_only_component_model_cell_count']}` "
                "registered descriptive-only; "
                f"`{debt['not_directly_registered_component_model_cell_count']}` "
                "not directly registered."
            ),
            "",
            "Globally registered but descriptive-only components: "
            + ", ".join(
                debt[
                    "globally_registered_but_descriptive_only_components"
                ]
            ),
            "",
            "| Model | Registered components | Causal-support-capable components | Registered but descriptive-only | Not directly registered |",
            "|---|---:|---:|---|---|",
        ]
    )
    for model_id in MODEL_IDS:
        row = debt["per_model"][model_id]
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    model_id,
                    row["registered_component_count"],
                    row["causal_support_capable_component_count"],
                    ", ".join(
                        row["registered_but_descriptive_only_components"]
                    ),
                    ", ".join(row["not_directly_registered_components"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Component result-level support gates",
            "",
            (
                "A causal-support-capable deliverable upgrades a component to "
                "supported only when its registered result passes the exact gate below."
            ),
            "",
            "| Component | Deliverable | Registered support gate |",
            "|---|---|---|",
        ]
    )
    for row in payload["component_result_gate_catalog"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["component_id"],
                    row["deliverable"],
                    row["registered_support_gate"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Registered component practical-equivalence weakening gates",
            "",
            (
                "A failed support gate or p > 0.05 is not equivalence. A component "
                "may use `registered_practical_equivalence` as weakening evidence "
                "only when every cell in one of the complete routes below passes "
                "its preregistered SESOI gate."
            ),
            "",
            "| Component | Deliverable | Complete practical-equivalence gate |",
            "|---|---|---|",
        ]
    )
    for row in payload["component_practical_equivalence_gate_catalog"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["component_id"],
                    row["deliverable"],
                    row["registered_practical_equivalence_gate"],
                )
            )
            + " |"
        )
    equivalence_outcomes = payload[
        "component_practical_equivalence_gate_outcomes"
    ]
    lines.extend(
        [
            "",
            "#### Registered practical-equivalence outcomes",
            "",
            (
                "These rows expose only the Boolean registered gate and its "
                "admitted source identity; no raw effect is copied here."
            ),
            "",
            f"- Routes: `{equivalence_outcomes['route_count']}`",
            f"- Model-specific route outcomes: `{equivalence_outcomes['model_route_count']}`",
            f"- Registered practical equivalence: `{equivalence_outcomes['registered_practical_equivalence_count']}`",
            f"- Practical equivalence not established: `{equivalence_outcomes['registered_practical_equivalence_not_established_count']}`",
            "",
            "| Component | Model | Deliverable | Equivalence outcome | Source SHA256 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in equivalence_outcomes["rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["component_id"],
                    row["method_id"],
                    row["deliverable"],
                    row["gate_outcome"],
                    row["source"]["sha256"],
                )
            )
            + " |"
        )
    outcomes = payload["component_result_gate_outcomes"]
    lines.extend(
        [
            "",
            "### Registered component gate outcomes",
            "",
            (
                "Every preregistered component/model route is evaluated below. "
                "`registered_support_not_established` is a fail-closed gate result; "
                "it is not automatically evidence of an opposite causal effect, "
                "component weakening, or hypothesis rejection."
            ),
            "",
            f"- Routes: `{outcomes['route_count']}`",
            f"- Model-specific route outcomes: `{outcomes['model_route_count']}`",
            f"- Registered support: `{outcomes['registered_support_count']}`",
            f"- Registered support not established: `{outcomes['registered_support_not_established_count']}`",
            "",
            "| Component | Model | Deliverable | Gate outcome | Source SHA256 |",
            "|---|---|---|---|---|",
        ]
    )
    for row in outcomes["rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["component_id"],
                    row["method_id"],
                    row["deliverable"],
                    row["gate_outcome"],
                    row["source"]["sha256"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Hypothesis and opportunity decision gates",
            "",
            "| Hypothesis | Support authorized | Required independent evidence groups | Required supported components | Boundary |",
            "|---|---:|---|---|---|",
        ]
    )
    for row in payload["decision_gate_catalog"]["hypothesis_supported"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["hypothesis_id"],
                    row["support_authorized_in_this_stage"],
                    json.dumps(row["required_independent_evidence_groups"]),
                    ", ".join(row["required_supported_components"]),
                    row["boundary"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "| Opportunity | Required evidence groups | Per-model required groups | Allowed model scope |",
            "|---|---|---|---|",
        ]
    )
    for row in payload["decision_gate_catalog"]["opportunity_primary"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["opportunity_id"],
                    json.dumps(row["required_evidence_groups"]),
                    json.dumps(row["per_model_required_evidence_groups"]),
                    ", ".join(row["allowed_model_scope"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
        "",
        "## Mechanically derived primary attribution evidence",
        "",
        (
            "The rows below use split-sample localization: fold 0 selects the "
            "adjacent post-block transition, fold 1 confirms that transition "
            "and supplies all seven-node inference. Seven-node effects are not "
            "two-fold replications. Descriptive head/group localization is never "
            "used to choose the primary component."
        ),
        str(
            payload["primary_attribution_evidence_census"]["fold_scope"][
                "claim_boundary"
            ]
        ),
        str(
            payload["primary_attribution_evidence_census"][
                "layer_selection_scope"
            ]["claim_boundary"]
        ),
        (
            "The primary component gate is target-margin based. Strict-transfer "
            "NDCG is an independent secondary family and is reported separately; "
            "a target-margin component is not automatically an NDCG transfer-failure "
            "cause. Statistical NDCG corroboration is separated from utility-relevant "
            "corroboration, which requires every same-component NDCG contrast CI to "
            "lie below the frozen -0.005 equivalence boundary."
        ),
        str(
            payload["primary_attribution_evidence_census"][
                "intervention_scope"
            ]["claim_boundary"]
        ),
        str(
            payload["primary_attribution_evidence_census"][
                "transfer_explanation_ladder"
            ]["claim_boundary"]
        ),
        "",
        "| Model | Localization status | Fold-1 reproduced | Node inference fold | Node effect two-fold replicated | Incoming block state | Incoming-state confound | Attention | MLP | Residual node | Norm state | Isolated norm boundary | Norm support without isolation | Residual composition criterion | Residual/norm criterion | Derived component | Primary endpoint | NDCG statistical corroboration | NDCG beyond -0.005 corroboration | Transfer-failure causal scope | Transfer explanation level | Evidence strength | Resolved |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|",
        ]
    )
    for row in payload["primary_attribution_evidence_census"]["rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["localization_status"],
                    row["fold1_transition_reproduced"],
                    row["selected_branch_node_inference_fold"],
                    row["node_effect_two_fold_replication_tested"],
                    row["incoming_block_state_registered_support"],
                    row[
                        "incoming_state_confounds_residual_or_norm_attribution"
                    ],
                    row["attention_branch_registered_support"],
                    row["mlp_branch_registered_support"],
                    row["residual_node_registered_support"],
                    row["normalization_node_registered_support"],
                    row["normalization_boundary_isolated_registered_support"],
                    row[
                        "normalization_state_support_without_boundary_isolation"
                    ],
                    row["residual_composition_criterion_met"],
                    row["residual_norm_interaction_criterion_met"],
                    row["derived_primary_component"],
                    row["primary_attribution_endpoint"],
                    row["strict_transfer_ndcg_component_corroborated"],
                    row[
                        "strict_transfer_ndcg_beyond_equivalence_component_corroborated"
                    ],
                    row["transfer_failure_causal_scope"],
                    row["transfer_explanation_level"],
                    row["derived_evidence_strength"],
                    row["derived_resolution"],
                )
            )
            + " |"
        )
    lines.extend(
        [
        "",
        "## Primary registered candidate bottleneck by model",
        "",
        "| Model | Primary component | Evidence strength | Transfer-failure causal scope | Transfer explanation level | Primary endpoint | NDCG statistical corroboration | NDCG beyond -0.005 corroboration | Fold-1 reproduced | Node inference fold | Node effect two-fold replicated | Adjacent-node role | Erasure boundary established | Attention support | MLP support | Post-block support | Residual composition | Residual/norm interaction | Rationale | Claim boundary | Evidence |",
        "|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for row in payload["primary_loss_attribution"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["primary_component"],
                    row["evidence_strength"],
                    row["transfer_failure_causal_scope"],
                    row["transfer_explanation_level"],
                    row["primary_attribution_endpoint"],
                    row["strict_transfer_ndcg_component_corroborated"],
                    row[
                        "strict_transfer_ndcg_beyond_equivalence_component_corroborated"
                    ],
                    row["fold1_transition_reproduced"],
                    row["selected_branch_node_inference_fold"],
                    row["node_effect_two_fold_replication_tested"],
                    row["within_block_adjacent_change_role"],
                    row["component_erasure_boundary_established"],
                    row["attention_branch_registered_support"],
                    row["mlp_branch_registered_support"],
                    row["postblock_registered_support"],
                    row["residual_composition_criterion_met"],
                    row["residual_norm_interaction_criterion_met"],
                    row["rationale"],
                    row["claim_boundary"],
                    ", ".join(row["evidence_deliverables"]),
                )
            )
            + " |"
        )

    cross_model = payload["cross_model_primary_attribution"]
    lines.extend(
        [
            "",
            "## Cross-model primary attribution boundary",
            "",
            str(cross_model["claim_boundary"]),
            "",
            "| Scope | Q2 component | Q3 component | Shared component | Both resolved | Same component | Beyond-Q2/Q3 generalization authorized | Universal LLM4Rec claim authorized | Rationale |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    cross_model["scope"],
                    cross_model["q2_primary_component"],
                    cross_model["q3_primary_component"],
                    cross_model["shared_primary_component"],
                    cross_model["both_models_resolved"],
                    cross_model["same_component_across_models"],
                    cross_model["generalization_beyond_q2_q3_authorized"],
                    cross_model[
                        "universal_llm4rec_mechanism_claim_authorized"
                    ],
                    cross_model["rationale"],
                )
            )
            + " |",
        ]
    )

    lines.extend(
        [
            "",
            "## Full layer-scan attenuation shape",
            "",
            str(payload["layerwise_attenuation_profile"]["interpretation_boundary"]),
            "",
            "| Model | Endpoint | Shape | Significant attenuation steps | Significant amplification steps | Distributed attenuation established | Exact layer is architecture evidence | Layer scan alone authorizes design | Registered follow-up | Design implication scope |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in payload["layerwise_attenuation_profile"]["shape_summary"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["shape"],
                    row["significant_attenuation_steps"],
                    row["significant_amplification_steps"],
                    row["distributed_attenuation_pattern_established"],
                    row["exact_layer_index_is_architecture_evidence"],
                    row["layer_scan_alone_authorizes_design"],
                    row["registered_followup"],
                    row["design_implication_scope"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### All-layer post-block sufficiency",
            "",
            "| Model | Endpoint | Block | Mean full-state sufficiency vs null | 95% CI | BH q | Directional description |",
            "|---|---|---:|---:|---|---:|---|",
        ]
    )
    for row in payload["layerwise_attenuation_profile"]["all_layer_rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["block_zero_based"],
                    row["mean"],
                    row["ci95"],
                    row["bh_q"],
                    row["directional_description"],
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "### Adjacent-layer changes",
            "",
            "| Model | Endpoint | Transition | Mean adjacent change | 95% CI | BH q | Directional description |",
            "|---|---|---|---:|---|---:|---|",
        ]
    )
    for row in payload["layerwise_attenuation_profile"]["adjacent_layer_rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["transition"],
                    row["mean"],
                    row["ci95"],
                    row["bh_q"],
                    row["directional_description"],
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Adjacent-node attenuation transition profile",
            "",
            str(payload["attenuation_transition_profile"]["interpretation_boundary"]),
            "",
            "| Model | Endpoint | Transition | Mean adjacent effect change | 95% CI | BH q | Directional description | Evidence role |",
            "|---|---|---|---:|---|---:|---|---|",
        ]
    )
    for row in payload["attenuation_transition_profile"]["rows"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["method_id"],
                    row["endpoint"],
                    row["transition"],
                    row["mean"],
                    row["ci95"],
                    row["bh_q"],
                    row["directional_description"],
                    row["evidence_role"],
                )
            )
            + " |"
        )

    lines.extend(
        [
        "",
        "## Negative evidence",
        "",
        str(narratives["negative_evidence_summary"]),
        "",
        "## Transformer component evidence matrix",
        "",
        "| Component | Model scope | Status | Negative-evidence basis | Finding | Claim boundary | Optimization implication | Evidence | Mechanical failures |",
        "|---|---|---|---|---|---|---|---|---|",
        ]
    )
    for row in payload["component_evidence_matrix"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["component_id"],
                    ", ".join(row["model_scope"]),
                    row["status"],
                    row["negative_evidence_basis"],
                    row["finding"],
                    row["claim_boundary"],
                    row["optimization_implication"],
                    ", ".join(row["evidence_deliverables"]),
                    ", ".join(row["mechanical_failure_records"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## H0–H5 evidence matrix",
            "",
            "| Hypothesis | Status | Negative-evidence basis | Rationale | Remaining uncertainty | Evidence |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in payload["hypothesis_status_matrix"]:
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["hypothesis_id"],
                    row["status"],
                    row["negative_evidence_basis"],
                    row["rationale"],
                    row["remaining_uncertainty"],
                    ", ".join(row["evidence_deliverables"]),
                )
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Architecture opportunity ranking",
            "",
            "| Rank | Opportunity | Model scope | Status | Rationale | Falsification gate | Evidence |",
            "|---:|---|---|---|---|---|---|",
        ]
    )
    for row in sorted(payload["architecture_opportunity_ranking"], key=lambda item: item["rank"]):
        lines.append(
            "| "
            + " | ".join(
                _cell(value)
                for value in (
                    row["rank"],
                    row["opportunity_id"],
                    ", ".join(row["model_scope"]),
                    row["status"],
                    row["rationale"],
                    row["falsification_gate"],
                    ", ".join(row["evidence_deliverables"]),
                )
            )
            + " |"
        )

    lines.extend(["", "## Architecture opportunity design cards", ""])
    for row in sorted(
        payload["architecture_opportunity_ranking"], key=lambda item: item["rank"]
    ):
        lines.extend(
            [
                f"### {row['rank']}. {row['opportunity_id']}",
                "",
                f"- Status: `{row['status']}`",
                f"- Model scope: {', '.join(row['model_scope'])}",
                f"- Innovation claim: {row['innovation_claim']}",
                f"- Required modules: {', '.join(row['required_modules'])}",
                f"- Training signal: {row['training_signal']}",
                f"- Training-data requirements: {row['training_data_requirements']}",
                f"- Exact no-op/null recovery invariant: {row['exact_null_recovery_invariant']}",
                f"- Falsification gate: {row['falsification_gate']}",
                f"- Critical ablations: {', '.join(row['critical_ablations'])}",
                f"- Evidence: {', '.join(row['evidence_deliverables'])}",
                f"- Stage boundary: `{row['stage_boundary']}`",
                "- Prior-work differences:",
                "",
            ]
        )
        for comparator, difference in row["prior_work_differences"].items():
            lines.append(f"  - {comparator}: {difference}")
        lines.append("")

    admission = payload["evidence_admission"]
    lines.extend(
        [
            "",
            "## Remaining uncertainty",
            "",
            str(narratives["remaining_uncertainty"]),
            "",
            "## Recommended next action",
            "",
            str(narratives["recommended_next_action"]),
            "",
            "## Evidence admission and boundaries",
            "",
            f"- Admitted deliverables: `{len(admission['deliverables'])}/19`",
            f"- Integrity-checked formal runs: `{admission['formal_completed_integrity_checked']}`",
            f"- Bound mechanical failures retained: `{len(admission['mechanical_failure_records'])}`",
            f"- Source-test content read by closeout audit: `{admission['source_test_content_read_by_closeout_audit']}`",
            f"- Qrels content read by closeout audit: `{admission['qrels_content_read_by_closeout_audit']}`",
            "",
            "### Mechanical failure record ledger",
            "",
        ]
    )
    failure_records = admission["mechanical_failure_records"]
    if failure_records:
        lines.extend(
            [
                "| Run | Status | Record path | SHA256 |",
                "|---|---|---|---|",
            ]
        )
        for record in failure_records:
            lines.append(
                "| "
                + " | ".join(
                    _cell(value)
                    for value in (
                        record["run_id"],
                        record["status"],
                        record["path"],
                        record["sha256"],
                    )
                )
                + " |"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "Boundary assertions:",
            "",
        ]
    )
    for key in REQUIRED_ASSERTIONS:
        lines.append(f"- `{key}`: `{payload['boundary_assertions'][key]}`")
    lines.extend(["", "Admitted deliverable identities:", ""])
    for name, identity in sorted(admission["deliverables"].items()):
        lines.append(
            f"- `{name}`: `{identity['sha256']}` — `{identity['path']}`"
        )
    lines.append("")
    return "\n".join(lines)


def _layerwise_attenuation_profile(
    root: Path, closeout: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose every registered D2 layer cell and a deterministic shape taxonomy."""

    identity = closeout["deliverables"]["d2_postblock"]
    path = root / str(identity["path"])
    if sha256_file(path) != identity["sha256"]:
        raise ValueError("D2 post-block synthesis identity changed")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(metrics, dict)
        or metrics.get("status") != "completed"
        or metrics.get("analysis_type")
        != "transformer_deep_dive_d2_postblock_synthesis"
        or metrics.get("endpoints") != ["target_margin", "ndcg@10"]
    ):
        raise ValueError("D2 post-block synthesis schema differs")
    all_layer = metrics.get("all_layer")
    adjacent = metrics.get("adjacent_transition")
    if (
        not isinstance(all_layer, Mapping)
        or not isinstance(adjacent, Mapping)
        or set(all_layer) != {"target_margin", "ndcg@10"}
        or set(adjacent) != {"target_margin", "ndcg@10"}
    ):
        raise ValueError("D2 layer profile endpoint coverage differs")

    all_rows = _normalize_layer_profile_rows(all_layer, adjacent=False)
    adjacent_rows = _normalize_layer_profile_rows(adjacent, adjacent=True)
    shape_summary = []
    for method_id in MODEL_IDS[2:]:
        for endpoint in ("target_margin", "ndcg@10"):
            cells = [
                row
                for row in adjacent_rows
                if row["method_id"] == method_id and row["endpoint"] == endpoint
            ]
            if len(cells) != len(POSTBLOCK_BLOCKS) - 1:
                raise ValueError("D2 adjacent-layer model/endpoint coverage differs")
            missing = [row["missing"] for row in cells]
            if any(missing) and not all(missing):
                raise ValueError("D2 adjacent-layer coverage is partially missing")
            attenuation = sum(
                row["directional_description"] == "significant_attenuation"
                for row in cells
            )
            amplification = sum(
                row["directional_description"] == "significant_amplification"
                for row in cells
            )
            if all(missing):
                shape = "gate_stopped_or_missing"
            elif attenuation and amplification:
                shape = "mixed_attenuation_and_amplification"
            elif attenuation >= 2:
                shape = "distributed_multi_step_attenuation"
            elif attenuation == 1:
                shape = "localized_single_attenuation_step"
            elif amplification:
                shape = "amplification_only_no_attenuation"
            else:
                shape = "no_registered_significant_adjacent_change"
            distributed = attenuation >= 2
            registered_followup = {
                "gate_stopped_or_missing": "stop_no_layer_based_inference",
                "no_registered_significant_adjacent_change": (
                    "do_not_force_layer_local_cause_use_fixed_breadth_and_readout_checks"
                ),
                "localized_single_attenuation_step": (
                    "use_split_sample_selected_transition_for_seven_node_decomposition"
                ),
                "distributed_multi_step_attenuation": (
                    "retain_all_steps_and_test_cross_layer_propagation_not_exact_layer"
                ),
                "mixed_attenuation_and_amplification": (
                    "combine_selected_transition_decomposition_with_cross_layer_shape"
                ),
                "amplification_only_no_attenuation": (
                    "do_not_infer_an_attenuation_bottleneck"
                ),
            }[shape]
            shape_summary.append(
                {
                    "method_id": method_id,
                    "endpoint": endpoint,
                    "shape": shape,
                    "significant_attenuation_steps": attenuation,
                    "significant_amplification_steps": amplification,
                    "distributed_attenuation_pattern_established": distributed,
                    "exact_layer_index_is_architecture_evidence": False,
                    "layer_scan_alone_authorizes_design": False,
                    "registered_followup": registered_followup,
                    "design_implication_scope": (
                        "distributed_pattern_candidate"
                        if distributed
                        else "component_evidence_required"
                    ),
                }
            )
    return {
        "source": {"path": str(identity["path"]), "sha256": identity["sha256"]},
        "all_layer_row_count": len(all_rows),
        "adjacent_layer_row_count": len(adjacent_rows),
        "shape_summary": shape_summary,
        "shape_taxonomy": [
            "gate_stopped_or_missing",
            "no_registered_significant_adjacent_change",
            "localized_single_attenuation_step",
            "distributed_multi_step_attenuation",
            "mixed_attenuation_and_amplification",
            "amplification_only_no_attenuation",
        ],
        "interpretation_boundary": (
            "All 15 registered post-block sufficiency cells and all 14 adjacent changes "
            "are retained per Q2/Q3 endpoint. A distributed pattern requires at least two "
            "BH-significant negative adjacent changes; a single exact block index remains "
            "localization metadata, is not architecture evidence, and cannot replace the "
            "seven-node component decomposition."
        ),
        "all_layer_rows": all_rows,
        "adjacent_layer_rows": adjacent_rows,
    }


def _normalize_layer_profile_rows(
    source: Mapping[str, Any], *, adjacent: bool
) -> list[dict[str, Any]]:
    expected_blocks = tuple(POSTBLOCK_BLOCKS[1:] if adjacent else POSTBLOCK_BLOCKS)
    expected = {
        (method_id, endpoint, block)
        for method_id in MODEL_IDS[2:]
        for endpoint in ("target_margin", "ndcg@10")
        for block in expected_blocks
    }
    observed = {}
    for endpoint in ("target_margin", "ndcg@10"):
        rows = source.get(endpoint)
        if not isinstance(rows, list):
            raise ValueError("D2 layer profile rows are missing")
        for row in rows:
            if not isinstance(row, Mapping):
                raise ValueError("D2 layer profile row is not an object")
            key = (
                str(row.get("method_id")),
                endpoint,
                int(row.get("block_zero_based", -1)),
            )
            if key in observed:
                raise ValueError("D2 layer profile has duplicate cells")
            observed[key] = row
    if set(observed) != expected:
        raise ValueError("D2 layer profile registered coverage differs")

    normalized = []
    for key in sorted(expected):
        source_row = observed[key]
        missing = source_row.get("missing_cell")
        q_value = source_row.get("bh_q")
        if (
            type(missing) is not bool
            or not isinstance(q_value, (int, float))
            or isinstance(q_value, bool)
            or not math.isfinite(float(q_value))
            or not 0.0 <= float(q_value) <= 1.0
        ):
            raise ValueError("D2 layer profile gate metadata differs")
        mean = source_row.get("mean")
        ci95 = source_row.get("ci95")
        if missing:
            if mean is not None or ci95 != [None, None]:
                raise ValueError("D2 missing layer cell exposes an effect")
            direction = "gate_stopped_or_missing"
        else:
            if (
                not isinstance(mean, (int, float))
                or isinstance(mean, bool)
                or not math.isfinite(float(mean))
                or not isinstance(ci95, list)
                or len(ci95) != 2
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in ci95
                )
            ):
                raise ValueError("D2 layer profile effect schema differs")
            significant = float(q_value) < 0.05
            if significant and float(mean) < 0.0:
                direction = (
                    "significant_attenuation"
                    if adjacent
                    else "significant_negative_sufficiency"
                )
            elif significant and float(mean) > 0.0:
                direction = (
                    "significant_amplification"
                    if adjacent
                    else "significant_positive_sufficiency"
                )
            else:
                direction = "unresolved_no_directional_claim"
        row = {
            "method_id": key[0],
            "endpoint": key[1],
            "block_zero_based": key[2],
            "mean": mean,
            "ci95": ci95,
            "bh_q": float(q_value),
            "bh_significant": float(q_value) < 0.05,
            "missing": missing,
            "directional_description": direction,
            "used_as_primary_component_attribution": False,
            "exact_layer_index_is_architecture_evidence": False,
        }
        if adjacent:
            row["transition"] = f"block_{key[2] - 1} -> block_{key[2]}"
        normalized.append(row)
    return normalized


def _component_probe_coverage() -> list[dict[str, Any]]:
    rows = []
    for component_id in COMPONENT_IDS:
        registered = sorted(COMPONENT_ALLOWED_DELIVERABLES[component_id])
        causal = sorted(
            COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component_id]
        )
        model_registered = {
            model_id: [
                deliverable
                for deliverable in registered
                if model_id
                in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
            ]
            for model_id in MODEL_IDS
        }
        model_causal = {
            model_id: [
                deliverable
                for deliverable in causal
                if model_id
                in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
            ]
            for model_id in MODEL_IDS
        }
        rows.append(
            {
                "component_id": component_id,
                "registered_deliverables": registered,
                "model_registered_deliverables": model_registered,
                "model_causal_support_deliverables": model_causal,
                "model_coverage_levels": {
                    model_id: (
                        "causal-support-capable"
                        if model_causal[model_id]
                        else "descriptive-only"
                        if model_registered[model_id]
                        else "not-directly-registered"
                    )
                    for model_id in MODEL_IDS
                },
                "causal_support_deliverables": causal,
                "causal_support_mechanical_dependencies": {
                    deliverable: sorted(dependencies)
                    for deliverable, dependencies in sorted(
                        COMPONENT_SUPPORT_MECHANICAL_DEPENDENCIES.get(
                            component_id, {}
                        ).items()
                    )
                },
                "coverage_boundary": COMPONENT_PROBE_CLAIM_BOUNDARIES[
                    component_id
                ],
            }
        )
    return rows


def _component_result_gate_catalog() -> list[dict[str, str]]:
    return [
        {
            "component_id": component_id,
            "deliverable": deliverable,
            "registered_support_gate": (
                RESULT_LEVEL_SUPPORTED_COMPONENT_GATE_DESCRIPTIONS[
                    (component_id, deliverable)
                ]
            ),
        }
        for component_id, deliverable in sorted(
            RESULT_LEVEL_SUPPORTED_COMPONENT_GATE_DESCRIPTIONS
        )
    ]


def _component_practical_equivalence_gate_catalog() -> list[dict[str, str]]:
    return [
        {
            "component_id": component_id,
            "deliverable": deliverable,
            "registered_practical_equivalence_gate": (
                RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS[
                    (component_id, deliverable)
                ]
            ),
        }
        for component_id, deliverable in sorted(
            RESULT_LEVEL_EQUIVALENCE_COMPONENT_GATE_DESCRIPTIONS
        )
    ]


def _component_practical_equivalence_gate_outcomes(
    root: Path,
    closeout: Mapping[str, Any],
    *,
    routes: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluate all registered SESOI routes without copying effect values."""

    active_routes = set(
        RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES
        if routes is None
        else routes
    )
    if not active_routes.issubset(RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES):
        raise ValueError("component equivalence outcome route is not registered")
    deliverables = closeout.get("deliverables")
    if not isinstance(deliverables, Mapping):
        raise ValueError("component equivalence outcomes lack closeout deliverables")
    deliverable_cache: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, dict[str, str]] = {}
    rows = []
    for component_id, deliverable in sorted(active_routes):
        if deliverable not in deliverable_cache:
            identity = deliverables.get(deliverable)
            if (
                not isinstance(identity, Mapping)
                or identity.get("status") != "completed"
            ):
                raise ValueError(
                    "component equivalence source is not admitted: "
                    f"{deliverable}"
                )
            relative_path = str(identity.get("path") or "")
            source_path = root / relative_path
            observed_sha256 = sha256_file(source_path)
            if observed_sha256 != identity.get("sha256"):
                raise ValueError(
                    "component equivalence source identity changed: "
                    f"{deliverable}"
                )
            try:
                metrics = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot read component equivalence source: {deliverable}"
                ) from exc
            expected_analysis_type = RESULT_LEVEL_ANALYSIS_TYPES[deliverable]
            if (
                not isinstance(metrics, dict)
                or metrics.get("status") != "completed"
                or metrics.get("analysis_type") != expected_analysis_type
            ):
                raise ValueError(
                    "component equivalence source schema differs: "
                    f"{deliverable}"
                )
            deliverable_cache[deliverable] = metrics
            source_cache[deliverable] = {
                "path": relative_path,
                "sha256": observed_sha256,
            }
        models = COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
        for method_id in sorted(models):
            equivalent = component_result_practical_equivalence(
                component_id,
                method_id,
                deliverable,
                deliverable_cache[deliverable],
            )
            rows.append(
                {
                    "component_id": component_id,
                    "method_id": method_id,
                    "deliverable": deliverable,
                    "gate_outcome": (
                        "registered_practical_equivalence"
                        if equivalent
                        else "registered_practical_equivalence_not_established"
                    ),
                    "registered_practical_equivalence": equivalent,
                    "source": source_cache[deliverable],
                }
            )
    equivalent_count = sum(
        row["registered_practical_equivalence"] is True for row in rows
    )
    return {
        "route_count": len(active_routes),
        "model_route_count": len(rows),
        "registered_practical_equivalence_count": equivalent_count,
        "registered_practical_equivalence_not_established_count": (
            len(rows) - equivalent_count
        ),
        "all_registered_routes_evaluated": active_routes
        == set(RESULT_LEVEL_EQUIVALENCE_COMPONENT_ROUTES),
        "raw_effect_values_emitted": False,
        "not_established_is_non_equivalence_evidence": False,
        "rows": rows,
    }


def _component_result_gate_outcomes(
    root: Path,
    closeout: Mapping[str, Any],
    *,
    routes: Iterable[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    """Evaluate every registered component/model gate without emitting effects."""

    active_routes = set(
        RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES if routes is None else routes
    )
    if not active_routes.issubset(RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES):
        raise ValueError("component result gate outcome route is not registered")
    deliverable_cache: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, dict[str, str]] = {}
    rows = []
    deliverables = closeout.get("deliverables")
    if not isinstance(deliverables, Mapping):
        raise ValueError("component result gate outcomes lack closeout deliverables")
    for component_id, deliverable in sorted(active_routes):
        if deliverable not in deliverable_cache:
            identity = deliverables.get(deliverable)
            if (
                not isinstance(identity, Mapping)
                or identity.get("status") != "completed"
            ):
                raise ValueError(
                    f"component result gate source is not admitted: {deliverable}"
                )
            relative_path = str(identity.get("path") or "")
            source_path = root / relative_path
            observed_sha256 = sha256_file(source_path)
            if observed_sha256 != identity.get("sha256"):
                raise ValueError(
                    f"component result gate source identity changed: {deliverable}"
                )
            try:
                metrics = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"cannot read component result gate source: {deliverable}"
                ) from exc
            expected_analysis_type = RESULT_LEVEL_ANALYSIS_TYPES[deliverable]
            if (
                not isinstance(metrics, dict)
                or metrics.get("status") != "completed"
                or metrics.get("analysis_type") != expected_analysis_type
            ):
                raise ValueError(
                    f"component result gate source schema differs: {deliverable}"
                )
            deliverable_cache[deliverable] = metrics
            source_cache[deliverable] = {
                "path": relative_path,
                "sha256": observed_sha256,
            }
        models = COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
        for method_id in sorted(models):
            supported = component_result_support(
                component_id,
                method_id,
                deliverable,
                deliverable_cache[deliverable],
            )
            rows.append(
                {
                    "component_id": component_id,
                    "method_id": method_id,
                    "deliverable": deliverable,
                    "gate_outcome": (
                        "registered_support"
                        if supported
                        else "registered_support_not_established"
                    ),
                    "registered_support": supported,
                    "source": source_cache[deliverable],
                }
            )
    registered_support_count = sum(
        row["registered_support"] is True for row in rows
    )
    return {
        "route_count": len(active_routes),
        "model_route_count": len(rows),
        "registered_support_count": registered_support_count,
        "registered_support_not_established_count": (
            len(rows) - registered_support_count
        ),
        "all_registered_routes_evaluated": active_routes
        == set(RESULT_LEVEL_SUPPORTED_COMPONENT_ROUTES),
        "raw_effect_values_emitted": False,
        "support_not_established_is_opposite_effect_evidence": False,
        "support_not_established_is_weakened_or_rejected_evidence": False,
        "rows": rows,
    }


def _primary_attribution_evidence_census(
    root: Path, closeout: Mapping[str, Any]
) -> dict[str, Any]:
    """Expose the exact D2-derived primary labels used by the validator."""

    sources: dict[str, dict[str, str]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for deliverable, expected_analysis_type in (
        ("d2_postblock", "transformer_deep_dive_d2_postblock_synthesis"),
        (
            "d2_selected_branches",
            "transformer_deep_dive_d2_selected_branch_synthesis",
        ),
    ):
        identity = closeout.get("deliverables", {}).get(deliverable)
        if (
            not isinstance(identity, Mapping)
            or identity.get("status") != "completed"
        ):
            raise ValueError(
                f"primary attribution source is not admitted: {deliverable}"
            )
        relative_path = str(identity.get("path") or "")
        path = root / relative_path
        observed_sha256 = sha256_file(path)
        if observed_sha256 != identity.get("sha256"):
            raise ValueError(
                f"primary attribution source identity changed: {deliverable}"
            )
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"cannot read primary attribution source: {deliverable}"
            ) from exc
        if (
            not isinstance(value, dict)
            or value.get("status") != "completed"
            or value.get("analysis_type") != expected_analysis_type
        ):
            raise ValueError(
                f"primary attribution source schema differs: {deliverable}"
            )
        metrics[deliverable] = value
        sources[deliverable] = {
            "path": relative_path,
            "sha256": observed_sha256,
        }
    rows = derive_primary_attribution_evidence(
        metrics["d2_postblock"], metrics["d2_selected_branches"]
    )
    return {
        "sources": sources,
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
        "cross_model_scope": derive_cross_model_primary_attribution(rows),
        "rows": rows,
        "model_count": len(rows),
        "deterministic_precedence": list(
            PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE
        ),
        "descriptive_head_or_group_used_as_primary_cause": False,
        "raw_effect_values_emitted": False,
    }


def _decision_gate_catalog() -> dict[str, list[dict[str, Any]]]:
    return {
        "hypothesis_supported": [
            {
                "hypothesis_id": hypothesis_id,
                "support_authorized_in_this_stage": bool(groups),
                "required_independent_evidence_groups": [
                    sorted(group) for group in groups
                ],
                "required_supported_components": list(
                    HYPOTHESIS_SUPPORTED_COMPONENT_REQUIREMENTS.get(
                        hypothesis_id, ()
                    )
                ),
                "boundary": (
                    "H5 cannot be supported or rejected without an independent second seed"
                    if hypothesis_id == "H5"
                    else "all listed independent evidence groups are mandatory"
                ),
            }
            for hypothesis_id, groups in (
                HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS.items()
            )
        ],
        "opportunity_primary": [
            {
                "opportunity_id": opportunity_id,
                "required_evidence_groups": [
                    sorted(group)
                    for group in OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS[
                        opportunity_id
                    ]
                ],
                "per_model_required_evidence_groups": [
                    sorted(group)
                    for group in OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS.get(
                        opportunity_id, ()
                    )
                ],
                "allowed_model_scope": sorted(
                    OPPORTUNITY_ALLOWED_MODEL_SCOPE[opportunity_id]
                ),
            }
            for opportunity_id in OPPORTUNITY_IDS
        ],
    }


def _attenuation_transition_profile(
    root: Path, closeout: Mapping[str, Any]
) -> dict[str, Any]:
    identity = closeout["deliverables"]["d2_selected_branches"]
    path = root / str(identity["path"])
    if sha256_file(path) != identity["sha256"]:
        raise ValueError("D2 selected-branch synthesis identity changed")
    metrics = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(metrics, dict)
        or metrics.get("status") != "completed"
        or metrics.get("analysis_type")
        != "transformer_deep_dive_d2_selected_branch_synthesis"
    ):
        raise ValueError("D2 selected-branch synthesis schema differs")
    source_rows = metrics.get("rows")
    if not isinstance(source_rows, list):
        raise ValueError("D2 selected-branch synthesis rows are missing")
    transitions = [
        (left, right, f"adjacent__{left}__to__{right}")
        for left, right in zip(SELECTED_NODES[:-1], SELECTED_NODES[1:])
    ]
    expected = {
        (model_id, endpoint, contrast_id): (left, right)
        for model_id in MODEL_IDS[2:]
        for endpoint in ("target_margin", "ndcg@10")
        for left, right, contrast_id in transitions
    }
    observed = {
        (
            str(row.get("method_id")),
            str(row.get("endpoint")),
            str(row.get("contrast_id")),
        ): row
        for row in source_rows
        if isinstance(row, Mapping) and row.get("group") == "adjacent_node"
    }
    if set(observed) != set(expected) or len(observed) != 24:
        raise ValueError("D2 adjacent-node profile coverage differs")
    rows = []
    for key in sorted(expected):
        source = observed[key]
        missing = source.get("missing")
        mean = source.get("mean")
        ci95 = source.get("ci95")
        q_value = source.get("bh_q")
        significant = source.get("bh_significant")
        evidence_role = source.get("evidence_role")
        if type(missing) is not bool:
            raise ValueError("D2 adjacent-node missing flag differs")
        if (
            not isinstance(q_value, (int, float))
            or isinstance(q_value, bool)
            or not math.isfinite(float(q_value))
            or not 0.0 <= float(q_value) <= 1.0
            or type(significant) is not bool
            or significant is not (float(q_value) < 0.05)
            or source.get("expected_sign") is not None
            or source.get("registered_support") is not False
        ):
            raise ValueError("D2 adjacent-node registration boundary differs")
        if missing:
            if mean is not None or evidence_role != "missing_or_gate_stopped":
                raise ValueError("D2 missing adjacent-node row exposes a mean")
            direction = "gate_stopped_or_missing"
        else:
            if (
                not isinstance(mean, (int, float))
                or isinstance(mean, bool)
                or not math.isfinite(float(mean))
                or not isinstance(ci95, list)
                or len(ci95) != 2
                or not all(
                    isinstance(value, (int, float))
                    and not isinstance(value, bool)
                    and math.isfinite(float(value))
                    for value in ci95
                )
                or evidence_role
                not in {
                    "registered_confirmatory_branch_localization",
                    "exploratory_unresolved_transition_branch_localization",
                }
            ):
                raise ValueError("D2 adjacent-node inference schema differs")
            if significant and float(mean) < 0.0:
                direction = "significant_attenuation"
            elif significant and float(mean) > 0.0:
                direction = "significant_amplification"
            else:
                direction = "unresolved_no_directional_claim"
        left, right = expected[key]
        rows.append(
            {
                "method_id": key[0],
                "endpoint": key[1],
                "contrast_id": key[2],
                "transition": f"{left} -> {right}",
                "mean": mean,
                "ci95": ci95,
                "bh_q": q_value,
                "bh_significant": significant,
                "missing": missing,
                "directional_description": direction,
                "evidence_role": evidence_role,
                "literal_hidden_state_sign_reversal_claimed": False,
                "used_as_primary_component_attribution": False,
            }
        )
    counts = Counter(row["directional_description"] for row in rows)
    return {
        "source": {"path": str(identity["path"]), "sha256": identity["sha256"]},
        "row_count": len(rows),
        "directional_description_counts": dict(sorted(counts.items())),
        "interpretation_boundary": (
            "These are all preregistered adjacent changes in patch sufficiency. "
            "A negative row describes attenuation between two nodes; it is not "
            "a literal hidden-state sign reversal and does not by itself assign "
            "the primary loss component."
        ),
        "rows": rows,
    }


def _registered_result_structure_census(
    root: Path,
    closeout: Mapping[str, Any],
    *,
    specs: Mapping[str, tuple[tuple[str, str, int | None, str], ...]] = (
        DELIVERABLE_RESULT_STRUCTURE_SPECS
    ),
) -> list[dict[str, Any]]:
    """Audit registered result shapes without selecting or summarizing effects."""

    root = root.resolve()
    deliverables = closeout.get("deliverables")
    if not isinstance(deliverables, Mapping) or set(deliverables) != set(specs):
        raise ValueError("result-structure census deliverable coverage differs")
    rows = []
    for deliverable in sorted(specs):
        identity = deliverables[deliverable]
        if (
            not isinstance(identity, Mapping)
            or identity.get("status") != "completed"
            or not isinstance(identity.get("path"), str)
            or not isinstance(identity.get("sha256"), str)
        ):
            raise ValueError(
                f"result-structure census identity differs: {deliverable}"
            )
        relative_path = Path(identity["path"])
        if relative_path.is_absolute():
            raise ValueError(
                f"result-structure census source must be relative: {deliverable}"
            )
        path = (root / relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"result-structure census source escapes root: {deliverable}"
            ) from error
        if not path.is_file() or sha256_file(path) != identity["sha256"]:
            raise ValueError(
                f"result-structure census source identity changed: {deliverable}"
            )
        result = json.loads(path.read_text(encoding="utf-8"))
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "completed"
            or not isinstance(result.get("analysis_type"), str)
            or not result["analysis_type"]
        ):
            raise ValueError(
                f"result-structure census result schema differs: {deliverable}"
            )
        structures = []
        for structure_path, mode, expected_count, role in specs[deliverable]:
            containers = _resolve_result_structure_path(result, structure_path)
            unit_count = _count_result_structure(containers, mode)
            if unit_count <= 0 or (
                expected_count is not None and unit_count != expected_count
            ):
                raise ValueError(
                    "result-structure census unit count differs: "
                    f"{deliverable}.{structure_path}={unit_count}, "
                    f"expected={expected_count}"
                )
            structures.append(
                {
                    "path": structure_path,
                    "mode": mode,
                    "matched_containers": len(containers),
                    "unit_count": unit_count,
                    "expected_unit_count": expected_count,
                    "exact_count_registered": expected_count is not None,
                    "role": role,
                }
            )
        rows.append(
            {
                "deliverable": deliverable,
                "analysis_type": result["analysis_type"],
                "source": {
                    "path": identity["path"],
                    "sha256": identity["sha256"],
                },
                "structures": structures,
                "multiplicity_metadata": _multiplicity_metadata(result),
                "outcome_values_used_for_structure_admission": False,
                "scientific_effect_values_summarized": False,
            }
        )
    return rows


def _resolve_result_structure_path(
    value: Mapping[str, Any], path: str
) -> list[Any]:
    values: list[Any] = [value]
    for token in path.split("."):
        next_values = []
        for current in values:
            if token == "*":
                if isinstance(current, Mapping):
                    next_values.extend(
                        current[key] for key in sorted(current, key=str)
                    )
                elif isinstance(current, list):
                    next_values.extend(current)
                else:
                    raise ValueError(
                        f"result-structure wildcard reached non-container: {path}"
                    )
            else:
                if not isinstance(current, Mapping) or token not in current:
                    raise ValueError(f"result-structure path is missing: {path}")
                next_values.append(current[token])
        values = next_values
        if not values:
            raise ValueError(f"result-structure path is empty: {path}")
    return values


def _count_result_structure(values: list[Any], mode: str) -> int:
    if mode == "list":
        expected_type = list
    elif mode == "mapping":
        expected_type = Mapping
    elif mode == "mapping_or_list":
        expected_type = (Mapping, list)
    else:
        raise ValueError(f"unknown result-structure count mode: {mode}")
    if any(not isinstance(value, expected_type) for value in values):
        raise ValueError(f"result-structure container type differs: {mode}")
    return sum(len(value) for value in values)


def _multiplicity_metadata(result: Mapping[str, Any]) -> dict[str, Any]:
    metadata = {}
    multiple_testing = result.get("multiple_testing")
    if isinstance(multiple_testing, Mapping):
        metadata["multiple_testing"] = dict(multiple_testing)
    family = result.get("family")
    if isinstance(family, Mapping):
        allowed = {
            "name",
            "definition",
            "planned_size",
            "registered_size",
            "observed_size",
            "multiple_testing",
            "alpha",
        }
        metadata["family"] = {
            key: family[key] for key in sorted(allowed & set(family))
        }
    return metadata


def _execution_census(closeout: Mapping[str, Any]) -> dict[str, Any]:
    declarations = [
        row
        for row in closeout.get("run_declarations", [])
        if isinstance(row, Mapping)
    ]
    statuses = Counter(str(row.get("status") or "missing") for row in declarations)
    stages = Counter(
        str(row.get("analysis_stage") or "unclassified") for row in declarations
    )
    methods = Counter(str(row.get("method_id") or "none") for row in declarations)
    formal_stages = Counter(
        str(row.get("analysis_stage") or "unclassified")
        for row in declarations
        if row.get("result_eligible") is True
    )
    deliverables = closeout.get("deliverables", {})
    return {
        "run_declaration_count": len(declarations),
        "run_status_counts": dict(sorted(statuses.items())),
        "run_stage_counts": dict(sorted(stages.items())),
        "run_method_counts": dict(sorted(methods.items())),
        "formal_run_stage_counts": dict(sorted(formal_stages.items())),
        "result_eligible_run_declarations": sum(
            row.get("result_eligible") is True for row in declarations
        ),
        "completed_result_eligible_runs": sum(
            row.get("result_eligible") is True and row.get("status") == "completed"
            for row in declarations
        ),
        "formal_completed_integrity_checked": int(
            closeout.get("formal_completed_integrity_checked", 0)
        ),
        "completed_deliverables": sum(
            isinstance(value, Mapping) and value.get("status") == "completed"
            for value in deliverables.values()
        ),
        "mechanical_failure_records": len(
            closeout.get("mechanical_failure_records", [])
        ),
        "dev_eval_ledger_entries": int(
            closeout.get("dev_eval_ledger", {}).get("entry_count", 0)
        ),
    }


def _resolve_output(root: Path, output: str | Path) -> Path:
    path = Path(output)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>")


def _coverage_cell(registered: list[str], causal: list[str]) -> str:
    if not registered:
        return "not directly registered"
    detail = f"causal-capable: {', '.join(causal)}" if causal else "descriptive-only"
    return f"{', '.join(registered)} ({detail})"


def _atomic_write_pair(
    first_path: Path,
    first_text: str,
    second_path: Path,
    second_text: str,
) -> None:
    temp_paths: list[Path] = []
    try:
        for path, content in (
            (first_path, first_text),
            (second_path, second_text),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
            )
            temp_path = Path(temp_name)
            temp_paths.append(temp_path)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        temp_paths[0].replace(first_path)
        temp_paths[1].replace(second_path)
    finally:
        for path in temp_paths:
            path.unlink(missing_ok=True)
