"""Retrospective cross-component synthesis of the 17 frozen supplements.

This derived artifact is deliberately descriptive.  It consumes every
retrospectively frozen supplemental output, opens no qrels or score bundle, and
cannot change component support or architecture ranking.  Its purpose is to
make representation -> candidate transport -> native readout mismatches
auditable without hand-copying values into a research note.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from myrec.mechanism.supplemental_evidence_registry import (
    EXPECTED_SUPPLEMENT_IDS,
    REGISTRY_PATH,
    audit_supplemental_evidence_registry,
)
from myrec.utils.hashing import sha256_file


ANALYSIS_TYPE = "transformer_cross_component_retrospective_descriptive_synthesis"
PENDING_PREOUTPUT_IDS = {
    "d4_mlp_feature_formation_extension",
    "d6_native_readout_diagnostics",
    "component_state_reverse_necessity_v2",
    "component_functional_design_gate_synthesis",
}
RETROSPECTIVE_IDS = set(EXPECTED_SUPPLEMENT_IDS) - PENDING_PREOUTPUT_IDS
EARLY_REGION = "blocks_00_06"
LATE_REGION = "blocks_21_27"


def build_cross_component_descriptive_synthesis(
    root: str | Path = ".",
    *,
    output_path: str | Path,
    overwrite: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load all 17 frozen outputs and emit fixed cross-component projections."""

    root_path = Path(root).resolve()
    registry_audit = audit_supplemental_evidence_registry(root_path)
    if registry_audit.get("failures"):
        raise ValueError("supplement registry has audit failures")
    completed_rows = {
        str(row["evidence_id"]): row
        for row in registry_audit["entries"]
        if row.get("status") == "completed"
        and row.get("status_at_registry") == "completed"
    }
    if set(completed_rows) != RETROSPECTIVE_IDS:
        raise ValueError("retrospective supplement coverage differs from frozen 17")
    registry = _load_yaml(root_path / REGISTRY_PATH)
    entry_by_id = {
        str(entry["evidence_id"]): entry for entry in registry["entries"]
    }
    payloads = {
        evidence_id: _load_json(root_path / entry_by_id[evidence_id]["path"])
        for evidence_id in sorted(RETROSPECTIVE_IDS)
    }

    representation = _representation_candidate_readout(payloads)
    routing = _routing_and_position(payloads)
    optimization = _optimization_and_parameterization(payloads)
    result = {
        "schema_version": 1,
        "analysis_type": ANALYSIS_TYPE,
        "status": "completed",
        "role": "retrospective_descriptive_cross_link_not_registered_family",
        "input_identities": [
            {
                "evidence_id": evidence_id,
                "path": completed_rows[evidence_id]["path"],
                "sha256": completed_rows[evidence_id]["sha256"],
                "analysis_type": completed_rows[evidence_id]["analysis_type"],
            }
            for evidence_id in sorted(RETROSPECTIVE_IDS)
        ],
        "input_count": len(payloads),
        "registry": registry_audit["registry"],
        "registry_manifest": registry_audit["registry_manifest"],
        "representation_candidate_readout": representation,
        "candidate_common_relative_and_score_rank_null": _common_relative_and_score_rank_null(
            representation, payloads["d7_objective_common_nullspace"]
        ),
        "candidate_common_relative_depth_trajectory": _candidate_common_relative_trajectory(
            payloads["d1_candidate_residual_geometry"],
            payloads["d1_candidate_block_flow"],
            payloads["d6_frozen_logit_lens"],
            payloads["d1_preference_subspace_geometry"],
        ),
        "common_mode_anisotropy_audit": _common_mode_anisotropy_audit(
            payloads["d1_activation_anisotropy"]
        ),
        "routing_and_position": routing,
        "optimization_and_parameterization": optimization,
        "fixed_cross_component_contrasts": _fixed_contrasts(representation),
        "interpretation_boundary": {
            "descriptive_only": True,
            "retrospective_cross_link": True,
            "confirmatory_family_member": False,
            "may_change_component_support": False,
            "may_change_design_ranking": False,
            "exact_layer_index_is_architecture_evidence": False,
            "causal_erasure_or_reversal_established": False,
            "operator_necessity_established": False,
            "cross_dataset_or_scale_generalization_established": False,
            "requires_pending_bidirectional_component_gates_for_design": True,
        },
        "qrels_read_by_this_synthesis": False,
        "score_bundles_read_by_this_synthesis": False,
        "source_test_opened": False,
        "command": list(command or []),
    }
    output = _resolve(root_path, output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    _atomic_write_json(output, result)
    return result


def _representation_candidate_readout(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    anisotropy = payloads["d1_activation_anisotropy"]
    flow = payloads["d1_candidate_block_flow"]
    residual = payloads["d1_candidate_residual_geometry"]
    subspace = payloads["d1_preference_subspace_geometry"]
    floor = payloads["d1_query_causal_floor"]
    rmsnorm = payloads["d2_rmsnorm_flow"]
    lens = payloads["d6_frozen_logit_lens"]
    result: dict[str, Any] = {}
    for model in ("q2", "q3"):
        regions: dict[str, Any] = {}
        for region in (EARLY_REGION, LATE_REGION):
            a = _row(anisotropy["fixed_region_rows"], model, region)
            f = _row(flow["fixed_region_rows"], model, region)
            r = _row(residual["fixed_region_rows"], model, region)
            s = _row(
                subspace["history_to_candidate_delta_fixed_region_rows"],
                model,
                region,
            )
            q = _row(floor["fixed_region_rows"], model, region)
            n = _row(rmsnorm["fixed_region_rows"], model, region)
            l = _row(lens["fixed_region_rows"], model, region)
            regions[region] = {
                "history_channel_participation_ratio": a[
                    "history_channel_participation_ratio"
                ],
                "history_top_10pct_channel_energy_share": a[
                    "history_top_10pct_channel_energy_share"
                ],
                "history_mean_pairwise_cosine": a[
                    "history_mean_pairwise_cosine"
                ],
                "history_fraction_orthogonal_to_query_floor": q[
                    "history_fraction_orthogonal_to_query_floor"
                ],
                "history_rms_over_query_floor": q["history_rms_over_query_floor"],
                "history_to_candidate_delta_cosine": s["mean_cosine"],
                "history_to_candidate_delta_absolute_cosine": s[
                    "mean_absolute_cosine"
                ],
                "history_to_candidate_signed_projection_scale": s[
                    "mean_signed_candidate_projection_scale"
                ],
                "candidate_over_history_delta_rms": s[
                    "mean_candidate_over_history_rms"
                ],
                "candidate_relative_energy_change": f[
                    "mean_candidate_relative_energy_change"
                ],
                "candidate_relative_energy_decreased_fraction": f[
                    "fraction_requests_candidate_relative_energy_decreased"
                ],
                "candidate_output_common_energy_fraction": f[
                    "mean_output_common_energy_fraction"
                ],
                "candidate_relative_residual_rms": r[
                    "mean_candidate_relative_residual_rms"
                ],
                "candidate_residual_to_common_rms_ratio": r[
                    "mean_residual_to_common_rms_ratio"
                ],
                "rmsnorm_total_delta_gain": n["total_delta_rms_gain"],
                "rmsnorm_common_delta_gain": n["common_delta_rms_gain"],
                "rmsnorm_residual_delta_gain": n["residual_delta_rms_gain"],
                "rmsnorm_residual_to_common_gain_ratio": n[
                    "residual_to_common_gain_ratio"
                ],
                "rmsnorm_total_delta_pre_post_cosine": n[
                    "total_delta_pre_post_cosine"
                ],
                "native_lens_full_over_null_candidate_relative_score_rms": l[
                    "full_over_null_candidate_relative_score_rms"
                ],
                "native_lens_common_history_score_cosine": l[
                    "common_history_score_cosine"
                ],
                "native_lens_common_history_same_sign_fraction": l[
                    "common_history_same_sign_fraction"
                ],
                "native_lens_score_common_energy_fraction": l[
                    "score_common_energy_fraction"
                ],
            }
        result[model] = {
            "regions": regions,
            "late_fold_stability": _late_fold_stability(
                anisotropy, subspace, rmsnorm, lens, model
            ),
        }
    return result


def _late_fold_stability(
    anisotropy: Mapping[str, Any],
    subspace: Mapping[str, Any],
    rmsnorm: Mapping[str, Any],
    lens: Mapping[str, Any],
    model: str,
) -> dict[str, Any]:
    def values(rows: Sequence[Mapping[str, Any]], field: str) -> list[float]:
        selected = [
            float(row[field])
            for row in rows
            if row.get("model_key") == model
            and row.get("region") == LATE_REGION
            and str(row.get("normalized_query_fold")) in {"0", "1", "fold0", "fold1"}
        ]
        if len(selected) != 2:
            raise ValueError(f"expected two late fold rows for {model}/{field}")
        return selected

    return {
        "history_channel_participation_ratio": values(
            anisotropy["fixed_region_rows"], "history_channel_participation_ratio"
        ),
        "history_mean_pairwise_cosine": values(
            anisotropy["fixed_region_rows"], "history_mean_pairwise_cosine"
        ),
        "history_to_candidate_delta_cosine": values(
            subspace["history_to_candidate_delta_fixed_region_rows"], "mean_cosine"
        ),
        "rmsnorm_total_delta_gain": values(
            rmsnorm["fixed_region_rows"], "total_delta_rms_gain"
        ),
        "native_lens_common_history_score_cosine": values(
            lens["fixed_region_rows"], "common_history_score_cosine"
        ),
    }


def _routing_and_position(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    attention = payloads["d3_attention_pattern_synthesis"]
    position = payloads["d3_full_null_position_shift_audit"]
    qk = payloads["d3_qk_stage_geometry_v3"]
    rope = payloads["d5_rope_position_geometry"]
    return {
        "attention_head_and_gqa_stability": attention["stability"],
        "attention_cell_count": len(attention["cells"]),
        "attention_mass_contribution_alignment": _attention_mass_contribution_audit(
            attention
        ),
        "natural_position_invariants": position["invariants"],
        "position_id_policy": position["position_id_policy"],
        "qk_stage_transition_consistency": qk[
            "stage_transition_consistency"
        ],
        "maximum_rope_norm_relative_l2_error": qk[
            "maximum_rope_norm_relative_l2_error"
        ],
        "rope_registered_modes": rope["registered_modes"],
        "rope_registered_block_count": len(rope["registered_rope_blocks_zero_based"]),
        "rope_eligible_requests_per_model": rope["eligible_requests_per_model"],
        "rope_causal_effect_claim": rope["causal_effect_claim"],
    }


def _attention_mass_contribution_audit(
    attention: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare attention mass with o-proj contribution magnitude in fixed cells."""

    cells = attention.get("cells")
    if not isinstance(cells, list) or len(cells) != 6:
        raise ValueError("attention pattern audit requires six fixed model/block cells")
    rows: list[dict[str, Any]] = []
    for cell in cells:
        method_id = str(cell["method_id"])
        if method_id == "q2_recranker_generalqwen":
            model = "q2"
        elif method_id == "q3_tallrec_generalqwen":
            model = "q3"
        else:
            raise ValueError("attention pattern audit model coverage differs")
        block = int(cell["block_zero_based"])
        if block not in {13, 20, 27}:
            raise ValueError("attention pattern audit block coverage differs")
        axes = cell.get("axes")
        if not isinstance(axes, Mapping) or set(axes) != {"query_head", "gqa_group"}:
            raise ValueError("attention pattern audit axis coverage differs")
        for axis in ("query_head", "gqa_group"):
            value = axes[axis]
            mass = value["history_attention_mass"]
            contribution = value["history_o_proj_contribution_norm"]
            rows.append(
                {
                    "model_key": model,
                    "block_zero_based_lineage_only": block,
                    "axis": axis,
                    "mass_contribution_pearson": float(
                        value["mass_contribution_pearson"]
                    ),
                    "same_top_index": value["same_top_index"] is True,
                    "top3_overlap_count": len(value["top3_index_overlap"]),
                    "mass_top3_share": float(mass["top_k_share"]),
                    "contribution_top3_share": float(
                        contribution["top_k_share"]
                    ),
                    "mass_effective_count_entropy": float(
                        mass["effective_count_entropy"]
                    ),
                    "contribution_effective_count_entropy": float(
                        contribution["effective_count_entropy"]
                    ),
                }
            )
    expected = {
        (model, block, axis)
        for model in ("q2", "q3")
        for block in (13, 20, 27)
        for axis in ("query_head", "gqa_group")
    }
    observed = {
        (row["model_key"], row["block_zero_based_lineage_only"], row["axis"])
        for row in rows
    }
    if observed != expected or len(rows) != len(expected):
        raise ValueError("attention mass/contribution fixed-cell coverage differs")
    correlations = [row["mass_contribution_pearson"] for row in rows]
    low_rows = [
        {
            "model_key": row["model_key"],
            "block_zero_based_lineage_only": row[
                "block_zero_based_lineage_only"
            ],
            "axis": row["axis"],
            "mass_contribution_pearson": row["mass_contribution_pearson"],
        }
        for row in rows
        if row["mass_contribution_pearson"] < 0.8
    ]
    return {
        "row_count": len(rows),
        "rows": rows,
        "mean_mass_contribution_pearson": sum(correlations) / len(correlations),
        "minimum_mass_contribution_pearson": min(correlations),
        "maximum_mass_contribution_pearson": max(correlations),
        "cells_below_0_8": low_rows,
        "cells_below_0_8_count": len(low_rows),
        "all_correlations_positive": all(value > 0.0 for value in correlations),
        "interpretation_boundary": {
            "descriptive_only": True,
            "attention_mass_is_universal_contribution_proxy": False,
            "contribution_norm_has_signed_preference_direction": False,
            "attention_mass_or_contribution_norm_establishes_value_causality": False,
            "top_head_or_group_may_be_selected_for_design": False,
            "requires_registered_edge_and_component_interventions": True,
        },
    }


def _optimization_and_parameterization(
    payloads: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    embedding = payloads["d0_embedding_readout_geometry"]
    nullspace = payloads["d7_objective_common_nullspace"]
    shares = payloads["d7_q2_objective_family_shares"]
    q2_update = payloads["d7_q2_parameter_update_geometry"]
    q2_anisotropy = payloads["d7_q2_update_anisotropy"]
    q3_lora = payloads["d7_q3_lora_head_geometry"]
    return {
        "training_capacity_crosslink": _training_capacity_crosslink(
            embedding, q2_update, q3_lora, shares
        ),
        "q2_embedding_role_relative_update_l2": {
            role: metrics["occurrence_weighted_relative_update_l2"]
            for role, metrics in embedding["q2_role_update_geometry"].items()
            if metrics.get("occurrences", 0) > 0
        },
        "q2_vocabulary_top_10pct_update_energy_share": embedding[
            "q2_vocabulary_update"
        ]["top_10pct_row_update_energy_share"],
        "q2_yes_no_direction_update_relative_to_base": embedding[
            "q2_yes_no_readout_direction"
        ]["direction_update_relative_to_base"],
        "q3_base_embedding_readout_frozen": embedding[
            "q3_embedding_readout_update"
        ]["base_embedding_readout_frozen"],
        "q3_trained_parameter_scope": embedding["q3_embedding_readout_update"][
            "trained_parameter_scope"
        ],
        "objective_pairwise_listwise_score_gradient_cosine_means": [
            {
                "surface": row["surface"],
                "control": row["control"],
                "mean": row["pairwise_listwise_score_gradient_cosine"]["mean"],
                "minimum": row["pairwise_listwise_score_gradient_cosine"][
                    "minimum"
                ],
            }
            for row in nullspace["cells"]
        ],
        "objective_common_shift": nullspace["common_shift"],
        "objective_family_share_overview": shares["overview"],
        "q2_global_update": q2_update["global_summary"],
        "q2_transformer_update": q2_update["transformer_summary"],
        "q2_layer_update_concentration": q2_update["layer_concentration"],
        "q2_update_early_late_contrasts": q2_anisotropy[
            "early_late_contrasts"
        ],
        "q3_lora_early_late_contrasts": q3_lora["early_late_contrasts"],
        "q3_vs_q2_same_geometry": q3_lora["q2_same_geometry_comparison"],
    }


def _training_capacity_crosslink(
    embedding: Mapping[str, Any],
    q2_update: Mapping[str, Any],
    q3_lora: Mapping[str, Any],
    objective_shares: Mapping[str, Any],
) -> dict[str, Any]:
    family = {str(row["family"]): row for row in q2_update["family_rows"]}
    expected_families = {
        "input_rmsnorm",
        "post_attention_rmsnorm",
        "q_norm",
        "k_norm",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "mlp_gate_proj",
        "mlp_up_proj",
        "mlp_down_proj",
    }
    if set(family) != expected_families:
        raise ValueError("Q2 parameter-family update coverage differs")
    function_shares = {
        "attention_projections": sum(
            float(family[name]["update_energy_share"])
            for name in ("q_proj", "k_proj", "v_proj", "o_proj")
        ),
        "mlp_projections": sum(
            float(family[name]["update_energy_share"])
            for name in ("mlp_gate_proj", "mlp_up_proj", "mlp_down_proj")
        ),
        "normalization": sum(
            float(family[name]["update_energy_share"])
            for name in (
                "input_rmsnorm",
                "post_attention_rmsnorm",
                "q_norm",
                "k_norm",
            )
        ),
    }
    if abs(sum(function_shares.values()) - 1.0) > 1.0e-9:
        raise ValueError("Q2 grouped update-energy shares do not sum to one")

    role_rows = embedding["q2_role_update_geometry"]
    role_relative_updates = {
        role: float(role_rows[role]["occurrence_weighted_relative_update_l2"])
        for role in ("query", "history", "candidate", "structural")
    }
    q3_scope = embedding["q3_embedding_readout_update"]
    if q3_scope.get("base_embedding_readout_frozen") is not True:
        raise ValueError("Q3 embedding/readout freeze boundary differs")

    q3_final = {
        str(row["projection"]): row
        for row in q3_lora["early_late_contrasts"]
        if row.get("state") == "frozen_final_checkpoint"
    }
    if set(q3_final) != {"q", "v"}:
        raise ValueError("Q3 final q/v head geometry coverage differs")
    q3_vs_q2_late = {
        str(row["projection"]): row
        for row in q3_lora["q2_same_geometry_comparison"]
        if row.get("region") == LATE_REGION
    }
    if set(q3_vs_q2_late) != {"q", "v"}:
        raise ValueError("Q3-vs-Q2 late q/v comparison coverage differs")

    objective_overview = objective_shares["overview"]
    observed_large_reallocation = int(
        objective_overview[
            "observed_cells_with_any_mean_share_difference_abs_ge_0_05"
        ]
    )
    return {
        "q2": {
            "global_relative_update_frobenius": q2_update["global_summary"][
                "relative_update_frobenius"
            ],
            "layer_update_rms_coefficient_of_variation": q2_update[
                "layer_concentration"
            ]["layer_update_rms_coefficient_of_variation"],
            "max_to_min_layer_update_rms_ratio": q2_update["layer_concentration"][
                "max_to_min_layer_update_rms_ratio"
            ],
            "update_energy_share_by_function": function_shares,
            "role_relative_updates": role_relative_updates,
            "history_to_candidate_role_update_ratio": _safe_ratio(
                role_relative_updates["history"], role_relative_updates["candidate"]
            ),
            "yes_no_readout_direction_update_relative_to_base": embedding[
                "q2_yes_no_readout_direction"
            ]["direction_update_relative_to_base"],
            "yes_no_update_rms_empirical_cdf": {
                "yes": embedding["q2_yes_no_readout_direction"][
                    "yes_update_rms_empirical_cdf"
                ],
                "no": embedding["q2_yes_no_readout_direction"][
                    "no_update_rms_empirical_cdf"
                ],
            },
            "direct_embedding_readout_update_available": True,
        },
        "q3": {
            "direct_embedding_readout_update_available": False,
            "trained_parameter_scope": q3_scope["trained_parameter_scope"],
            "late_final_head_participation": {
                projection: float(
                    row["late_head_normalized_participation_ratio"]
                )
                for projection, row in q3_final.items()
            },
            "late_q3_minus_q2_head_participation": {
                projection: float(row["q3_minus_q2"])
                for projection, row in q3_vs_q2_late.items()
            },
        },
        "objective_geometry": {
            "maximum_observed_mean_parameter_family_share_difference_abs": (
                objective_overview["maximum_observed_mean_share_difference_abs"]
            ),
            "maximum_observed_request_mean_total_variation": objective_overview[
                "maximum_observed_request_mean_total_variation"
            ],
            "observed_cells_with_family_reallocation_at_or_above_0_05": (
                observed_large_reallocation
            ),
        },
        "cross_model_boundary": {
            "frozen_direct_readout_is_a_shared_constraint": False,
            "q3_qv_only_adaptation_may_be_model_specific_constraint": True,
            "capacity_restriction_alone_explains_both_models": False,
            "large_ranknet_listnet_family_reallocation_observed": (
                observed_large_reallocation > 0
            ),
            "requires_pending_exact_optimizer_replay": True,
        },
        "interpretation_boundary": (
            "Checkpoint deltas and gradient-family shares describe available and used "
            "capacity, not causal optimization failure. Q3 cannot directly rotate the "
            "frozen readout, whereas Q2 can; therefore readout freezing is not a shared "
            "explanation. Exact gradient-to-update attribution awaits the registered replay."
        ),
    }


def _fixed_contrasts(representation: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for model in ("q2", "q3"):
        early = representation[model]["regions"][EARLY_REGION]
        late = representation[model]["regions"][LATE_REGION]
        result[model] = {
            "late_minus_early_history_channel_participation_ratio": late[
                "history_channel_participation_ratio"
            ]
            - early["history_channel_participation_ratio"],
            "late_minus_early_history_pairwise_cosine": late[
                "history_mean_pairwise_cosine"
            ]
            - early["history_mean_pairwise_cosine"],
            "late_minus_early_history_to_candidate_delta_cosine": late[
                "history_to_candidate_delta_cosine"
            ]
            - early["history_to_candidate_delta_cosine"],
            "late_candidate_alignment_minus_native_common_score_alignment": late[
                "history_to_candidate_delta_cosine"
            ]
            - late["native_lens_common_history_score_cosine"],
            "late_rmsnorm_residual_common_gain_ratio_distance_from_one": abs(
                late["rmsnorm_residual_to_common_gain_ratio"] - 1.0
            ),
            "late_fold_ranges": {
                key: max(values) - min(values)
                for key, values in representation[model][
                    "late_fold_stability"
                ].items()
            },
        }
    return result


def _common_relative_and_score_rank_null(
    representation: Mapping[str, Any],
    objective_nullspace: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate hidden candidate geometry from the exact scalar-score nullspace."""

    per_model: dict[str, Any] = {}
    for model in ("q2", "q3"):
        per_region: dict[str, Any] = {}
        for region in (EARLY_REGION, LATE_REGION):
            row = representation[model]["regions"][region]
            candidate_common = float(row["candidate_output_common_energy_fraction"])
            score_common = float(row["native_lens_score_common_energy_fraction"])
            per_region[region] = {
                "candidate_common_energy_fraction": candidate_common,
                "candidate_relative_energy_fraction": 1.0 - candidate_common,
                "candidate_common_to_relative_energy_ratio": _safe_ratio(
                    candidate_common, 1.0 - candidate_common
                ),
                "native_score_common_energy_fraction": score_common,
                "native_score_rank_effective_energy_fraction": 1.0 - score_common,
                "native_score_common_to_rank_effective_energy_ratio": _safe_ratio(
                    score_common, 1.0 - score_common
                ),
            }
        per_model[model] = per_region

    objective_cells = objective_nullspace.get("cells")
    if not isinstance(objective_cells, list) or not objective_cells:
        raise ValueError("objective nullspace cells missing")
    maxima = {
        "maximum_absolute_common_shift_loss_delta": 0.0,
        "maximum_absolute_gradient_sum": 0.0,
        "maximum_hessian_times_ones_l2": 0.0,
        "maximum_absolute_common_direction_rayleigh": 0.0,
    }
    objective_count = 0
    for cell in objective_cells:
        objectives = cell.get("objectives")
        if not isinstance(objectives, Mapping):
            raise ValueError("objective nullspace cell objectives missing")
        for metrics in objectives.values():
            objective_count += 1
            for key in maxima:
                maxima[key] = max(maxima[key], abs(float(metrics[key])))
    return {
        "per_model_region": per_model,
        "objective_common_shift": objective_nullspace["common_shift"],
        "objective_cells": len(objective_cells),
        "objective_instances": objective_count,
        "objective_rank_null_numerical_audit": maxima,
        "interpretation_boundary": (
            "One minus hidden candidate-common energy is only a descriptive "
            "candidate-relative fraction at the observed interface. Downstream "
            "nonlinear operators can convert hidden candidate-common displacement "
            "into candidate-relative displacement, so hidden common energy is not "
            "an exact rank-null direction. Only a common shift of final scalar "
            "candidate scores is exactly rank-null for the audited losses. Neither "
            "fact proves causal waste or that explicit centering will improve ranking."
        ),
    }


def _common_mode_anisotropy_audit(
    anisotropy: Mapping[str, Any],
) -> dict[str, Any]:
    """Distinguish candidate-common magnitude from request-specific direction.

    This is a fixed four-stage descriptive audit.  It does not label a shared
    direction as irrelevant and does not turn hidden common displacement into
    an exact score-null direction.
    """

    stages = (EARLY_REGION, "blocks_07_13", "blocks_14_20", LATE_REGION)
    fields = (
        "common_energy_fraction",
        "common_global_mean_energy_fraction",
        "common_mean_pairwise_cosine",
        "history_global_mean_energy_fraction",
        "history_mean_pairwise_cosine",
        "history_channel_participation_ratio",
        "history_top_1pct_channel_energy_share",
        "history_top_10pct_channel_energy_share",
        "common_history_channel_energy_cosine",
        "common_channel_participation_ratio",
        "residual_channel_participation_ratio",
        "residual_top_1pct_channel_energy_share",
    )
    per_model: dict[str, Any] = {}
    for model in ("q2", "q3"):
        selected = [
            _row(anisotropy["fixed_region_rows"], model, stage)
            for stage in stages
        ]
        sequences = {
            field: [float(row[field]) for row in selected] for field in fields
        }
        per_model[model] = {
            "fixed_functional_stages": list(stages),
            "sequences": sequences,
            "late_minus_early": {
                field: sequences[field][-1] - sequences[field][0]
                for field in fields
            },
            "late": {field: sequences[field][-1] for field in fields},
            "all_stage_common_global_mean_energy_fraction_above_half": all(
                value > 0.5
                for value in sequences["common_global_mean_energy_fraction"]
            ),
            "all_stage_common_mean_pairwise_cosine_positive": all(
                value > 0.0
                for value in sequences["common_mean_pairwise_cosine"]
            ),
        }

    late_global = {
        model: per_model[model]["late"]["common_global_mean_energy_fraction"]
        for model in ("q2", "q3")
    }
    late_pairwise = {
        model: per_model[model]["late"]["common_mean_pairwise_cosine"]
        for model in ("q2", "q3")
    }
    late_history_participation = {
        model: per_model[model]["late"]["history_channel_participation_ratio"]
        for model in ("q2", "q3")
    }
    late_history_top_one = {
        model: per_model[model]["late"]["history_top_1pct_channel_energy_share"]
        for model in ("q2", "q3")
    }
    return {
        "per_model": per_model,
        "cross_model_late": {
            "common_global_mean_energy_fraction": late_global,
            "common_mean_pairwise_cosine": late_pairwise,
            "history_channel_participation_ratio": late_history_participation,
            "history_top_1pct_channel_energy_share": late_history_top_one,
            "both_models_common_global_mean_energy_fraction_above_half": all(
                value > 0.5 for value in late_global.values()
            ),
            "both_models_common_mean_pairwise_cosine_positive": all(
                value > 0.0 for value in late_pairwise.values()
            ),
        },
        "interpretation_boundary": {
            "descriptive_only": True,
            "high_request_shared_direction_proves_nonpersonalized_signal": False,
            "hidden_candidate_common_is_exact_rank_null": False,
            "generic_history_or_prompt_response_is_a_competing_explanation": True,
            "requires_history_specific_bidirectional_component_gates": True,
            "meaning": (
                "A high global-mean energy fraction or positive cross-request cosine "
                "shows that candidate-common history deltas share a direction across "
                "requests. It does not establish that the direction is useless or "
                "non-personalized, and downstream nonlinearities can still convert it "
                "into candidate-relative score changes."
            ),
        },
    }


def _candidate_common_relative_trajectory(
    residual: Mapping[str, Any],
    flow: Mapping[str, Any],
    lens: Mapping[str, Any],
    preference: Mapping[str, Any],
) -> dict[str, Any]:
    """Join every frozen block output to hidden and frozen-readout geometry."""

    block_indices = flow.get("block_zero_based_indices")
    hidden_indices = residual.get("hidden_state_indices")
    lens_indices = lens.get("hidden_state_indices")
    if block_indices != list(range(28)):
        raise ValueError("candidate flow block coverage must be exactly 0..27")
    if hidden_indices != list(range(29)) or lens_indices != list(range(29)):
        raise ValueError("candidate residual/lens state coverage must be exactly 0..28")

    rows: list[dict[str, Any]] = []
    maximum_flow_residual_difference = 0.0
    for model in ("q2", "q3"):
        for block in block_indices:
            output_state = int(block) + 1
            flow_all = _indexed_row(
                flow["block_rows"], model, "all", "block_zero_based", block
            )
            residual_all = _indexed_row(
                residual["state_rows"],
                model,
                "all",
                "hidden_state_index",
                output_state,
            )
            lens_all = _indexed_row(
                lens["state_rows"],
                model,
                "all",
                "hidden_state_index",
                output_state,
            )
            transport_all = _indexed_row(
                preference["history_to_candidate_delta_transport_rows"],
                model,
                "all",
                "hidden_state_index",
                output_state,
            )
            brand_all = _preference_excess_row(
                preference["real_minus_random_rows"],
                model,
                "all",
                output_state,
                task="brand",
            )
            category_all = _preference_excess_row(
                preference["real_minus_random_rows"],
                model,
                "all",
                output_state,
                task="category",
            )
            brand_real = _preference_state_row(
                preference["state_rows"],
                model,
                "all",
                output_state,
                task="brand",
                label_control="real_labels",
            )
            brand_random = _preference_state_row(
                preference["state_rows"],
                model,
                "all",
                output_state,
                task="brand",
                label_control="random_labels",
            )
            category_real = _preference_state_row(
                preference["state_rows"],
                model,
                "all",
                output_state,
                task="category",
                label_control="real_labels",
            )
            category_random = _preference_state_row(
                preference["state_rows"],
                model,
                "all",
                output_state,
                task="category",
                label_control="random_labels",
            )
            brand_isotropic = _shared_isotropic_baseline(brand_real, brand_random)
            category_isotropic = _shared_isotropic_baseline(
                category_real, category_random
            )
            input_residual = _indexed_row(
                residual["state_rows"],
                model,
                "all",
                "hidden_state_index",
                block,
            )
            hidden_common = float(residual_all["mean_common_energy_fraction"])
            flow_common = float(flow_all["mean_output_common_energy_fraction"])
            maximum_flow_residual_difference = max(
                maximum_flow_residual_difference, abs(hidden_common - flow_common)
            )
            hidden_relative = 1.0 - hidden_common
            update_common = float(flow_all["mean_update_common_energy_fraction"])
            update_relative = 1.0 - update_common
            score_common = float(lens_all["score_common_energy_fraction"])
            score_relative = 1.0 - score_common
            input_common = input_residual.get("mean_common_energy_fraction")
            input_relative = (
                None if input_common is None else 1.0 - float(input_common)
            )

            fold_differences: dict[str, float] = {}
            for fold in ("0", "1"):
                fold_flow = _indexed_row(
                    flow["block_rows"],
                    model,
                    fold,
                    "block_zero_based",
                    block,
                )
                fold_residual = _indexed_row(
                    residual["state_rows"],
                    model,
                    fold,
                    "hidden_state_index",
                    output_state,
                )
                fold_lens = _indexed_row(
                    lens["state_rows"],
                    model,
                    fold,
                    "hidden_state_index",
                    output_state,
                )
                sign = 1.0 if fold == "0" else -1.0
                fold_differences["hidden_candidate_relative_energy_fraction"] = (
                    fold_differences.get(
                        "hidden_candidate_relative_energy_fraction", 0.0
                    )
                    + sign
                    * (1.0 - float(fold_residual["mean_common_energy_fraction"]))
                )
                fold_differences["block_update_candidate_relative_energy_fraction"] = (
                    fold_differences.get(
                        "block_update_candidate_relative_energy_fraction", 0.0
                    )
                    + sign
                    * (1.0 - float(fold_flow["mean_update_common_energy_fraction"]))
                )
                fold_differences["frozen_lens_score_relative_energy_fraction"] = (
                    fold_differences.get(
                        "frozen_lens_score_relative_energy_fraction", 0.0
                    )
                    + sign
                    * (1.0 - float(fold_lens["score_common_energy_fraction"]))
                )

            rows.append(
                {
                    "model_key": model,
                    "block_zero_based_lineage_only": block,
                    "output_hidden_state_index_lineage_only": output_state,
                    "normalized_output_depth": output_state / 28.0,
                    "fixed_functional_stage": _fixed_stage(block),
                    "hidden_output_candidate_common_energy_fraction": hidden_common,
                    "hidden_output_candidate_relative_energy_fraction": hidden_relative,
                    "hidden_candidate_relative_fraction_change_from_input": (
                        None
                        if input_relative is None
                        else hidden_relative - input_relative
                    ),
                    "block_update_common_energy_fraction": update_common,
                    "block_update_candidate_relative_energy_fraction": update_relative,
                    "candidate_relative_input_rms": float(
                        flow_all["candidate_relative_input_rms"]
                    ),
                    "candidate_relative_output_rms": float(
                        flow_all["candidate_relative_output_rms"]
                    ),
                    "candidate_relative_update_rms": float(
                        flow_all["candidate_relative_update_rms"]
                    ),
                    "mean_candidate_relative_energy_change": float(
                        flow_all["mean_candidate_relative_energy_change"]
                    ),
                    "fraction_requests_candidate_relative_energy_decreased": float(
                        flow_all[
                            "fraction_requests_candidate_relative_energy_decreased"
                        ]
                    ),
                    "mean_candidate_relative_input_update_cosine": _optional_float(
                        flow_all["mean_candidate_relative_input_update_cosine"]
                    ),
                    "mean_candidate_relative_update_projection_coefficient": _optional_float(
                        flow_all[
                            "mean_candidate_relative_update_projection_coefficient"
                        ]
                    ),
                    "mean_candidate_relative_output_input_rms_ratio": _optional_float(
                        flow_all["mean_candidate_relative_output_input_rms_ratio"]
                    ),
                    "history_to_candidate_delta_cosine": _optional_float(
                        transport_all["mean_cosine"]
                    ),
                    "history_to_candidate_delta_absolute_cosine": _optional_float(
                        transport_all["mean_absolute_cosine"]
                    ),
                    "history_to_candidate_signed_projection_scale": _optional_float(
                        transport_all["mean_signed_candidate_projection_scale"]
                    ),
                    "candidate_over_history_delta_rms": _optional_float(
                        transport_all["mean_candidate_over_history_rms"]
                    ),
                    "candidate_relative_brand_real_minus_random_projection_fraction": _optional_float(
                        residual_all[
                            "mean_residual_brand_real_minus_random_projection"
                        ]
                    ),
                    "candidate_relative_category_real_minus_random_projection_fraction": _optional_float(
                        residual_all[
                            "mean_residual_category_real_minus_random_projection"
                        ]
                    ),
                    "candidate_relative_brand_real_minus_random_isotropic_multiple": _safe_ratio(
                        float(
                            residual_all[
                                "mean_residual_brand_real_minus_random_projection"
                            ]
                        ),
                        brand_isotropic,
                    ),
                    "candidate_relative_category_real_minus_random_isotropic_multiple": _safe_ratio(
                        float(
                            residual_all[
                                "mean_residual_category_real_minus_random_projection"
                            ]
                        ),
                        category_isotropic,
                    ),
                    "candidate_readout_brand_real_minus_random_energy_fraction": _optional_float(
                        brand_all["real_minus_random_energy_fraction"]
                    ),
                    "candidate_readout_category_real_minus_random_energy_fraction": _optional_float(
                        category_all["real_minus_random_energy_fraction"]
                    ),
                    "candidate_readout_brand_real_minus_random_isotropic_multiple": (
                        float(brand_real["mean_fraction_over_isotropic_baseline"])
                        - float(
                            brand_random["mean_fraction_over_isotropic_baseline"]
                        )
                    ),
                    "candidate_readout_category_real_minus_random_isotropic_multiple": (
                        float(category_real["mean_fraction_over_isotropic_baseline"])
                        - float(
                            category_random[
                                "mean_fraction_over_isotropic_baseline"
                            ]
                        )
                    ),
                    "frozen_lens_score_common_energy_fraction": score_common,
                    "frozen_lens_score_relative_energy_fraction": score_relative,
                    "frozen_lens_common_history_score_cosine": _optional_float(
                        lens_all["common_history_score_cosine"]
                    ),
                    "frozen_lens_common_history_same_sign_fraction": _optional_float(
                        lens_all["common_history_same_sign_fraction"]
                    ),
                    "frozen_lens_full_over_null_candidate_relative_score_rms": _optional_float(
                        lens_all[
                            "full_over_null_candidate_relative_score_rms"
                        ]
                    ),
                    "history_candidate_minus_frozen_lens_common_history_cosine": (
                        None
                        if transport_all["mean_cosine"] is None
                        or lens_all["common_history_score_cosine"] is None
                        else float(transport_all["mean_cosine"])
                        - float(lens_all["common_history_score_cosine"])
                    ),
                    "frozen_lens_relative_minus_hidden_relative_fraction": (
                        score_relative - hidden_relative
                    ),
                    "frozen_lens_relative_to_hidden_relative_fraction_ratio": (
                        _safe_ratio(score_relative, hidden_relative)
                    ),
                    "fold0_minus_fold1": fold_differences,
                }
            )

    stage_rows: list[dict[str, Any]] = []
    stage_metrics = {
        "mean_hidden_output_candidate_relative_energy_fraction": (
            "hidden_output_candidate_relative_energy_fraction"
        ),
        "mean_block_update_candidate_relative_energy_fraction": (
            "block_update_candidate_relative_energy_fraction"
        ),
        "mean_frozen_lens_score_relative_energy_fraction": (
            "frozen_lens_score_relative_energy_fraction"
        ),
        "mean_frozen_lens_relative_minus_hidden_relative_fraction": (
            "frozen_lens_relative_minus_hidden_relative_fraction"
        ),
        "mean_frozen_lens_common_history_score_cosine": (
            "frozen_lens_common_history_score_cosine"
        ),
        "mean_frozen_lens_common_history_same_sign_fraction": (
            "frozen_lens_common_history_same_sign_fraction"
        ),
        "mean_frozen_lens_full_over_null_candidate_relative_score_rms": (
            "frozen_lens_full_over_null_candidate_relative_score_rms"
        ),
        "mean_history_candidate_minus_frozen_lens_common_history_cosine": (
            "history_candidate_minus_frozen_lens_common_history_cosine"
        ),
        "mean_hidden_candidate_relative_fraction_change_from_input": (
            "hidden_candidate_relative_fraction_change_from_input"
        ),
        "mean_candidate_relative_energy_change": (
            "mean_candidate_relative_energy_change"
        ),
        "mean_fraction_requests_candidate_relative_energy_decreased": (
            "fraction_requests_candidate_relative_energy_decreased"
        ),
        "mean_candidate_relative_input_update_cosine": (
            "mean_candidate_relative_input_update_cosine"
        ),
        "mean_candidate_relative_update_projection_coefficient": (
            "mean_candidate_relative_update_projection_coefficient"
        ),
        "mean_candidate_relative_output_input_rms_ratio": (
            "mean_candidate_relative_output_input_rms_ratio"
        ),
        "mean_history_to_candidate_delta_cosine": (
            "history_to_candidate_delta_cosine"
        ),
        "mean_history_to_candidate_delta_absolute_cosine": (
            "history_to_candidate_delta_absolute_cosine"
        ),
        "mean_history_to_candidate_signed_projection_scale": (
            "history_to_candidate_signed_projection_scale"
        ),
        "mean_candidate_over_history_delta_rms": "candidate_over_history_delta_rms",
        "mean_candidate_relative_brand_real_minus_random_projection_fraction": (
            "candidate_relative_brand_real_minus_random_projection_fraction"
        ),
        "mean_candidate_relative_category_real_minus_random_projection_fraction": (
            "candidate_relative_category_real_minus_random_projection_fraction"
        ),
        "mean_candidate_relative_brand_real_minus_random_isotropic_multiple": (
            "candidate_relative_brand_real_minus_random_isotropic_multiple"
        ),
        "mean_candidate_relative_category_real_minus_random_isotropic_multiple": (
            "candidate_relative_category_real_minus_random_isotropic_multiple"
        ),
        "mean_candidate_readout_brand_real_minus_random_energy_fraction": (
            "candidate_readout_brand_real_minus_random_energy_fraction"
        ),
        "mean_candidate_readout_category_real_minus_random_energy_fraction": (
            "candidate_readout_category_real_minus_random_energy_fraction"
        ),
        "mean_candidate_readout_brand_real_minus_random_isotropic_multiple": (
            "candidate_readout_brand_real_minus_random_isotropic_multiple"
        ),
        "mean_candidate_readout_category_real_minus_random_isotropic_multiple": (
            "candidate_readout_category_real_minus_random_isotropic_multiple"
        ),
    }
    for model in ("q2", "q3"):
        for stage in (EARLY_REGION, "blocks_07_13", "blocks_14_20", LATE_REGION):
            selected = [
                row
                for row in rows
                if row["model_key"] == model
                and row["fixed_functional_stage"] == stage
            ]
            if len(selected) != 7:
                raise ValueError(f"expected seven rows for {model}/{stage}")
            stage_rows.append(
                {
                    "model_key": model,
                    "fixed_functional_stage": stage,
                    **{
                        summary_key: _mean_present(
                            [row[row_key] for row in selected]
                        )
                        for summary_key, row_key in stage_metrics.items()
                    },
                    "maximum_absolute_fold_difference": {
                        metric: max(
                            abs(row["fold0_minus_fold1"][metric])
                            for row in selected
                        )
                        for metric in (
                            "hidden_candidate_relative_energy_fraction",
                            "block_update_candidate_relative_energy_fraction",
                            "frozen_lens_score_relative_energy_fraction",
                        )
                    },
                }
            )

    composition_shift_audit: dict[str, Any] = {}
    semantic_alignment_audit: dict[str, Any] = {}
    for model in ("q2", "q3"):
        model_stages = [row for row in stage_rows if row["model_key"] == model]
        composition_shift_audit[model] = {
            "stages_where_relative_fraction_declines_while_mean_energy_increases": [
                row["fixed_functional_stage"]
                for row in model_stages
                if row[
                    "mean_hidden_candidate_relative_fraction_change_from_input"
                ]
                < 0.0
                and row["mean_candidate_relative_energy_change"] > 0.0
            ],
            "all_stage_mean_candidate_relative_energy_changes_positive": all(
                row["mean_candidate_relative_energy_change"] > 0.0
                for row in model_stages
            ),
            "late_mean_fraction_requests_with_candidate_relative_energy_decrease": next(
                row[
                    "mean_fraction_requests_candidate_relative_energy_decreased"
                ]
                for row in model_stages
                if row["fixed_functional_stage"] == LATE_REGION
            ),
        }
        semantic_alignment_audit[model] = {
            "stages_with_positive_history_candidate_but_negative_frozen_score_history_alignment": [
                row["fixed_functional_stage"]
                for row in model_stages
                if row["mean_history_to_candidate_delta_cosine"] > 0.0
                and row["mean_frozen_lens_common_history_score_cosine"] < 0.0
            ],
            "late_history_candidate_minus_frozen_score_history_cosine": next(
                row[
                    "mean_history_candidate_minus_frozen_lens_common_history_cosine"
                ]
                for row in model_stages
                if row["fixed_functional_stage"] == LATE_REGION
            ),
            "candidate_relative_brand_real_minus_random_projection_sequence": [
                row[
                    "mean_candidate_relative_brand_real_minus_random_projection_fraction"
                ]
                for row in model_stages
            ],
            "candidate_relative_category_real_minus_random_projection_sequence": [
                row[
                    "mean_candidate_relative_category_real_minus_random_projection_fraction"
                ]
                for row in model_stages
            ],
            "candidate_relative_brand_real_minus_random_isotropic_multiple_sequence": [
                row[
                    "mean_candidate_relative_brand_real_minus_random_isotropic_multiple"
                ]
                for row in model_stages
            ],
            "candidate_relative_category_real_minus_random_isotropic_multiple_sequence": [
                row[
                    "mean_candidate_relative_category_real_minus_random_isotropic_multiple"
                ]
                for row in model_stages
            ],
        }

    return {
        "row_count": len(rows),
        "rows": rows,
        "fixed_stage_rows": stage_rows,
        "composition_shift_audit": composition_shift_audit,
        "semantic_alignment_audit": semantic_alignment_audit,
        "probe_subspace_transport": _probe_subspace_transport(preference),
        "maximum_flow_vs_residual_common_fraction_identity_error": (
            maximum_flow_residual_difference
        ),
        "interpretation_boundary": {
            "descriptive_only": True,
            "all_28_blocks_reported_without_best_layer_selection": True,
            "absolute_block_indices_are_lineage_only": True,
            "intermediate_frozen_logit_lens_is_non_native": True,
            "relative_readout_enrichment_proves_hidden_conversion": False,
            "absolute_relative_energy_growth_proves_preference_semantics_preserved": False,
            "negative_input_update_cosine_equals_block_output_reversal": False,
            "train_only_probe_alignment_proves_preference_semantics": False,
            "brand_category_probes_exhaust_preference_semantics": False,
            "lack_of_positive_probe_excess_proves_semantic_absence": False,
            "history_candidate_cosine_proves_correct_user_specific_transfer": False,
            "may_change_registered_causal_layer_or_node": False,
            "may_establish_operator_causality_or_design_ranking": False,
            "functional_use": (
                "Interpret, but never retarget, the registered attention/MLP/"
                "residual/readout sufficiency and necessity interfaces."
            ),
        },
    }


def _probe_subspace_transport(preference: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "model_key": row["model_key"],
            "position": row["position"],
            "task": row["task"],
            "fixed_functional_stage": row["region"],
            "real_mean_squared_canonical_cosine": float(
                row["real_mean_squared_canonical_cosine"]
            ),
            "random_mean_squared_canonical_cosine": float(
                row["random_mean_squared_canonical_cosine"]
            ),
            "real_minus_random_mean_squared_canonical_cosine": float(
                row["real_minus_random_mean_squared_canonical_cosine"]
            ),
        }
        for row in preference["probe_subspace_fixed_region_rows"]
        if row.get("position") in {"history_summary_end", "candidate_readout"}
    ]
    if len(rows) != 32:
        raise ValueError("preference probe transport coverage must be 32 fixed rows")
    comparisons = []
    for model in ("q2", "q3"):
        for position in ("history_summary_end", "candidate_readout"):
            for task in ("brand", "category"):
                early = _probe_transport_row(rows, model, position, task, EARLY_REGION)
                late = _probe_transport_row(rows, model, position, task, LATE_REGION)
                comparisons.append(
                    {
                        "model_key": model,
                        "position": position,
                        "task": task,
                        "late_minus_early_real_continuity": (
                            late["real_mean_squared_canonical_cosine"]
                            - early["real_mean_squared_canonical_cosine"]
                        ),
                        "late_minus_early_random_continuity": (
                            late["random_mean_squared_canonical_cosine"]
                            - early["random_mean_squared_canonical_cosine"]
                        ),
                        "both_real_and_random_continuity_increased": (
                            late["real_mean_squared_canonical_cosine"]
                            > early["real_mean_squared_canonical_cosine"]
                            and late["random_mean_squared_canonical_cosine"]
                            > early["random_mean_squared_canonical_cosine"]
                        ),
                        "late_real_minus_random_continuity": late[
                            "real_minus_random_mean_squared_canonical_cosine"
                        ],
                    }
                )
    return {
        "rows": rows,
        "comparisons": comparisons,
        "maximum_absolute_real_minus_random_continuity": max(
            abs(row["real_minus_random_mean_squared_canonical_cosine"])
            for row in rows
        ),
        "all_real_and_random_late_continuities_increased": all(
            row["both_real_and_random_continuity_increased"]
            for row in comparisons
        ),
        "interpretation_boundary": (
            "Adjacent probe-row-space continuity is reported with its random-label "
            "control. Joint real/random growth is generic geometric stabilization, "
            "not evidence that preference semantics are preserved or used."
        ),
    }


def _probe_transport_row(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    position: str,
    task: str,
    stage: str,
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if row["model_key"] == model
        and row["position"] == position
        and row["task"] == task
        and row["fixed_functional_stage"] == stage
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one probe transport row for {model}/{position}/{task}/{stage}"
        )
    return selected[0]


def _indexed_row(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    fold: str,
    index_key: str,
    index: int,
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("model_key") == model
        and _normalized_fold(row.get("normalized_query_fold")) == fold
        and row.get(index_key) == index
    ]
    if len(selected) != 1:
        raise ValueError(
            f"expected one row for {model}/fold={fold}/{index_key}={index}"
        )
    return selected[0]


def _preference_excess_row(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    fold: str,
    hidden_state_index: int,
    *,
    task: str,
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("model_key") == model
        and _normalized_fold(row.get("normalized_query_fold")) == fold
        and row.get("hidden_state_index") == hidden_state_index
        and row.get("position") == "candidate_readout"
        and row.get("task") == task
    ]
    if len(selected) != 1:
        raise ValueError(
            "expected one candidate-readout preference row for "
            f"{model}/fold={fold}/state={hidden_state_index}/task={task}"
        )
    return selected[0]


def _preference_state_row(
    rows: Sequence[Mapping[str, Any]],
    model: str,
    fold: str,
    hidden_state_index: int,
    *,
    task: str,
    label_control: str,
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("model_key") == model
        and _normalized_fold(row.get("normalized_query_fold")) == fold
        and row.get("hidden_state_index") == hidden_state_index
        and row.get("position") == "candidate_readout"
        and row.get("task") == task
        and row.get("label_control") == label_control
    ]
    if len(selected) != 1:
        raise ValueError(
            "expected one candidate-readout preference state row for "
            f"{model}/fold={fold}/state={hidden_state_index}/task={task}/"
            f"control={label_control}"
        )
    return selected[0]


def _shared_isotropic_baseline(
    real: Mapping[str, Any], random: Mapping[str, Any]
) -> float:
    real_value = float(real["isotropic_rank_over_hidden_baseline"])
    random_value = float(random["isotropic_rank_over_hidden_baseline"])
    if abs(real_value - random_value) > 1.0e-15 or real_value <= 0.0:
        raise ValueError("real/random preference isotropic baseline differs")
    return real_value


def _normalized_fold(value: Any) -> str:
    text = str(value)
    if text in {"0", "fold0"}:
        return "0"
    if text in {"1", "fold1"}:
        return "1"
    if text == "all":
        return "all"
    raise ValueError(f"unknown normalized-query fold label: {value}")


def _fixed_stage(block: int) -> str:
    if 0 <= block <= 6:
        return EARLY_REGION
    if 7 <= block <= 13:
        return "blocks_07_13"
    if 14 <= block <= 20:
        return "blocks_14_20"
    if 21 <= block <= 27:
        return LATE_REGION
    raise ValueError(f"block outside fixed stage coverage: {block}")


def _mean_present(values: Sequence[float | None]) -> float:
    present = [float(value) for value in values if value is not None]
    if not present:
        raise ValueError("at least one finite value required")
    return sum(present) / len(present)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator > 0.0 else None


def _row(
    rows: Sequence[Mapping[str, Any]], model: str, region: str
) -> Mapping[str, Any]:
    selected = [
        row
        for row in rows
        if row.get("model_key") == model
        and row.get("region") == region
        and row.get("normalized_query_fold") == "all"
    ]
    if len(selected) != 1:
        raise ValueError(f"expected one all-fold row for {model}/{region}")
    return selected[0]


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "completed":
        raise ValueError(f"completed JSON object required: {path}")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML object required: {path}")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
