from __future__ import annotations

import json
from pathlib import Path

import pytest

import myrec.mechanism.cross_component_descriptive_synthesis as synthesis


def _completed_rows() -> list[dict]:
    return [
        {
            "evidence_id": evidence_id,
            "path": f"inputs/{evidence_id}.json",
            "sha256": f"{index:064x}",
            "analysis_type": f"analysis_{evidence_id}",
            "status": "completed",
            "status_at_registry": "completed",
        }
        for index, evidence_id in enumerate(sorted(synthesis.RETROSPECTIVE_IDS), 1)
    ]


def test_frozen_retrospective_inventory_is_exactly_seventeen() -> None:
    assert len(synthesis.RETROSPECTIVE_IDS) == 17
    assert len(synthesis.PENDING_PREOUTPUT_IDS) == 4
    assert synthesis.RETROSPECTIVE_IDS.isdisjoint(synthesis.PENDING_PREOUTPUT_IDS)


def test_fixed_contrasts_are_hand_computed() -> None:
    representation = {
        "q2": {
            "regions": {
                synthesis.EARLY_REGION: {
                    "history_channel_participation_ratio": 0.4,
                    "history_mean_pairwise_cosine": 0.2,
                    "history_to_candidate_delta_cosine": 0.1,
                },
                synthesis.LATE_REGION: {
                    "history_channel_participation_ratio": 0.1,
                    "history_mean_pairwise_cosine": 0.7,
                    "history_to_candidate_delta_cosine": 0.6,
                    "native_lens_common_history_score_cosine": -0.2,
                    "rmsnorm_residual_to_common_gain_ratio": 1.05,
                },
            },
            "late_fold_stability": {
                "history_to_candidate_delta_cosine": [0.55, 0.65]
            },
        },
        "q3": {
            "regions": {
                synthesis.EARLY_REGION: {
                    "history_channel_participation_ratio": 0.2,
                    "history_mean_pairwise_cosine": 0.3,
                    "history_to_candidate_delta_cosine": 0.05,
                },
                synthesis.LATE_REGION: {
                    "history_channel_participation_ratio": 0.25,
                    "history_mean_pairwise_cosine": 0.5,
                    "history_to_candidate_delta_cosine": 0.15,
                    "native_lens_common_history_score_cosine": 0.10,
                    "rmsnorm_residual_to_common_gain_ratio": 0.9,
                },
            },
            "late_fold_stability": {
                "history_to_candidate_delta_cosine": [0.14, 0.16]
            },
        },
    }
    result = synthesis._fixed_contrasts(representation)
    assert result["q2"]["late_minus_early_history_channel_participation_ratio"] == pytest.approx(-0.3)
    assert result["q2"]["late_minus_early_history_pairwise_cosine"] == pytest.approx(0.5)
    assert result["q2"]["late_minus_early_history_to_candidate_delta_cosine"] == pytest.approx(0.5)
    assert result["q2"]["late_candidate_alignment_minus_native_common_score_alignment"] == pytest.approx(0.8)
    assert result["q2"]["late_rmsnorm_residual_common_gain_ratio_distance_from_one"] == pytest.approx(0.05)
    assert result["q2"]["late_fold_ranges"]["history_to_candidate_delta_cosine"] == pytest.approx(0.10)


def test_row_requires_one_all_fold_cell() -> None:
    row = {
        "model_key": "q2",
        "region": synthesis.LATE_REGION,
        "normalized_query_fold": "all",
    }
    assert synthesis._row([row], "q2", synthesis.LATE_REGION) is row
    with pytest.raises(ValueError, match="expected one"):
        synthesis._row([row, dict(row)], "q2", synthesis.LATE_REGION)


def test_attention_mass_contribution_audit_is_hand_computed() -> None:
    cells = []
    for model in ("q2", "q3"):
        for block in (13, 20, 27):
            axes = {}
            for axis in ("query_head", "gqa_group"):
                correlation = 0.9
                if model == "q3" and block == 13:
                    correlation = 0.5 if axis == "query_head" else 0.7
                axes[axis] = {
                    "mass_contribution_pearson": correlation,
                    "same_top_index": axis == "gqa_group",
                    "top3_index_overlap": [1, 2],
                    "history_attention_mass": {
                        "top_k_share": 0.6,
                        "effective_count_entropy": 4.0,
                    },
                    "history_o_proj_contribution_norm": {
                        "top_k_share": 0.7,
                        "effective_count_entropy": 3.0,
                    },
                }
            cells.append(
                {
                    "method_id": f"{model}_recranker_generalqwen"
                    if model == "q2"
                    else "q3_tallrec_generalqwen",
                    "block_zero_based": block,
                    "axes": axes,
                }
            )
    result = synthesis._attention_mass_contribution_audit({"cells": cells})
    assert result["row_count"] == 12
    assert result["mean_mass_contribution_pearson"] == pytest.approx(0.85)
    assert result["minimum_mass_contribution_pearson"] == pytest.approx(0.5)
    assert result["cells_below_0_8_count"] == 2
    assert {row["axis"] for row in result["cells_below_0_8"]} == {
        "query_head",
        "gqa_group",
    }
    assert result["interpretation_boundary"][
        "attention_mass_or_contribution_norm_establishes_value_causality"
    ] is False


def test_common_relative_and_score_rank_null_is_hand_computed() -> None:
    representation = {
        model: {
            "regions": {
                synthesis.EARLY_REGION: {
                    "candidate_output_common_energy_fraction": 0.75,
                    "native_lens_score_common_energy_fraction": 0.60,
                },
                synthesis.LATE_REGION: {
                    "candidate_output_common_energy_fraction": 0.80,
                    "native_lens_score_common_energy_fraction": 0.25,
                },
            }
        }
        for model in ("q2", "q3")
    }
    objective = {
        "common_shift": 137.0,
        "cells": [
            {
                "objectives": {
                    "pairwise": {
                        "maximum_absolute_common_shift_loss_delta": 2e-14,
                        "maximum_absolute_gradient_sum": 3e-16,
                        "maximum_hessian_times_ones_l2": 4e-16,
                        "maximum_absolute_common_direction_rayleigh": 5e-17,
                    },
                    "listwise": {
                        "maximum_absolute_common_shift_loss_delta": 1e-14,
                        "maximum_absolute_gradient_sum": 2e-16,
                        "maximum_hessian_times_ones_l2": 6e-16,
                        "maximum_absolute_common_direction_rayleigh": 1e-17,
                    },
                }
            }
        ],
    }
    result = synthesis._common_relative_and_score_rank_null(representation, objective)
    q2_late = result["per_model_region"]["q2"][synthesis.LATE_REGION]
    assert q2_late["candidate_relative_energy_fraction"] == pytest.approx(0.2)
    assert q2_late["candidate_common_to_relative_energy_ratio"] == pytest.approx(4.0)
    assert q2_late["native_score_rank_effective_energy_fraction"] == pytest.approx(0.75)
    assert q2_late["native_score_common_to_rank_effective_energy_ratio"] == pytest.approx(1 / 3)
    assert result["objective_instances"] == 2
    assert result["objective_rank_null_numerical_audit"]["maximum_hessian_times_ones_l2"] == pytest.approx(6e-16)


def test_common_mode_anisotropy_audit_is_hand_computed() -> None:
    stages = (
        synthesis.EARLY_REGION,
        "blocks_07_13",
        "blocks_14_20",
        synthesis.LATE_REGION,
    )
    rows = []
    for model_offset, model in enumerate(("q2", "q3")):
        for stage_index, stage in enumerate(stages):
            base = 0.6 + 0.01 * model_offset + 0.02 * stage_index
            rows.append(
                {
                    "model_key": model,
                    "normalized_query_fold": "all",
                    "region": stage,
                    "common_energy_fraction": base + 0.2,
                    "common_global_mean_energy_fraction": base,
                    "common_mean_pairwise_cosine": base - 0.1,
                    "history_global_mean_energy_fraction": base - 0.2,
                    "history_mean_pairwise_cosine": base - 0.3,
                    "history_channel_participation_ratio": base - 0.5,
                    "history_top_1pct_channel_energy_share": base - 0.35,
                    "history_top_10pct_channel_energy_share": base - 0.15,
                    "common_history_channel_energy_cosine": base - 0.4,
                    "common_channel_participation_ratio": base - 0.5,
                    "residual_channel_participation_ratio": base - 0.45,
                    "residual_top_1pct_channel_energy_share": base - 0.42,
                }
            )
    result = synthesis._common_mode_anisotropy_audit(
        {"fixed_region_rows": rows}
    )
    q2 = result["per_model"]["q2"]
    assert q2["late"]["common_global_mean_energy_fraction"] == pytest.approx(0.66)
    assert q2["late_minus_early"]["common_energy_fraction"] == pytest.approx(0.06)
    assert q2["all_stage_common_global_mean_energy_fraction_above_half"] is True
    assert result["cross_model_late"][
        "both_models_common_global_mean_energy_fraction_above_half"
    ] is True
    assert result["cross_model_late"]["history_channel_participation_ratio"][
        "q3"
    ] == pytest.approx(0.17)
    assert result["interpretation_boundary"][
        "hidden_candidate_common_is_exact_rank_null"
    ] is False


def test_training_capacity_crosslink_is_hand_computed() -> None:
    embedding = {
        "q2_role_update_geometry": {
            role: {"occurrence_weighted_relative_update_l2": value}
            for role, value in {
                "query": 0.4,
                "history": 0.2,
                "candidate": 0.1,
                "structural": 0.05,
            }.items()
        },
        "q2_yes_no_readout_direction": {
            "direction_update_relative_to_base": 0.3,
            "yes_update_rms_empirical_cdf": 0.9,
            "no_update_rms_empirical_cdf": 0.8,
        },
        "q3_embedding_readout_update": {
            "base_embedding_readout_frozen": True,
            "trained_parameter_scope": "q/v LoRA only",
        },
    }
    attention = {name: 0.1 for name in ("q_proj", "k_proj", "v_proj", "o_proj")}
    mlp = {name: 0.2 for name in ("mlp_gate_proj", "mlp_up_proj", "mlp_down_proj")}
    norms = {
        name: 0.0
        for name in (
            "input_rmsnorm",
            "post_attention_rmsnorm",
            "q_norm",
            "k_norm",
        )
    }
    q2_update = {
        "family_rows": [
            {"family": name, "update_energy_share": share}
            for name, share in {**attention, **mlp, **norms}.items()
        ],
        "global_summary": {"relative_update_frobenius": 0.01},
        "layer_concentration": {
            "layer_update_rms_coefficient_of_variation": 0.2,
            "max_to_min_layer_update_rms_ratio": 2.0,
        },
    }
    q3_lora = {
        "early_late_contrasts": [
            {
                "state": "frozen_final_checkpoint",
                "projection": projection,
                "late_head_normalized_participation_ratio": value,
            }
            for projection, value in (("q", 0.7), ("v", 0.9))
        ],
        "q2_same_geometry_comparison": [
            {
                "region": synthesis.LATE_REGION,
                "projection": projection,
                "q3_minus_q2": value,
            }
            for projection, value in (("q", -0.1), ("v", 0.0))
        ],
    }
    shares = {
        "overview": {
            "maximum_observed_mean_share_difference_abs": 0.003,
            "maximum_observed_request_mean_total_variation": 0.01,
            "observed_cells_with_any_mean_share_difference_abs_ge_0_05": 0,
        }
    }
    result = synthesis._training_capacity_crosslink(
        embedding, q2_update, q3_lora, shares
    )
    assert result["q2"]["update_energy_share_by_function"] == pytest.approx(
        {"attention_projections": 0.4, "mlp_projections": 0.6, "normalization": 0.0}
    )
    assert result["q2"]["history_to_candidate_role_update_ratio"] == pytest.approx(2.0)
    assert result["q3"]["late_q3_minus_q2_head_participation"]["q"] == pytest.approx(-0.1)
    assert result["cross_model_boundary"][
        "frozen_direct_readout_is_a_shared_constraint"
    ] is False
    assert result["cross_model_boundary"][
        "large_ranknet_listnet_family_reallocation_observed"
    ] is False


def test_candidate_common_relative_trajectory_is_hand_computed() -> None:
    residual_rows = []
    flow_rows = []
    lens_rows = []
    transport_rows = []
    preference_excess_rows = []
    preference_state_rows = []
    probe_transport_rows = []
    for model_offset, model in enumerate(("q2", "q3")):
        for fold_offset, fold in enumerate(("all", "0", "1")):
            shift = model_offset * 0.01 + fold_offset * 0.001
            for state in range(29):
                common = None if state == 0 else 0.8 - state * 0.001 + shift
                residual_rows.append(
                    {
                        "model_key": model,
                        "normalized_query_fold": fold,
                        "hidden_state_index": state,
                        "mean_common_energy_fraction": common,
                        "mean_residual_brand_real_minus_random_projection": (
                            None if state == 0 else 0.03 + shift
                        ),
                        "mean_residual_category_real_minus_random_projection": (
                            None if state == 0 else 0.04 + shift
                        ),
                    }
                )
                lens_rows.append(
                    {
                        "model_key": model,
                        "normalized_query_fold": fold,
                        "hidden_state_index": state,
                        "score_common_energy_fraction": (
                            None if state == 0 else 0.6 - state * 0.001 + shift
                        ),
                        "common_history_score_cosine": (
                            None if state == 0 else 0.1 + shift
                        ),
                        "common_history_same_sign_fraction": (
                            None if state == 0 else 0.55 + shift
                        ),
                        "full_over_null_candidate_relative_score_rms": (
                            None if state == 0 else 1.2 + shift
                        ),
                    }
                )
                transport_rows.append(
                    {
                        "model_key": model,
                        "normalized_query_fold": fold,
                        "hidden_state_index": state,
                        "mean_cosine": None if state == 0 else 0.2 + shift,
                        "mean_absolute_cosine": None if state == 0 else 0.3 + shift,
                        "mean_signed_candidate_projection_scale": (
                            None if state == 0 else 0.4 + shift
                        ),
                        "mean_candidate_over_history_rms": (
                            None if state == 0 else 0.5 + shift
                        ),
                    }
                )
                for task, value in (("brand", 0.05), ("category", 0.06)):
                    preference_excess_rows.append(
                        {
                            "model_key": model,
                            "normalized_query_fold": fold,
                            "hidden_state_index": state,
                            "position": "candidate_readout",
                            "task": task,
                            "real_minus_random_energy_fraction": (
                                None if state == 0 else value + shift
                            ),
                        }
                    )
                    for label_control, isotropic_multiple in (
                        ("real_labels", 0.3),
                        ("random_labels", 0.1),
                    ):
                        preference_state_rows.append(
                            {
                                "model_key": model,
                                "normalized_query_fold": fold,
                                "hidden_state_index": state,
                                "position": "candidate_readout",
                                "task": task,
                                "label_control": label_control,
                                "isotropic_rank_over_hidden_baseline": 0.01,
                                "mean_fraction_over_isotropic_baseline": (
                                    None
                                    if state == 0
                                    else isotropic_multiple + shift
                                ),
                            }
                        )
            for block in range(28):
                output_common = 0.8 - (block + 1) * 0.001 + shift
                flow_rows.append(
                    {
                        "model_key": model,
                        "normalized_query_fold": fold,
                        "block_zero_based": block,
                        "mean_output_common_energy_fraction": output_common,
                        "mean_update_common_energy_fraction": 0.7 + shift,
                        "candidate_relative_input_rms": 1.0,
                        "candidate_relative_output_rms": 1.1,
                        "candidate_relative_update_rms": 0.2,
                        "mean_candidate_relative_energy_change": 0.21,
                        "fraction_requests_candidate_relative_energy_decreased": 0.4,
                        "mean_candidate_relative_input_update_cosine": -0.2,
                        "mean_candidate_relative_update_projection_coefficient": -0.1,
                        "mean_candidate_relative_output_input_rms_ratio": 1.1,
                    }
                )
    stages = (
        synthesis.EARLY_REGION,
        "blocks_07_13",
        "blocks_14_20",
        synthesis.LATE_REGION,
    )
    for model in ("q2", "q3"):
        for position in ("history_summary_end", "candidate_readout"):
            for task in ("brand", "category"):
                for stage_index, stage in enumerate(stages):
                    real = 0.1 + stage_index * 0.1
                    random = 0.05 + stage_index * 0.1
                    probe_transport_rows.append(
                        {
                            "model_key": model,
                            "position": position,
                            "task": task,
                            "region": stage,
                            "real_mean_squared_canonical_cosine": real,
                            "random_mean_squared_canonical_cosine": random,
                            "real_minus_random_mean_squared_canonical_cosine": (
                                real - random
                            ),
                        }
                    )
    result = synthesis._candidate_common_relative_trajectory(
        {"hidden_state_indices": list(range(29)), "state_rows": residual_rows},
        {"block_zero_based_indices": list(range(28)), "block_rows": flow_rows},
        {"hidden_state_indices": list(range(29)), "state_rows": lens_rows},
        {
            "history_to_candidate_delta_transport_rows": transport_rows,
            "real_minus_random_rows": preference_excess_rows,
            "state_rows": preference_state_rows,
            "probe_subspace_fixed_region_rows": probe_transport_rows,
        },
    )
    assert result["row_count"] == 56
    q2_block0 = result["rows"][0]
    assert q2_block0["hidden_output_candidate_relative_energy_fraction"] == pytest.approx(0.201)
    assert q2_block0["block_update_candidate_relative_energy_fraction"] == pytest.approx(0.3)
    assert q2_block0["mean_candidate_relative_energy_change"] == pytest.approx(0.21)
    assert q2_block0["mean_candidate_relative_input_update_cosine"] == pytest.approx(-0.2)
    assert q2_block0["history_to_candidate_delta_cosine"] == pytest.approx(0.2)
    assert q2_block0[
        "candidate_relative_brand_real_minus_random_projection_fraction"
    ] == pytest.approx(0.03)
    assert q2_block0[
        "candidate_readout_category_real_minus_random_energy_fraction"
    ] == pytest.approx(0.06)
    assert q2_block0[
        "candidate_relative_brand_real_minus_random_isotropic_multiple"
    ] == pytest.approx(3.0)
    assert q2_block0[
        "candidate_readout_category_real_minus_random_isotropic_multiple"
    ] == pytest.approx(0.2)
    assert q2_block0[
        "history_candidate_minus_frozen_lens_common_history_cosine"
    ] == pytest.approx(0.1)
    assert q2_block0["frozen_lens_score_relative_energy_fraction"] == pytest.approx(0.401)
    assert q2_block0["frozen_lens_relative_minus_hidden_relative_fraction"] == pytest.approx(0.2)
    assert q2_block0["hidden_candidate_relative_fraction_change_from_input"] is None
    assert q2_block0["fold0_minus_fold1"]["hidden_candidate_relative_energy_fraction"] == pytest.approx(0.001)
    assert result["maximum_flow_vs_residual_common_fraction_identity_error"] == pytest.approx(0.0)
    assert len(result["fixed_stage_rows"]) == 8
    assert result["composition_shift_audit"]["q2"][
        "all_stage_mean_candidate_relative_energy_changes_positive"
    ] is True
    assert result["semantic_alignment_audit"]["q2"][
        "stages_with_positive_history_candidate_but_negative_frozen_score_history_alignment"
    ] == []
    assert result["probe_subspace_transport"][
        "all_real_and_random_late_continuities_increased"
    ] is True
    assert result["probe_subspace_transport"][
        "maximum_absolute_real_minus_random_continuity"
    ] == pytest.approx(0.05)
    assert result["interpretation_boundary"]["may_change_registered_causal_layer_or_node"] is False
    assert result["interpretation_boundary"][
        "absolute_relative_energy_growth_proves_preference_semantics_preserved"
    ] is False
    assert result["interpretation_boundary"][
        "brand_category_probes_exhaust_preference_semantics"
    ] is False
    assert result["interpretation_boundary"][
        "lack_of_positive_probe_excess_proves_semantic_absence"
    ] is False
    assert result["interpretation_boundary"][
        "negative_input_update_cosine_equals_block_output_reversal"
    ] is False


def test_builder_consumes_all_frozen_inputs_and_preserves_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _completed_rows()
    monkeypatch.setattr(
        synthesis,
        "audit_supplemental_evidence_registry",
        lambda root: {
            "failures": [],
            "entries": rows,
            "registry": {"path": "registry.yaml", "sha256": "a" * 64},
            "registry_manifest": {"path": "manifest.yaml", "sha256": "b" * 64},
        },
    )
    monkeypatch.setattr(
        synthesis,
        "_load_yaml",
        lambda path: {
            "entries": [
                {
                    "evidence_id": row["evidence_id"],
                    "path": row["path"],
                }
                for row in rows
            ]
        },
    )
    monkeypatch.setattr(
        synthesis,
        "_load_json",
        lambda path: {"status": "completed", "path": str(path)},
    )
    monkeypatch.setattr(
        synthesis,
        "_representation_candidate_readout",
        lambda payloads: {"q2": {}, "q3": {}},
    )
    monkeypatch.setattr(
        synthesis, "_routing_and_position", lambda payloads: {"complete": True}
    )
    monkeypatch.setattr(
        synthesis,
        "_optimization_and_parameterization",
        lambda payloads: {"complete": True},
    )
    monkeypatch.setattr(
        synthesis,
        "_fixed_contrasts",
        lambda representation: {"q2": {}, "q3": {}},
    )
    monkeypatch.setattr(
        synthesis,
        "_common_relative_and_score_rank_null",
        lambda representation, objective: {"complete": True},
    )
    monkeypatch.setattr(
        synthesis,
        "_candidate_common_relative_trajectory",
        lambda residual, flow, lens, preference: {"complete": True},
    )
    monkeypatch.setattr(
        synthesis,
        "_common_mode_anisotropy_audit",
        lambda anisotropy: {"complete": True},
    )

    output = tmp_path / "derived" / "metrics.json"
    result = synthesis.build_cross_component_descriptive_synthesis(
        tmp_path, output_path=output, command=["synthesize", "--fixed"]
    )
    assert result["input_count"] == 17
    assert result["interpretation_boundary"]["may_change_design_ranking"] is False
    assert result["interpretation_boundary"]["causal_erasure_or_reversal_established"] is False
    assert result["qrels_read_by_this_synthesis"] is False
    assert result["command"] == ["synthesize", "--fixed"]
    assert json.loads(output.read_text(encoding="utf-8"))["input_count"] == 17


def test_builder_fails_if_retrospective_inventory_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _completed_rows()[:-1]
    monkeypatch.setattr(
        synthesis,
        "audit_supplemental_evidence_registry",
        lambda root: {
            "failures": [],
            "entries": rows,
            "registry": {},
            "registry_manifest": {},
        },
    )
    with pytest.raises(ValueError, match="coverage differs"):
        synthesis.build_cross_component_descriptive_synthesis(
            tmp_path, output_path=tmp_path / "metrics.json"
        )
