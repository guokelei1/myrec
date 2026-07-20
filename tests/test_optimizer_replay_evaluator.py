import pytest

from myrec.mechanism.optimizer_replay_evaluator import (
    _audit_step500_binding,
    _implementation_digest,
    _q3_cell,
    _summary,
)


def test_optimizer_replay_summary_is_hand_computed():
    result = _summary([1.0, 3.0])
    assert result["count"] == 2
    assert result["mean"] == 2.0
    assert result["minimum"] == 1.0
    assert result["maximum"] == 3.0


def test_optimizer_evaluator_rechecks_frozen_step500_binding():
    expected = {
        "checkpoint_id": "checkpoint",
        "optimizer_steps": 500,
        "scheduler_last_epoch": 500,
        "scheduler_step_count": 501,
        "current_lr": 5.0e-6,
        "optimizer_parameter_count": 310,
        "parameter_order_digest": "order",
        "trainer_state_sha256": "trainer",
        "progress_sha256": "progress",
        "model_weights_sha256": "weights",
    }
    observed = {
        **{key: expected[key] for key in (
            "checkpoint_id",
            "optimizer_steps",
            "scheduler_last_epoch",
            "scheduler_step_count",
            "current_lr",
            "optimizer_parameter_count",
            "parameter_order_digest",
        )},
        "status": "passed",
        "method_id": "q2_recranker_generalqwen",
        "deep_dive_manifest_sha256": "manifest",
        "all_moments_finite": True,
        "rng_state_complete": True,
        "bf16_scaler_empty": True,
        "observed_hashes": {
            "trainer_state_sha256": "trainer",
            "progress_sha256": "progress",
            "model_weights_sha256": "weights",
        },
    }
    _audit_step500_binding(
        observed,
        expected,
        method_id="q2_recranker_generalqwen",
        manifest_sha256="manifest",
    )
    observed["scheduler_step_count"] = 500
    with pytest.raises(ValueError, match="differs from manifest"):
        _audit_step500_binding(
            observed,
            expected,
            method_id="q2_recranker_generalqwen",
            manifest_sha256="manifest",
        )


def test_optimizer_evaluator_accepts_model_specific_nonempty_digests():
    assert _implementation_digest(
        {"implementation_identity": {"digest": "q2-runtime"}, "run_contract": {"implementation_digest": "q2-runtime"}}
    ) == "q2-runtime"
    assert _implementation_digest(
        {"implementation_identity": {"digest": "q3-runtime"}, "run_contract": {"implementation_digest": "q3-runtime"}}
    ) == "q3-runtime"
    with pytest.raises(ValueError, match="digest is missing"):
        _implementation_digest({})
    with pytest.raises(ValueError, match="differs from run contract"):
        _implementation_digest(
            {"implementation_identity": {"digest": "q2-runtime"}, "run_contract": {"implementation_digest": "drifted"}}
        )


def test_q3_cell_preserves_weight_and_step_delta_norms_separately():
    stages = (
        "raw_gradient",
        "clipped_gradient",
        "adam_preconditioned_direction",
        "moment_delta",
        "weight_decay_delta",
        "total_delta",
    )
    modes = {}
    for mode in ("a_only", "b_only", "joint"):
        modes[mode] = {
            stage: {"norm": 1.0, "family_share": {"f": 1.0}}
            for stage in stages
        }
        modes[mode]["actual_step_vs_algebra_identity"] = {
            "relative_l2_error": 0.0
        }
    path_metrics = (
        "a_only_replay_function_norm",
        "b_only_replay_function_norm",
        "joint_function_norm",
        "joint_a_component_norm",
        "joint_b_component_norm",
        "joint_second_order_interaction_norm",
        "function_recomposition_max_abs_error",
        "step500_effective_weight_norm",
        "post_step501_effective_weight_norm",
        "step501_effective_delta_norm",
    )
    paths = []
    for block in range(28):
        for projection in ("q", "v"):
            row = {"block_zero_based": block, "projection": projection}
            row.update({metric: float(index + 1) for index, metric in enumerate(path_metrics)})
            paths.append(row)
    result = _q3_cell([{"modes": modes, "lora_paths": paths} for _ in range(6)])
    target = result["lora_paths"]["13"]["q"]
    assert target["step500_effective_weight_norm"]["mean"] == 8.0
    assert target["post_step501_effective_weight_norm"]["mean"] == 9.0
    assert target["step501_effective_delta_norm"]["mean"] == 10.0
