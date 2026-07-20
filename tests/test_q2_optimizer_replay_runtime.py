from __future__ import annotations

import torch
import pytest

from myrec.mechanism.q2_optimizer_replay_runtime import (
    _combined_gradient_identity,
    _next_linear_lr,
    _vector_relative_error,
)


def test_combined_raw_gradient_identity_is_hand_computed():
    gradients = {
        "pairwise_ranknet": {"x": torch.tensor([2.0, -4.0])},
        "listwise_softmax": {"x": torch.tensor([4.0, 2.0])},
        "combined_half_half": {"x": torch.tensor([3.0, -1.0])},
    }
    result = _combined_gradient_identity(gradients)
    assert result["maximum_absolute_error"] == 0.0
    assert result["relative_l2_error"] == 0.0
    assert result["passed"] is True


def test_q2_next_scheduler_lr_is_hand_computed():
    state = {
        "training_contract": {"total_optimizer_steps": 967},
        "scheduler": {"base_lrs": [1e-5], "last_epoch": 500},
    }
    assert _next_linear_lr(state) == pytest.approx(1e-5 * 466 / 871)


def test_vector_relative_error_is_hand_computed():
    result = _vector_relative_error(
        {"x": torch.tensor([2.0, 0.0])},
        {"x": torch.tensor([1.0, 0.0])},
    )
    assert result["maximum_absolute_error"] == 1.0
    assert result["relative_l2_error"] == 1.0
