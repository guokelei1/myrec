from __future__ import annotations

import torch

from cpdlr.losses import compute_probe_loss


def test_train_only_loss_is_finite_and_backpropagates() -> None:
    final = torch.tensor([[1.2, -0.4, 0.1]], requires_grad=True)
    null = torch.tensor([[0.9, -0.2, 0.0]], requires_grad=True)
    delta = final - null
    outputs = {
        "final": final,
        "null": null,
        "tangent_delta": delta,
    }
    batch = {
        "candidate_mask": torch.tensor([[True, True, True]]),
        "exact_repeat": torch.tensor([[False, False, True]]),
        "labels": torch.tensor([[1.0, 0.0, 0.0]]),
        "teacher_scores": torch.tensor([[0.8, -0.1, 0.0]]),
    }
    config = {
        "loss": {
            "anchor_temperature": 1.0,
            "anchor_weight": 0.5,
            "consistency_weight": 0.2,
            "ranking_weight": 1.0,
            "repeat_margin": 0.05,
            "repeat_weight": 0.1,
            "transfer_margin": 0.1,
            "transfer_weight": 0.2,
        }
    }
    loss, rows = compute_probe_loss(
        outputs,
        batch,
        config,
        corruption_delta=torch.zeros_like(final),
    )
    assert torch.isfinite(loss)
    assert set(rows) == {"anchor", "consistency", "ranking", "repeat", "total", "transfer"}
    loss.backward()
    assert final.grad is not None
