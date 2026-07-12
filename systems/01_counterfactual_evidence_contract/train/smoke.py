"""Tiny tensor smoke fixtures; no repository data are touched."""

from __future__ import annotations

from typing import Any

import torch

from model.cect import CECTModel, multi_positive_listwise_loss
from train.engine import model_from_config


def tiny_batch(text_dim: int = 512, history_width: int = 20) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260708)
    batch_size, candidate_count = 2, 3
    batch = {
        "query": torch.randn(batch_size, text_dim, generator=generator),
        "candidates": torch.randn(
            batch_size, candidate_count, text_dim, generator=generator
        ),
        "candidate_indices": torch.tensor([[11, 12, 13], [21, 22, 23]]),
        "candidate_categories": torch.tensor([[4, 5, 6], [7, 8, 9]]),
        "candidate_mask": torch.ones(batch_size, candidate_count, dtype=torch.bool),
        "labels": torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "base_scores": torch.tensor([[0.2, -0.1, 0.0], [0.1, 0.0, -0.2]]),
        "history": torch.randn(
            batch_size, history_width, text_dim, generator=generator
        ),
        "history_indices": torch.full(
            (batch_size, history_width), -1, dtype=torch.long
        ),
        "history_categories": torch.zeros(
            batch_size, history_width, dtype=torch.long
        ),
        "history_event_weights": torch.zeros(batch_size, history_width),
        "history_mask": torch.zeros(batch_size, history_width, dtype=torch.bool),
        "wrong_history": torch.randn(
            batch_size, history_width, text_dim, generator=generator
        ),
        "wrong_history_indices": torch.full(
            (batch_size, history_width), -1, dtype=torch.long
        ),
        "wrong_history_categories": torch.zeros(
            batch_size, history_width, dtype=torch.long
        ),
        "wrong_history_event_weights": torch.zeros(batch_size, history_width),
        "wrong_history_mask": torch.zeros(
            batch_size, history_width, dtype=torch.bool
        ),
    }
    # Row 0 contains one protected exact event and two non-exact events.
    batch["history_indices"][0, :3] = torch.tensor([11, 31, 32])
    batch["history_categories"][0, :3] = torch.tensor([4, 5, 10])
    batch["history_event_weights"][0, :3] = torch.tensor([2.0, 1.0, 1.0])
    batch["history_mask"][0, :3] = True
    # Row 1 has no observational history, exercising the exact fallback.
    batch["wrong_history_indices"][:, :2] = torch.tensor([[41, 42], [51, 52]])
    batch["wrong_history_categories"][:, :2] = torch.tensor([[4, 12], [7, 13]])
    batch["wrong_history_event_weights"][:, :2] = 1.0
    batch["wrong_history_mask"][:, :2] = True
    return batch


def run_smoke(config: dict[str, Any], device: str) -> dict[str, Any]:
    model: CECTModel = model_from_config(config, "contract").to(device)
    batch = {key: value.to(device) for key, value in tiny_batch().items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    loss = multi_positive_listwise_loss(
        output.scores, batch["labels"], batch["candidate_mask"]
    )
    evidence_mask = output.nonexact_mask
    loss = loss + 0.01 * output.energies[evidence_mask].mean()
    loss = loss + 0.01 * output.values[evidence_mask].mean()
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("smoke loss is non-finite")
    loss.backward()
    required = {
        "certificate_head": model.energy_head.weight.grad,
        "transformer_attention": model.transformer.layers[0].self_attn.in_proj_weight.grad,
        "value_head": model.value_head.weight.grad,
    }
    gradient_norms = {}
    for name, gradient in required.items():
        if gradient is None or not bool(torch.isfinite(gradient).all().item()):
            raise FloatingPointError(f"missing/non-finite smoke gradient: {name}")
        norm = float(gradient.norm().detach().cpu())
        if norm <= 0.0:
            raise ValueError(f"zero smoke gradient: {name}")
        gradient_norms[name] = norm
    optimizer.step()
    return {
        "device": device,
        "finite_loss": True,
        "gradient_norms": gradient_norms,
        "loss": float(loss.detach().cpu()),
        "optimizer_step_completed": True,
    }
