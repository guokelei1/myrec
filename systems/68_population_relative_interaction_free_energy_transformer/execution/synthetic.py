"""Locked data-free C68 population-nuisance task generator."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F


def _orthogonal_unit(value: Tensor, *axes: Tensor) -> Tensor:
    for axis in axes:
        value = value - (value * axis).sum(dim=-1, keepdim=True) * axis
    return F.normalize(value, dim=-1)


def _shuffle_events(values: Tensor, generator: torch.Generator) -> Tensor:
    batch, count, dim = values.shape
    order = torch.stack([torch.randperm(count, generator=generator) for _ in range(batch)])
    return torch.gather(values, 1, order[..., None].expand(-1, -1, dim))


def synthetic_batch(
    config: Mapping[str, Any],
    *,
    seed: int,
    batch_size: int,
    device: torch.device,
    unsupported_fraction: float | None = None,
) -> dict[str, Tensor]:
    row = config["synthetic_G0"]
    dim = int(config["model"]["input_dim"])
    history_count = int(row["history_events"])
    useful_count = int(row["useful_events"])
    reference_count = int(row["reference_events"])
    candidate_count = int(row["candidates"])
    if not (0 < useful_count < history_count):
        raise ValueError("useful_events must be between zero and history_events")
    fraction = float(row["unsupported_fraction"] if unsupported_fraction is None else unsupported_fraction)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    query = F.normalize(torch.randn(batch_size, dim, generator=generator), dim=-1)
    preference = _orthogonal_unit(
        torch.randn(batch_size, dim, generator=generator), query
    )
    wrong_preference = _orthogonal_unit(
        torch.randn(batch_size, dim, generator=generator), query, preference
    )
    population = _orthogonal_unit(
        torch.randn(batch_size, dim, generator=generator), query, preference
    ) * float(row["nuisance_scale"])

    candidates = F.normalize(
        torch.randn(batch_size, candidate_count, dim, generator=generator), dim=-1
    )
    base_scores = float(row["base_strength"]) * torch.einsum(
        "bd,bcd->bc", query, candidates
    )
    personalized_direction = query + float(row["preference_strength"]) * preference
    personalized_scores = torch.einsum("bd,bcd->bc", personalized_direction, candidates)

    unsupported = torch.rand(batch_size, generator=generator) < fraction
    labels = torch.where(
        unsupported,
        base_scores.argmax(dim=1),
        personalized_scores.argmax(dim=1),
    )

    pop_scale = float(row["population_user_scale"])
    noise = float(row["event_noise"])
    history_axes = F.normalize(
        torch.randn(batch_size, history_count, dim, generator=generator), dim=-1
    )
    reference_axes = F.normalize(
        torch.randn(batch_size, reference_count, dim, generator=generator), dim=-1
    )
    history = population[:, None, :] + pop_scale * history_axes
    reference = population[:, None, :] + pop_scale * reference_axes

    useful = population[:, None, :] + preference[:, None, :]
    useful = useful + noise * torch.randn(
        batch_size, useful_count, dim, generator=generator
    )
    history[:, :useful_count] = useful
    wrong = history.clone()
    wrong_useful = population[:, None, :] + wrong_preference[:, None, :]
    wrong_useful = wrong_useful + noise * torch.randn(
        batch_size, useful_count, dim, generator=generator
    )
    wrong[:, :useful_count] = wrong_useful

    if bool(unsupported.any()):
        unsupported_axes = F.normalize(
            torch.randn(int(unsupported.sum()), history_count, dim, generator=generator),
            dim=-1,
        )
        neutral = population[unsupported, None, :] + pop_scale * unsupported_axes
        history[unsupported] = neutral
        wrong[unsupported] = neutral.roll(1, dims=1)

    history = _shuffle_events(history, generator)
    wrong = _shuffle_events(wrong, generator)
    history = F.normalize(
        history + noise * torch.randn(history.shape, generator=generator), dim=-1
    )
    wrong = F.normalize(
        wrong + noise * torch.randn(wrong.shape, generator=generator), dim=-1
    )
    reference = F.normalize(
        reference + noise * torch.randn(reference.shape, generator=generator), dim=-1
    )

    return {
        "query": query.to(device),
        "candidates": candidates.to(device),
        "history": history.to(device),
        "wrong_history": wrong.to(device),
        "reference": reference.to(device),
        "base_scores": base_scores.to(device),
        "labels": labels.to(device),
        "unsupported": unsupported.to(device),
        "history_present": torch.ones(batch_size, dtype=torch.bool, device=device),
        "query_present": torch.ones(batch_size, dtype=torch.bool, device=device),
    }
