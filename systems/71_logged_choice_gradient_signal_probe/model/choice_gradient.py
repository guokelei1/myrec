"""Parameter-free logged-choice gradient signal operators."""

from __future__ import annotations

import torch
from torch import Tensor


def normalize_rows(values: Tensor, epsilon: float) -> Tensor:
    if values.ndim not in (1, 2):
        raise ValueError("C71 normalization expects [D] or [N,D]")
    return values / values.norm(dim=-1, keepdim=True).clamp_min(float(epsilon))


def stable_softmax(logits: Tensor) -> Tensor:
    if logits.ndim != 1 or not len(logits):
        raise ValueError("C71 softmax expects a nonempty vector")
    shifted = logits - logits.max()
    weights = shifted.exp()
    return weights / weights.sum().clamp_min(torch.finfo(weights.dtype).tiny)


def episode_value(
    historical_query: Tensor,
    slate: Tensor,
    selected_index: int,
    *,
    mode: str,
    temperature: float,
    epsilon: float,
) -> Tensor:
    if historical_query.ndim != 1 or slate.ndim != 2:
        raise ValueError("C71 episode shapes differ")
    if slate.shape[1] != historical_query.shape[0] or not 0 <= selected_index < len(slate):
        raise ValueError("C71 selected episode index differs")
    query = normalize_rows(historical_query, epsilon)
    items = normalize_rows(slate, epsilon)
    selected = items[int(selected_index)]
    if mode == "choice_gradient":
        weights = stable_softmax((items @ query) / float(temperature))
        value = selected - weights @ items
    elif mode == "uniform_slate":
        value = selected - items.mean(dim=0)
    elif mode == "positive_only":
        value = selected
    else:
        raise ValueError(f"unknown C71 episode mode: {mode}")
    norm = value.norm()
    return torch.where(norm > float(epsilon), value / norm.clamp_min(float(epsilon)), torch.zeros_like(value))


def memory_value(
    current_query: Tensor,
    episode_queries: Tensor,
    episode_values: Tensor,
    *,
    temperature: float,
    epsilon: float,
) -> Tensor:
    if episode_queries.ndim != 2 or episode_values.shape != episode_queries.shape:
        raise ValueError("C71 memory episode shapes differ")
    if not len(episode_queries):
        return torch.zeros_like(current_query)
    query = normalize_rows(current_query, epsilon)
    keys = normalize_rows(episode_queries, epsilon)
    weights = stable_softmax((keys @ query) / float(temperature))
    return weights @ episode_values


def score_memory(
    current_query: Tensor,
    candidates: Tensor,
    memory: Tensor,
    *,
    correction_scale: float,
    epsilon: float,
) -> tuple[Tensor, Tensor]:
    query = normalize_rows(current_query, epsilon)
    items = normalize_rows(candidates, epsilon)
    base = items @ query
    correction = float(correction_scale) * (items @ memory)
    return base + correction, correction
