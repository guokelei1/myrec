"""Fixed latent-state generator for the C45 data-free gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class SyntheticRows:
    query: torch.Tensor
    candidates: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    labels: torch.Tensor
    oracle_scores: torch.Tensor
    query_only_scores: torch.Tensor

    def to(self, device: torch.device) -> "SyntheticRows":
        return SyntheticRows(*[value.to(device) for value in self.__dict__.values()])

    def subset(self, indices: torch.Tensor) -> "SyntheticRows":
        return SyntheticRows(*[value[indices] for value in self.__dict__.values()])


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), 1e-8)


def _fixed_matrix(rng: np.random.Generator, dim: int, scale: float) -> np.ndarray:
    value = rng.normal(size=(dim, dim))
    q, _ = np.linalg.qr(value)
    return (scale * q).astype(np.float32)


def generate(
    config: Mapping[str, Any], *, requests: int, split_offset: int
) -> SyntheticRows:
    row = config["generator"]
    seed = int(row["seed"]) + int(split_offset)
    rng = np.random.default_rng(seed)
    dim = int(row["input_dim"])
    length = int(row["history_length"])
    candidate_count = int(row["candidate_count"])
    query_strength = float(row["query_strength"])
    history_strength = float(row["history_strength"])
    event_noise = float(row["event_noise"])
    candidate_noise = float(row["candidate_noise"])
    hard_fraction = float(row["hard_negative_fraction"])

    # Generator matrices are fixed across train/validation by deriving them
    # from the unshifted registered seed.
    matrix_rng = np.random.default_rng(int(row["seed"]))
    transition = _fixed_matrix(matrix_rng, dim, 0.72)
    event_map = _fixed_matrix(matrix_rng, dim, 0.88)
    cross_map = _fixed_matrix(matrix_rng, dim, 0.42)
    position = matrix_rng.normal(0.0, 0.35, size=(length, dim)).astype(np.float32)

    query = _normalize(rng.normal(size=(requests, dim))).astype(np.float32)
    user = _normalize(rng.normal(size=(requests, dim))).astype(np.float32)
    state = rng.normal(0.0, 0.25, size=(requests, dim)).astype(np.float32)
    history_rows: list[np.ndarray] = []
    effects: list[np.ndarray] = []
    for step in range(length):
        local = np.tanh(state @ transition.T + position[step])
        event = np.tanh(
            1.45 * local
            + 0.75 * user
            + rng.normal(0.0, event_noise, size=(requests, dim))
        ).astype(np.float32)
        null_state = np.tanh(state @ transition.T + position[step]).astype(np.float32)
        factual_state = np.tanh(
            state @ transition.T
            + position[step]
            + event @ event_map.T
            + (state * event) @ cross_map.T
        ).astype(np.float32)
        history_rows.append(event)
        effects.append(factual_state - null_state)
        state = factual_state
    history = np.stack(history_rows, axis=1).astype(np.float32)
    effect = np.stack(effects, axis=1)
    attention = np.einsum("bd,bhd->bh", query, effect) / np.sqrt(dim)
    attention = np.exp(attention - attention.max(axis=1, keepdims=True))
    attention /= attention.sum(axis=1, keepdims=True)
    profile = np.einsum("bh,bhd->bd", attention, effect)
    target = _normalize(query_strength * query + history_strength * profile)

    candidates = np.empty((requests, candidate_count, dim), dtype=np.float32)
    candidates[:, 0] = _normalize(
        target + rng.normal(0.0, candidate_noise, size=(requests, dim))
    )
    hard_count = max(1, int(round((candidate_count - 1) * hard_fraction)))
    for index in range(1, 1 + hard_count):
        candidates[:, index] = _normalize(
            query + rng.normal(0.0, candidate_noise, size=(requests, dim))
        )
    for index in range(1 + hard_count, candidate_count):
        candidates[:, index] = _normalize(rng.normal(size=(requests, dim)))

    # Randomize the positive position without changing its construction.
    positive_position = rng.integers(0, candidate_count, size=requests)
    for request, destination in enumerate(positive_position):
        candidates[request, [0, destination]] = candidates[request, [destination, 0]]
    labels = np.zeros((requests, candidate_count), dtype=np.float32)
    labels[np.arange(requests), positive_position] = 1.0
    oracle = np.einsum("bd,bcd->bc", target, candidates).astype(np.float32)
    query_only = np.einsum("bd,bcd->bc", query, candidates).astype(np.float32)
    return SyntheticRows(
        query=torch.from_numpy(query),
        candidates=torch.from_numpy(candidates),
        history=torch.from_numpy(history),
        history_mask=torch.ones(requests, length, dtype=torch.bool),
        labels=torch.from_numpy(labels),
        oracle_scores=torch.from_numpy(oracle),
        query_only_scores=torch.from_numpy(query_only),
    )


def wrong_history(rows: SyntheticRows) -> torch.Tensor:
    count = rows.history.shape[0]
    donor = (torch.arange(count, device=rows.history.device) + 1) % count
    return rows.history[donor]


def shuffled_history(rows: SyntheticRows) -> torch.Tensor:
    length = rows.history.shape[1]
    if length < 2:
        raise ValueError("C45 shuffle requires at least two events")
    order = torch.arange(length - 1, -1, -1, device=rows.history.device)
    if bool(torch.equal(order, torch.arange(length, device=rows.history.device))):
        raise RuntimeError("C45 shuffle is identity")
    return rows.history[:, order]


def ndcg10(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = labels.argmax(dim=1)
    order = scores.argsort(dim=1, descending=True, stable=True)
    rank = (order == positive[:, None]).to(torch.int64).argmax(dim=1)
    value = 1.0 / torch.log2(rank.to(torch.float64) + 2.0)
    return torch.where(rank < 10, value, torch.zeros_like(value))


def complete_orders(scores: torch.Tensor) -> torch.Tensor:
    return scores.argsort(dim=1, descending=True, stable=True)
