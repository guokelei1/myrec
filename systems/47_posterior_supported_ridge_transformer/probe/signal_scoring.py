"""Frozen NumPy realization of the C47 fixed operators.

This module is deliberately dataset-agnostic.  It consumes only query,
history, and candidate state matrices and implements the exact S0 operators
declared before outcome access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FixedScores:
    base: np.ndarray
    posterior_supported: np.ndarray
    plain_ridge: np.ndarray
    softmax_attention: np.ndarray
    correction: np.ndarray
    plain_correction: np.ndarray
    support: np.ndarray
    query_write: np.ndarray


def normalize_rows(values: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("C47 row normalization expects a matrix")
    denominator = np.maximum(np.linalg.norm(values, axis=1, keepdims=True), epsilon)
    return values / denominator


def normalize_vector(value: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 1:
        raise ValueError("C47 vector normalization expects one vector")
    return value / max(float(np.linalg.norm(value)), epsilon)


def fixed_scores(
    query: np.ndarray,
    history: np.ndarray,
    candidates: np.ndarray,
    *,
    ridge: float = 1.0,
    softmax_temperature: float = 0.1,
    epsilon: float = 1e-6,
) -> FixedScores:
    """Score one request with the frozen base, KRR, PSRT, and attention rules."""

    if ridge <= 0:
        raise ValueError("C47 ridge must be positive")
    if softmax_temperature <= 0:
        raise ValueError("C47 softmax temperature must be positive")
    q = normalize_vector(query, epsilon=epsilon)
    c = normalize_rows(candidates, epsilon=epsilon)
    h_raw = np.asarray(history, dtype=np.float32)
    if h_raw.ndim != 2 or h_raw.shape[1] != q.shape[0]:
        raise ValueError("C47 history shape differs")
    if c.shape[1] != q.shape[0] or not len(c):
        raise ValueError("C47 candidate shape differs")
    base = c @ q
    if not len(h_raw):
        zeros = np.zeros(len(c), dtype=np.float32)
        return FixedScores(
            base=base.astype(np.float32, copy=False),
            posterior_supported=base.astype(np.float32, copy=True),
            plain_ridge=base.astype(np.float32, copy=True),
            softmax_attention=base.astype(np.float32, copy=True),
            correction=zeros,
            plain_correction=zeros.copy(),
            support=zeros.copy(),
            query_write=np.zeros_like(q),
        )

    h = normalize_rows(h_raw, epsilon=epsilon)
    normal = h @ h.T + np.eye(len(h), dtype=np.float32) * float(ridge)
    hq = h @ q
    alpha_q = np.linalg.solve(normal, hq)
    query_write = h.T @ alpha_q
    hc = h @ c.T
    alpha_c = np.linalg.solve(normal, hc)
    support = np.clip(np.sum(hc * alpha_c, axis=0), 0.0, 1.0)
    plain_correction = c @ query_write
    correction = support * plain_correction

    logits = (h @ q) / float(softmax_temperature)
    logits = logits - float(np.max(logits))
    weights = np.exp(logits).astype(np.float32, copy=False)
    weights /= float(np.sum(weights))
    softmax_write = weights @ h
    softmax_correction = c @ softmax_write
    return FixedScores(
        base=base.astype(np.float32, copy=False),
        posterior_supported=(base + correction).astype(np.float32, copy=False),
        plain_ridge=(base + plain_correction).astype(np.float32, copy=False),
        softmax_attention=(base + softmax_correction).astype(np.float32, copy=False),
        correction=correction.astype(np.float32, copy=False),
        plain_correction=plain_correction.astype(np.float32, copy=False),
        support=support.astype(np.float32, copy=False),
        query_write=query_write.astype(np.float32, copy=False),
    )


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    values = (
        np.concatenate(rows).astype(np.float32, copy=False)
        if rows
        else np.empty(0, dtype=np.float32)
    )
    return np.asarray(offsets, dtype=np.int64), values


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    offsets = np.asarray(offsets, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    return [
        values[int(offsets[index]) : int(offsets[index + 1])].copy()
        for index in range(len(offsets) - 1)
    ]


def max_row_difference(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("C47 difference rows differ")
    return max(float(np.max(np.abs(a - b))) for a, b in zip(first, second))
