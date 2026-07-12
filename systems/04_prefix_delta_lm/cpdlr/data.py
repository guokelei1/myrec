"""Materialized probe dataset and request-level collation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .io import iter_jsonl
from .tokenization import PrefixTokenizer


class ProbeDataset:
    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        self.rows = list(iter_jsonl(path))
        if limit is not None:
            self.rows = self.rows[: int(limit)]
        if not self.rows:
            raise ValueError(f"empty probe dataset: {path}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


def collate_requests(
    rows: list[dict[str, Any]],
    tokenizer: PrefixTokenizer,
    seed: int,
    corruption: str | None,
    structured: bool,
) -> dict[str, Any]:
    max_candidates = max(len(row["candidates"]) for row in rows)
    pairs = []
    candidate_mask = torch.zeros((len(rows), max_candidates), dtype=torch.bool)
    labels = torch.zeros((len(rows), max_candidates), dtype=torch.float32)
    teacher = torch.zeros((len(rows), max_candidates), dtype=torch.float32)
    exact = torch.zeros((len(rows), max_candidates), dtype=torch.bool)
    history_present = torch.zeros(len(rows), dtype=torch.bool)
    for row_index, row in enumerate(rows):
        history_present[row_index] = bool(row.get("history"))
        candidates = row["candidates"]
        for candidate_index in range(max_candidates):
            candidate = candidates[min(candidate_index, len(candidates) - 1)]
            pairs.append((row, candidate))
            if candidate_index < len(candidates):
                candidate_mask[row_index, candidate_index] = True
                labels[row_index, candidate_index] = float(candidate["label"])
                teacher[row_index, candidate_index] = float(candidate["anchor_score"])
                exact[row_index, candidate_index] = bool(candidate["exact_repeat"])
    factual = tokenizer.batch_encode(
        pairs, "factual", seed, structured=structured
    )
    null = tokenizer.batch_encode(pairs, "null", seed, structured=structured)
    result: dict[str, Any] = {
        "candidate_mask": candidate_mask,
        "exact_repeat": exact,
        "factual_inputs": factual,
        "history_present": history_present,
        "labels": labels,
        "null_inputs": null,
        "request_ids": [str(row["request_id"]) for row in rows],
        "shape": (len(rows), max_candidates),
        "teacher_scores": teacher,
    }
    if corruption:
        result["corruption"] = corruption
        result["corrupt_inputs"] = tokenizer.batch_encode(
            pairs, corruption, seed, structured=structured
        )
        if corruption == "query_masked":
            result["corrupt_null_inputs"] = tokenizer.batch_encode(
                pairs, "query_masked_null", seed, structured=structured
            )
        else:
            result["corrupt_null_inputs"] = null
    return result


def move_inputs(inputs: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in inputs.items()}


def reshape_logits(values: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return values.reshape(shape[0], shape[1])
