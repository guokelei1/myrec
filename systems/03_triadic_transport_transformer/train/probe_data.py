"""Candidate-level views over request-level C03 prepared features."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset


class ProbeCandidateDataset(Dataset[dict[str, Any]]):
    def __init__(self, feature_path: str | Path, *, validation: bool) -> None:
        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        self.metadata = payload["metadata"]
        self.requests = [
            request
            for request in payload["requests"]
            if bool(request["is_validation"]) is validation
        ]
        self.index: list[tuple[int, int]] = []
        for request_index, request in enumerate(self.requests):
            for candidate_index in range(int(request["candidates"].shape[0])):
                self.index.append((request_index, candidate_index))
        if not self.index:
            split = "validation" if validation else "fit"
            raise ValueError(f"empty prepared {split} candidate dataset")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        request_index, candidate_index = self.index[index]
        request = self.requests[request_index]
        candidate_hash = request["candidate_item_hashes"][candidate_index]
        exact_match = (
            request["history_item_hashes"].eq(candidate_hash) & request["history_mask"]
        )
        return {
            "request_id": request["request_id"],
            "candidate_item_id": request["candidate_item_ids"][candidate_index],
            "user_hash": torch.tensor(request["user_hash"], dtype=torch.long),
            "candidate_item_hash": candidate_hash,
            "query": request["query"].float(),
            "candidate": request["candidates"][candidate_index].float(),
            "history": request["history"].float(),
            "coarse_history": request["coarse_history"].float(),
            "history_mask": request["history_mask"],
            "event_types": request["event_types"],
            "history_item_hashes": request["history_item_hashes"],
            "exact_match": exact_match,
            "label": request["labels"][candidate_index],
        }


def collate_candidates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "user_hash",
        "candidate_item_hash",
        "query",
        "candidate",
        "history",
        "coarse_history",
        "history_mask",
        "event_types",
        "history_item_hashes",
        "exact_match",
        "label",
    )
    result: dict[str, Any] = {
        "request_id": [row["request_id"] for row in rows],
        "candidate_item_id": [row["candidate_item_id"] for row in rows],
    }
    for key in tensor_keys:
        result[key] = torch.stack([row[key] for row in rows])
    return result


def to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
