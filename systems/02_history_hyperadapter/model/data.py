"""Packed, label-isolated data access for the C02 probe."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np


def _request_ids(path: Path) -> list[str]:
    rows: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(str(json.loads(line)["request_id"]))
    return rows


@dataclass
class C02Split:
    split: str
    request_ids: list[str]
    candidate_offsets: np.ndarray
    candidate_embedding_indices: np.ndarray
    candidate_item_ids: np.ndarray
    candidate_labels: np.ndarray | None
    history_offsets: np.ndarray
    history_embedding_indices: np.ndarray
    history_item_ids: np.ndarray
    history_event_weights: np.ndarray
    history_category_ids: np.ndarray
    candidate_category_ids: np.ndarray
    wrong_request_indices: np.ndarray
    base_scores: np.ndarray | None

    @classmethod
    def load(
        cls,
        shared_root: str | Path,
        feature_root: str | Path,
        split: str,
    ) -> "C02Split":
        shared = Path(shared_root) / split
        local = Path(feature_root) / split
        base_path = local / "base_scores.npy"
        obj = cls(
            split=split,
            request_ids=_request_ids(shared / "request_ids.jsonl"),
            candidate_offsets=np.load(shared / "candidate_offsets.npy", mmap_mode="r"),
            candidate_embedding_indices=np.load(
                shared / "candidate_embedding_indices.npy", mmap_mode="r"
            ),
            candidate_item_ids=np.load(
                shared / "candidate_item_ids.npy", mmap_mode="r"
            ),
            # Evaluation splits are label-free by construction.  Do not even
            # open the packed label-shaped array outside train.
            candidate_labels=(
                np.load(shared / "candidate_labels.npy", mmap_mode="r")
                if split == "train"
                else None
            ),
            history_offsets=np.load(shared / "history_offsets.npy", mmap_mode="r"),
            history_embedding_indices=np.load(
                shared / "history_embedding_indices.npy", mmap_mode="r"
            ),
            history_item_ids=np.load(local / "history_item_ids.npy", mmap_mode="r"),
            history_event_weights=np.load(
                local / "history_event_weights.npy", mmap_mode="r"
            ),
            history_category_ids=np.load(
                local / "history_category_ids.npy", mmap_mode="r"
            ),
            candidate_category_ids=np.load(
                local / "candidate_category_ids.npy", mmap_mode="r"
            ),
            wrong_request_indices=np.load(
                local / "wrong_request_indices.npy", mmap_mode="r"
            ),
            base_scores=(np.load(base_path, mmap_mode="r") if base_path.exists() else None),
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        requests = len(self.request_ids)
        if len(self.candidate_offsets) != requests + 1:
            raise ValueError(f"candidate offsets mismatch for {self.split}")
        if len(self.history_offsets) != requests + 1:
            raise ValueError(f"history offsets mismatch for {self.split}")
        candidate_rows = int(self.candidate_offsets[-1])
        for array in (
            self.candidate_embedding_indices,
            self.candidate_item_ids,
            self.candidate_category_ids,
        ):
            if len(array) != candidate_rows:
                raise ValueError(f"candidate row mismatch for {self.split}")
        if self.split == "train":
            if self.candidate_labels is None or len(self.candidate_labels) != candidate_rows:
                raise ValueError("train labels are missing or misaligned")
        elif self.candidate_labels is not None:
            raise ValueError(f"non-train labels must not be loaded for {self.split}")
        if self.base_scores is not None and len(self.base_scores) != candidate_rows:
            raise ValueError(f"base score row mismatch for {self.split}")
        history_rows = int(self.history_offsets[-1])
        for array in (
            self.history_embedding_indices,
            self.history_item_ids,
            self.history_event_weights,
            self.history_category_ids,
        ):
            if len(array) != history_rows:
                raise ValueError(f"history row mismatch for {self.split}")
        if len(self.wrong_request_indices) != requests:
            raise ValueError(f"wrong-donor request mismatch for {self.split}")
        for index, donor in enumerate(self.wrong_request_indices):
            donor = int(donor)
            if donor < 0 or donor >= requests:
                raise ValueError(f"invalid wrong donor {donor} for request {index}")

    def __len__(self) -> int:
        return len(self.request_ids)

    def candidate_count(self, index: int) -> int:
        return int(self.candidate_offsets[index + 1] - self.candidate_offsets[index])

    def history_count(self, index: int, history_limit: int) -> int:
        count = int(self.history_offsets[index + 1] - self.history_offsets[index])
        return min(count, history_limit)

    def structural_subset(self, index: int) -> str:
        """Classify from the complete history, independent of model truncation."""

        hs = int(self.history_offsets[index])
        he = int(self.history_offsets[index + 1])
        if hs == he:
            return "no_history"
        cs = int(self.candidate_offsets[index])
        ce = int(self.candidate_offsets[index + 1])
        repeated = np.intersect1d(
            np.asarray(self.candidate_item_ids[cs:ce]),
            np.asarray(self.history_item_ids[hs:he]),
            assume_unique=False,
        ).size
        return "repeat" if repeated else "nonrepeat"


def frozen_train_indices(
    data: C02Split,
    seed: int,
    train_requests: int,
    validation_requests: int,
    internal_fraction_boundary: float,
) -> tuple[np.ndarray, np.ndarray]:
    boundary = int(math.floor(len(data) * internal_fraction_boundary))
    if not 0 < boundary < len(data):
        raise ValueError("invalid internal train boundary")
    if train_requests > boundary:
        raise ValueError("requested train sample exceeds pre-boundary population")
    if validation_requests >= len(data) - boundary:
        raise ValueError("validation sample does not fit after internal boundary")
    rng = np.random.default_rng(seed)
    train = np.sort(
        rng.choice(np.arange(boundary, dtype=np.int64), train_requests, replace=False)
    )
    validation = np.arange(len(data) - validation_requests, len(data), dtype=np.int64)
    if np.intersect1d(train, validation).size:
        raise AssertionError("train and internal validation overlap")
    return train, validation


def iter_request_batches(
    data: C02Split,
    indices: Sequence[int] | np.ndarray,
    *,
    history_limit: int,
    max_requests: int,
    max_padded_candidates: int,
    max_padded_history: int,
    seed: int,
    shuffle: bool,
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    batch: list[int] = []
    max_candidates = 0
    max_history = 0
    for raw_index in order:
        index = int(raw_index)
        candidate_count = data.candidate_count(index)
        history_count = data.history_count(index, history_limit)
        next_size = len(batch) + 1
        next_candidates = max(max_candidates, candidate_count)
        next_history = max(max_history, max(history_count, 1))
        if batch and (
            next_size > max_requests
            or next_size * next_candidates > max_padded_candidates
            or next_size * next_history > max_padded_history
        ):
            yield np.asarray(batch, dtype=np.int64)
            batch = []
            max_candidates = 0
            max_history = 0
        batch.append(index)
        max_candidates = max(max_candidates, candidate_count)
        max_history = max(max_history, max(history_count, 1))
    if batch:
        yield np.asarray(batch, dtype=np.int64)


def collate_requests(
    data: C02Split,
    request_indices: np.ndarray,
    *,
    history_limit: int,
) -> dict[str, np.ndarray | list[str]]:
    batch = len(request_indices)
    max_candidates = max(data.candidate_count(int(index)) for index in request_indices)
    max_history = max(
        1,
        max(data.history_count(int(index), history_limit) for index in request_indices),
    )
    candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
    candidate_item_ids = np.full((batch, max_candidates), -1, dtype=np.int64)
    candidate_category_ids = np.full((batch, max_candidates), -1, dtype=np.int64)
    candidate_labels = np.zeros((batch, max_candidates), dtype=np.float32)
    candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
    base_scores = np.zeros((batch, max_candidates), dtype=np.float32)

    history_indices = np.zeros((batch, max_history), dtype=np.int64)
    history_item_ids = np.full((batch, max_history), -1, dtype=np.int64)
    history_category_ids = np.full((batch, max_history), -1, dtype=np.int64)
    history_event_weights = np.zeros((batch, max_history), dtype=np.float32)
    history_mask = np.zeros((batch, max_history), dtype=bool)
    wrong_history_indices = np.zeros((batch, max_history), dtype=np.int64)
    wrong_history_item_ids = np.full((batch, max_history), -1, dtype=np.int64)
    wrong_history_category_ids = np.full((batch, max_history), -1, dtype=np.int64)
    wrong_history_event_weights = np.zeros((batch, max_history), dtype=np.float32)
    wrong_history_mask = np.zeros((batch, max_history), dtype=bool)
    shuffled_history_indices = np.zeros((batch, max_history), dtype=np.int64)
    shuffled_history_item_ids = np.full((batch, max_history), -1, dtype=np.int64)
    shuffled_history_category_ids = np.full((batch, max_history), -1, dtype=np.int64)
    shuffled_history_event_weights = np.zeros((batch, max_history), dtype=np.float32)
    shuffled_history_mask = np.zeros((batch, max_history), dtype=bool)
    request_ids: list[str] = []

    for row, raw_index in enumerate(request_indices):
        index = int(raw_index)
        request_ids.append(data.request_ids[index])
        cs = int(data.candidate_offsets[index])
        ce = int(data.candidate_offsets[index + 1])
        count = ce - cs
        candidate_indices[row, :count] = data.candidate_embedding_indices[cs:ce]
        candidate_item_ids[row, :count] = data.candidate_item_ids[cs:ce]
        candidate_category_ids[row, :count] = data.candidate_category_ids[cs:ce]
        if data.candidate_labels is not None:
            candidate_labels[row, :count] = data.candidate_labels[cs:ce]
        candidate_mask[row, :count] = True
        if data.base_scores is not None:
            base_scores[row, :count] = data.base_scores[cs:ce]

        _fill_history(
            data,
            index,
            history_limit,
            history_indices[row],
            history_item_ids[row],
            history_category_ids[row],
            history_event_weights[row],
            history_mask[row],
            reverse=False,
        )
        donor = int(data.wrong_request_indices[index])
        _fill_history(
            data,
            donor,
            history_limit,
            wrong_history_indices[row],
            wrong_history_item_ids[row],
            wrong_history_category_ids[row],
            wrong_history_event_weights[row],
            wrong_history_mask[row],
            reverse=False,
        )
        _fill_history(
            data,
            index,
            history_limit,
            shuffled_history_indices[row],
            shuffled_history_item_ids[row],
            shuffled_history_category_ids[row],
            shuffled_history_event_weights[row],
            shuffled_history_mask[row],
            reverse=True,
        )

    repeat_mask = (
        candidate_item_ids[:, :, None] == history_item_ids[:, None, :]
    ) & candidate_mask[:, :, None] & history_mask[:, None, :]
    wrong_repeat_mask = (
        candidate_item_ids[:, :, None] == wrong_history_item_ids[:, None, :]
    ) & candidate_mask[:, :, None] & wrong_history_mask[:, None, :]
    shuffled_repeat_mask = (
        candidate_item_ids[:, :, None] == shuffled_history_item_ids[:, None, :]
    ) & candidate_mask[:, :, None] & shuffled_history_mask[:, None, :]

    return {
        "request_indices": np.asarray(request_indices, dtype=np.int64),
        "request_ids": request_ids,
        "candidate_indices": candidate_indices,
        "candidate_item_ids": candidate_item_ids,
        "candidate_category_ids": candidate_category_ids,
        "candidate_labels": candidate_labels,
        "candidate_mask": candidate_mask,
        "base_scores": base_scores,
        "history_indices": history_indices,
        "history_item_ids": history_item_ids,
        "history_category_ids": history_category_ids,
        "history_event_weights": history_event_weights,
        "history_mask": history_mask,
        "repeat_mask": repeat_mask,
        "wrong_history_indices": wrong_history_indices,
        "wrong_history_item_ids": wrong_history_item_ids,
        "wrong_history_category_ids": wrong_history_category_ids,
        "wrong_history_event_weights": wrong_history_event_weights,
        "wrong_history_mask": wrong_history_mask,
        "wrong_repeat_mask": wrong_repeat_mask,
        "shuffled_history_indices": shuffled_history_indices,
        "shuffled_history_item_ids": shuffled_history_item_ids,
        "shuffled_history_category_ids": shuffled_history_category_ids,
        "shuffled_history_event_weights": shuffled_history_event_weights,
        "shuffled_history_mask": shuffled_history_mask,
        "shuffled_repeat_mask": shuffled_repeat_mask,
    }


def _fill_history(
    data: C02Split,
    request_index: int,
    history_limit: int,
    output_indices: np.ndarray,
    output_item_ids: np.ndarray,
    output_category_ids: np.ndarray,
    output_event_weights: np.ndarray,
    output_mask: np.ndarray,
    *,
    reverse: bool,
) -> None:
    start = int(data.history_offsets[request_index])
    end = int(data.history_offsets[request_index + 1])
    start = max(start, end - history_limit)
    order = np.arange(start, end, dtype=np.int64)
    if reverse:
        order = order[::-1]
    count = min(len(order), len(output_indices))
    if not count:
        return
    order = order[:count]
    output_indices[:count] = data.history_embedding_indices[order]
    output_item_ids[:count] = data.history_item_ids[order]
    output_category_ids[:count] = data.history_category_ids[order]
    output_event_weights[:count] = data.history_event_weights[order]
    output_mask[:count] = True
