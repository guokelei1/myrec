"""Label-staged C26 token interface for C64."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from myrec.eval.metrics import ScoredCandidate, sort_candidates


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    @property
    def positions(self) -> dict[int, int]:
        return {int(value): row for row, value in enumerate(self.request_indices)}

    def row(self, index: int, count: int) -> np.ndarray:
        position = self.positions
        if int(index) not in position:
            raise PermissionError(f"C64 exposed fit label unavailable: {index}")
        row = position[int(index)]
        start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
        if stop - start != count:
            raise ValueError("C64 fit label candidate count differs")
        return np.asarray(self.values[start:stop], dtype=np.float32).copy()


class PackedData:
    def __init__(self, root: Path) -> None:
        self.request_ids = [
            str(json.loads(line)["request_id"])
            for line in (root / "request_ids.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.candidate_offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_indices = np.load(root / "candidate_embedding_indices.npy", mmap_mode="r")
        self.candidate_item_ids = np.load(root / "candidate_item_ids.npy", mmap_mode="r")
        self.history_offsets = np.load(root / "history_offsets.npy", mmap_mode="r")
        self.history_indices = np.load(root / "history_embedding_indices.npy", mmap_mode="r")
        self.history_weights = np.load(root / "history_event_weights.npy", mmap_mode="r")

    def candidates(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_indices[start:stop], dtype=np.int64)

    def candidate_ids(self, index: int) -> list[str]:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return [str(value) for value in self.candidate_item_ids[start:stop].tolist()]

    def history(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return (
            np.asarray(self.history_indices[start:stop], dtype=np.int64),
            np.asarray(self.history_weights[start:stop], dtype=np.float32),
        )


def deterministic_split(
    indices: Sequence[int], request_ids: Sequence[str], *, seed: int, train_count: int
) -> tuple[list[int], list[int]]:
    ranked = sorted(
        (int(index) for index in indices),
        key=lambda index: hashlib.sha256(
            f"c64:{seed}:{request_ids[index]}".encode()
        ).digest(),
    )
    return ranked[:train_count], ranked[train_count:]


class C64Store:
    def __init__(self, config: Mapping[str, Any], repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.data = PackedData(repo_root / config["paths"]["packed_train_root"])
        self.selection = json.loads(
            (repo_root / config["paths"]["c26_selection"]).read_text(encoding="utf-8")
        )
        root = repo_root / config["paths"]["c26_artifact_root"]
        self.root = root
        self.feature_indices = np.load(root / "feature_request_indices.npy", mmap_mode="r")
        self.feature_position = {int(value): row for row, value in enumerate(self.feature_indices)}
        self.feature_offsets = np.load(root / "feature_candidate_offsets.npy", mmap_mode="r")
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self.item_indices = np.load(root / "item_embedding_indices.npy", mmap_mode="r")
        self.item_token_ids = np.load(root / "item_token_ids.npy", mmap_mode="r")
        self.item_attention = np.load(root / "item_attention_mask.npy", mmap_mode="r")
        self.item_content = np.load(root / "item_content_mask.npy", mmap_mode="r")
        self.query_token_ids = np.load(root / "query_token_ids.npy", mmap_mode="r")
        self.query_attention = np.load(root / "query_attention_mask.npy", mmap_mode="r")
        self.query_content = np.load(root / "query_content_mask.npy", mmap_mode="r")
        fit = [int(value) for value in self.selection["roles"]["fit"]["indices"]]
        self.train_indices, self.validation_indices = deterministic_split(
            fit,
            self.data.request_ids,
            seed=int(config["selection"]["split_seed"]),
            train_count=int(config["selection"]["train_requests"]),
        )
        donors = [
            int(value)
            for value in self.selection["wrong_history_donors"]["fit"]["indices"]
        ]
        self.donor = dict(zip(fit, donors))
        self._labels: CompactLabels | None = None

    def _fit_labels(self) -> CompactLabels:
        if self._labels is None:
            self._labels = CompactLabels(
                request_indices=np.load(self.root / "fit_request_indices.npy", mmap_mode="r"),
                offsets=np.load(self.root / "fit_label_offsets.npy", mmap_mode="r"),
                values=np.load(self.root / "fit_labels.npy", mmap_mode="r"),
            )
        return self._labels

    def labels(self, index: int) -> np.ndarray:
        return self._fit_labels().row(index, len(self.data.candidates(index)))

    def candidate_hash(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for index in indices:
            payload = json.dumps(
                [self.data.request_ids[int(index)], self.data.candidate_ids(int(index))],
                separators=(",", ":"),
            ).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def split_manifest(self) -> dict[str, Any]:
        overlap = set(self.train_indices) & set(self.validation_indices)
        return {
            "candidate_id": "c64",
            "train_indices": self.train_indices,
            "validation_indices": self.validation_indices,
            "train_requests": len(self.train_indices),
            "validation_requests": len(self.validation_indices),
            "overlap": len(overlap),
            "train_candidate_hash": self.candidate_hash(self.train_indices),
            "validation_candidate_hash": self.candidate_hash(self.validation_indices),
            "fit_labels_opened": False,
            "fresh_dev_test_qrels_opened": False,
        }

    def _item_rows(self, indices: np.ndarray) -> np.ndarray:
        rows = np.searchsorted(self.item_indices, indices)
        if bool((rows >= len(self.item_indices)).any()) or not np.array_equal(
            np.asarray(self.item_indices[rows], dtype=np.int64), indices
        ):
            raise KeyError("C64 item token is unavailable")
        return rows

    @staticmethod
    def _sample_positions(
        labels: np.ndarray, count: int, rng: np.random.Generator
    ) -> np.ndarray:
        if len(labels) <= count:
            return np.arange(len(labels), dtype=np.int64)
        positives = np.flatnonzero(labels > 0)
        negatives = np.flatnonzero(labels <= 0)
        if not len(positives):
            raise ValueError("C64 sampled request has no positive candidate")
        positive = int(rng.choice(positives))
        needed = count - 1
        if len(negatives) >= needed:
            selected_negative = rng.choice(negatives, size=needed, replace=False)
        else:
            remaining = np.asarray(
                [value for value in range(len(labels)) if value != positive], dtype=np.int64
            )
            selected_negative = rng.choice(remaining, size=needed, replace=False)
        positions = np.concatenate((np.asarray([positive]), selected_negative.astype(np.int64)))
        rng.shuffle(positions)
        return positions

    def collate(
        self,
        indices: Sequence[int],
        *,
        label_access: bool,
        history_source: str = "true",
        sampled_candidates: int | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        if history_source not in {"true", "wrong", "empty"}:
            raise ValueError("C64 history source differs")
        if sampled_candidates is not None and (not label_access or rng is None):
            raise PermissionError("C64 candidate sampling requires exposed fit labels and RNG")
        request_indices = [int(value) for value in indices]
        candidate_positions: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        for index in request_indices:
            count = len(self.data.candidates(index))
            labels = self.labels(index) if label_access else np.zeros(count, dtype=np.float32)
            if sampled_candidates is None:
                positions = np.arange(count, dtype=np.int64)
            else:
                positions = self._sample_positions(labels, sampled_candidates, rng)
            candidate_positions.append(positions)
            label_rows.append(labels[positions])
        candidate_counts = [len(value) for value in candidate_positions]
        max_candidates = max(candidate_counts)
        max_history = int(self.config["selection"]["max_history"])
        token_length = int(self.item_token_ids.shape[1])
        batch = len(request_indices)

        query_ids = np.zeros((batch, token_length), dtype=np.int64)
        query_attention = np.zeros((batch, token_length), dtype=bool)
        query_content = np.zeros((batch, token_length), dtype=bool)
        candidate_ids = np.zeros((batch, max_candidates, token_length), dtype=np.int64)
        candidate_attention = np.zeros((batch, max_candidates, token_length), dtype=bool)
        candidate_content = np.zeros((batch, max_candidates, token_length), dtype=bool)
        candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
        history_ids = np.zeros((batch, max_history, token_length), dtype=np.int64)
        history_attention = np.zeros((batch, max_history, token_length), dtype=bool)
        history_content = np.zeros((batch, max_history, token_length), dtype=bool)
        history_event_mask = np.zeros((batch, max_history), dtype=bool)
        base = np.zeros((batch, max_candidates), dtype=np.float32)
        item_only = np.zeros((batch, max_candidates), dtype=np.float32)
        labels = np.zeros((batch, max_candidates), dtype=np.float32)
        repeat_request = np.zeros(batch, dtype=bool)
        candidate_item_ids = np.empty((batch, max_candidates), dtype=object)
        request_ids: list[str] = []

        for row, (index, positions) in enumerate(zip(request_indices, candidate_positions)):
            feature_row = self.feature_position[index]
            query_ids[row] = np.asarray(self.query_token_ids[feature_row], dtype=np.int64)
            query_attention[row] = np.asarray(self.query_attention[feature_row], dtype=bool)
            query_content[row] = np.asarray(self.query_content[feature_row], dtype=bool)
            all_candidates = self.data.candidates(index)
            selected_candidates = all_candidates[positions]
            item_rows = self._item_rows(selected_candidates)
            count = len(positions)
            candidate_ids[row, :count] = np.asarray(self.item_token_ids[item_rows], dtype=np.int64)
            candidate_attention[row, :count] = np.asarray(self.item_attention[item_rows], dtype=bool)
            candidate_content[row, :count] = np.asarray(self.item_content[item_rows], dtype=bool)
            candidate_mask[row, :count] = True
            all_item_ids = self.data.candidate_ids(index)
            candidate_item_ids[row, :count] = [all_item_ids[int(value)] for value in positions]
            start, stop = int(self.feature_offsets[feature_row]), int(self.feature_offsets[feature_row + 1])
            full_base = np.asarray(self.base_scores[start:stop], dtype=np.float32)
            if len(full_base) != len(all_candidates):
                raise ValueError("C64 base score candidate count differs")
            base[row, :count] = full_base[positions]
            labels[row, :count] = label_rows[row]

            true_history, weights = self.data.history(index)
            repeat = all_candidates[:, None] == true_history[None, :] if len(true_history) else np.zeros((len(all_candidates), 0), dtype=bool)
            repeat_request[row] = bool(repeat.any())
            if len(true_history):
                reverse = np.maximum(len(true_history) - np.arange(len(true_history)), 1).astype(np.float32)
                recurrence = weights / np.sqrt(reverse)
                item_component = 3.0 * (repeat.astype(np.float32) * recurrence[None]).sum(-1)
                selected_item = item_component[positions]
                item_std = float(selected_item.std())
                if item_std > 1e-8:
                    selected_item = (selected_item - selected_item.mean()) / item_std
                else:
                    selected_item = np.zeros_like(selected_item)
                item_only[row, :count] = 0.3 * base[row, :count] + 0.7 * selected_item
            else:
                item_only[row, :count] = base[row, :count]

            if history_source == "empty":
                selected_history = np.empty(0, dtype=np.int64)
            else:
                source = index if history_source == "true" else self.donor[index]
                selected_history = self.data.history(source)[0][-max_history:]
            if len(selected_history):
                history_rows = self._item_rows(selected_history)
                length = len(history_rows)
                history_ids[row, :length] = np.asarray(self.item_token_ids[history_rows], dtype=np.int64)
                history_attention[row, :length] = np.asarray(self.item_attention[history_rows], dtype=bool)
                history_content[row, :length] = np.asarray(self.item_content[history_rows], dtype=bool)
                history_event_mask[row, :length] = True
            request_ids.append(self.data.request_ids[index])

        return {
            "request_indices": np.asarray(request_indices, dtype=np.int64),
            "request_ids": request_ids,
            "candidate_item_ids": candidate_item_ids,
            "query_input_ids": query_ids,
            "query_attention_mask": query_attention,
            "query_content_mask": query_content,
            "candidate_input_ids": candidate_ids,
            "candidate_attention_mask": candidate_attention,
            "candidate_content_mask": candidate_content,
            "history_input_ids": history_ids,
            "history_attention_mask": history_attention,
            "history_content_mask": history_content,
            "history_event_mask": history_event_mask,
            "candidate_mask": candidate_mask,
            "base_scores": base,
            "item_only_scores": item_only,
            "repeat_request": repeat_request,
            "query_present": query_attention.any(axis=-1),
            "labels": labels,
        }


TENSOR_NAMES = (
    "query_input_ids",
    "query_attention_mask",
    "query_content_mask",
    "candidate_input_ids",
    "candidate_attention_mask",
    "candidate_content_mask",
    "history_input_ids",
    "history_attention_mask",
    "history_content_mask",
    "history_event_mask",
    "candidate_mask",
    "base_scores",
    "item_only_scores",
    "repeat_request",
    "query_present",
    "labels",
)


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.asarray(batch[name])).to(device)
        for name in TENSOR_NAMES
    }


def iter_training_batches(
    indices: Sequence[int], *, seed: int, batch_size: int
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), batch_size):
        yield order[start : start + batch_size]


def iter_validation_batches(
    store: C64Store,
    indices: Sequence[int],
    *,
    max_requests: int,
    max_sequences: int,
) -> Iterator[np.ndarray]:
    batch: list[int] = []
    sequences = 0
    max_history = int(store.config["selection"]["max_history"])
    for index in indices:
        candidate_count = len(store.data.candidates(int(index)))
        history_count = min(max_history, len(store.data.history(int(index))[0]))
        cost = 1 + candidate_count + history_count
        if batch and (len(batch) >= max_requests or sequences + cost > max_sequences):
            yield np.asarray(batch, dtype=np.int64)
            batch, sequences = [], 0
        batch.append(int(index))
        sequences += cost
    if batch:
        yield np.asarray(batch, dtype=np.int64)


def rankings(request_id: str, item_ids: Sequence[str], scores: np.ndarray) -> list[str]:
    return [
        value.item_id
        for value in sort_candidates(
            request_id,
            [
                ScoredCandidate(str(item_id), float(score))
                for item_id, score in zip(item_ids, scores)
            ],
        )
    ]
