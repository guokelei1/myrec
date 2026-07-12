"""C23 label-isolated frozen features, compact labels, and batching."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from train.structure import PackedStructure, candidate_key_sha256, read_json, sha256_file


FEATURE_ROLES = ("fit", "internal_A", "structural_nohistory", "structural_nonrepeat")


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("compact label offsets differ")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("compact label values differ")

    @property
    def positions(self) -> dict[int, int]:
        return {int(index): row for row, index in enumerate(self.request_indices)}

    def padded(
        self, request_indices: Sequence[int], candidate_mask: np.ndarray
    ) -> np.ndarray:
        output = np.zeros(candidate_mask.shape, dtype=np.float32)
        positions = self.positions
        for row, raw_index in enumerate(request_indices):
            index = int(raw_index)
            if index not in positions:
                raise PermissionError(f"C23 labels unavailable for request {index}")
            label_row = positions[index]
            start, stop = int(self.offsets[label_row]), int(self.offsets[label_row + 1])
            count = int(candidate_mask[row].sum())
            if stop - start != count:
                raise ValueError("compact label candidate count changed")
            output[row, :count] = self.values[start:stop]
        return output

    def rows(self, request_indices: Sequence[int], counts: Sequence[int]) -> list[np.ndarray]:
        if len(request_indices) != len(counts):
            raise ValueError("compact label row/count length differs")
        positions = self.positions
        output: list[np.ndarray] = []
        for raw_index, raw_count in zip(request_indices, counts):
            index, count = int(raw_index), int(raw_count)
            if index not in positions:
                raise PermissionError(f"C23 labels unavailable for request {index}")
            row = positions[index]
            start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
            if stop - start != count:
                raise ValueError("compact label candidate count changed")
            output.append(np.asarray(self.values[start:stop], dtype=np.float32).copy())
        return output


def open_original_selected_labels(
    *,
    data: PackedStructure,
    indices: Sequence[int],
    label_path: str | Path,
    selection_sha256: str,
    selection_path: str | Path,
) -> CompactLabels:
    """Open values only after the immutable selection is reverified."""

    if sha256_file(selection_path) != selection_sha256:
        raise RuntimeError("C23 selection changed before label opening")
    labels = np.load(label_path, mmap_mode="r")
    if len(labels) != int(data.candidate_offsets[-1]):
        raise ValueError("original train label row count changed")
    offsets = [0]
    values: list[np.ndarray] = []
    ordered = np.asarray([int(value) for value in indices], dtype=np.int64)
    for index in ordered:
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        values.append(np.asarray(labels[start:stop], dtype=np.float32).copy())
        offsets.append(offsets[-1] + stop - start)
    return CompactLabels(
        request_indices=ordered,
        offsets=np.asarray(offsets, dtype=np.int64),
        values=np.concatenate(values) if values else np.empty(0, dtype=np.float32),
    )


class FrozenFeatureStore:
    """Memory-mapped D2p states with no implicit access to train labels."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        paths = config["paths"]
        self.config = config
        self.data = PackedStructure.load(paths["packed_train_root"])
        self.selection = read_json(paths["selection"])
        root = Path(paths["artifact_root"])
        self.feature_request_indices = np.load(
            root / "feature_request_indices.npy", mmap_mode="r"
        )
        self.query_embeddings = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.item_embedding_indices = np.load(
            root / "item_embedding_indices.npy", mmap_mode="r"
        )
        self.item_embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r")
        self.feature_candidate_offsets = np.load(
            root / "feature_candidate_offsets.npy", mmap_mode="r"
        )
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self._feature_position = {
            int(index): row for row, index in enumerate(self.feature_request_indices)
        }
        if len(self._feature_position) != len(self.feature_request_indices):
            raise ValueError("C23 feature requests are not unique")
        if not np.all(np.diff(self.item_embedding_indices) > 0):
            raise ValueError("C23 item-state indices must be strictly sorted")
        if len(self.item_embedding_indices) != len(self.item_embeddings):
            raise ValueError("C23 item-state rows differ")
        if len(self.query_embeddings) != len(self.feature_request_indices):
            raise ValueError("C23 query-state rows differ")
        if len(self.feature_candidate_offsets) != len(self.feature_request_indices) + 1:
            raise ValueError("C23 feature score offsets differ")
        if int(self.feature_candidate_offsets[-1]) != len(self.base_scores):
            raise ValueError("C23 base score rows differ")

    def role_indices(self, role: str) -> list[int]:
        return [int(value) for value in self.selection["roles"][role]["indices"]]

    def candidate_key_sha256(self, indices: Sequence[int]) -> str:
        return candidate_key_sha256(self.data, indices)

    def _item_states(self, indices: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(self.item_embedding_indices, indices)
        if bool((positions >= len(self.item_embedding_indices)).any()):
            raise ValueError("C23 item index lies outside materialized states")
        recovered = self.item_embedding_indices[positions]
        if not np.array_equal(recovered, indices):
            raise ValueError("C23 item state is missing")
        return np.asarray(self.item_embeddings[positions], dtype=np.float32)

    def _feature_scores(self, index: int) -> np.ndarray:
        if index not in self._feature_position:
            raise PermissionError(f"C23 features unavailable for request {index}")
        row = self._feature_position[index]
        start, stop = (
            int(self.feature_candidate_offsets[row]),
            int(self.feature_candidate_offsets[row + 1]),
        )
        return np.asarray(self.base_scores[start:stop], dtype=np.float32)

    def collate(
        self,
        request_indices: Sequence[int],
        *,
        labels: CompactLabels | None = None,
    ) -> dict[str, Any]:
        indices = np.asarray([int(value) for value in request_indices], dtype=np.int64)
        if len(indices) == 0:
            raise ValueError("empty C23 batch")
        candidate_counts = np.asarray(
            [
                int(self.data.candidate_offsets[i + 1] - self.data.candidate_offsets[i])
                for i in indices
            ],
            dtype=np.int64,
        )
        history_counts = np.asarray(
            [
                int(self.data.history_offsets[i + 1] - self.data.history_offsets[i])
                for i in indices
            ],
            dtype=np.int64,
        )
        max_candidates = int(candidate_counts.max())
        max_history = max(1, int(history_counts.max()))
        batch = len(indices)
        candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_item_ids = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
        history_indices = np.zeros((batch, max_history), dtype=np.int64)
        history_mask = np.zeros((batch, max_history), dtype=bool)
        event_weights = np.zeros((batch, max_history), dtype=np.float32)
        base_scores = np.zeros((batch, max_candidates), dtype=np.float32)
        request_ids: list[str] = []
        for row, index in enumerate(indices):
            cs, ce = int(self.data.candidate_offsets[index]), int(
                self.data.candidate_offsets[index + 1]
            )
            hs, he = int(self.data.history_offsets[index]), int(
                self.data.history_offsets[index + 1]
            )
            cc, hc = ce - cs, he - hs
            candidate_indices[row, :cc] = self.data.candidate_embedding_indices[cs:ce]
            candidate_item_ids[row, :cc] = self.data.candidate_item_ids[cs:ce]
            candidate_mask[row, :cc] = True
            if hc:
                history_indices[row, :hc] = self.data.history_embedding_indices[hs:he]
                history_mask[row, :hc] = True
                event_weights[row, :hc] = np.asarray(
                    self.data.history_event_weights[hs:he], dtype=np.float32
                )
            scores = self._feature_scores(int(index))
            if len(scores) != cc:
                raise ValueError("C23 base score candidate count differs")
            base_scores[row, :cc] = scores
            request_ids.append(self.data.request_ids[int(index)])
        repeat_mask = (
            candidate_indices[:, :, None] == history_indices[:, None, :]
        ) & candidate_mask[:, :, None] & history_mask[:, None, :]
        reverse = history_counts[:, None] - np.arange(max_history)[None, :]
        reverse = np.maximum(reverse, 1).astype(np.float32)
        recurrence_weights = event_weights / np.sqrt(reverse)
        item_component = 3.0 * (
            repeat_mask.astype(np.float32) * recurrence_weights[:, None, :]
        ).sum(axis=-1)

        result: dict[str, Any] = {
            "request_indices": indices,
            "request_ids": request_ids,
            "query_numpy": np.asarray(
                self.query_embeddings[[self._feature_position[int(i)] for i in indices]],
                dtype=np.float32,
            ),
            "candidate_indices": candidate_indices,
            "candidate_item_ids": candidate_item_ids,
            "candidate_numpy": self._item_states(candidate_indices),
            "candidate_mask_numpy": candidate_mask,
            "history_indices": history_indices,
            "history_numpy": self._item_states(history_indices),
            "history_mask_numpy": history_mask,
            "event_weights_numpy": event_weights,
            "repeat_mask_numpy": repeat_mask,
            "base_scores_numpy": base_scores,
            "item_component_numpy": item_component,
        }
        if labels is not None:
            result["labels_numpy"] = labels.padded(indices, candidate_mask)
        return result


def masked_zscore_numpy(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for row in range(len(values)):
        selected = np.asarray(values[row, mask[row]], dtype=np.float64)
        if len(selected) == 0:
            continue
        scale = float(selected.std())
        if scale > 1e-8:
            output[row, mask[row]] = ((selected - selected.mean()) / scale).astype(np.float32)
    return output


def to_device(batch: Mapping[str, Any], device: torch.device, *, beta: float) -> dict[str, Any]:
    candidate_mask = torch.from_numpy(batch["candidate_mask_numpy"]).to(device)
    base = torch.from_numpy(batch["base_scores_numpy"]).to(device)
    item_component = torch.from_numpy(batch["item_component_numpy"]).to(device)
    base_z = torch.from_numpy(
        masked_zscore_numpy(batch["base_scores_numpy"], batch["candidate_mask_numpy"])
    ).to(device)
    item_z = torch.from_numpy(
        masked_zscore_numpy(batch["item_component_numpy"], batch["candidate_mask_numpy"])
    ).to(device)
    item_only = beta * base_z + (1.0 - beta) * item_z
    output: dict[str, Any] = {
        "query": torch.from_numpy(batch["query_numpy"]).to(device),
        "candidates": torch.from_numpy(batch["candidate_numpy"]).to(device),
        "candidate_mask": candidate_mask,
        "history": torch.from_numpy(batch["history_numpy"]).to(device),
        "history_mask": torch.from_numpy(batch["history_mask_numpy"]).to(device),
        "event_weights": torch.from_numpy(batch["event_weights_numpy"]).to(device),
        "repeat_mask": torch.from_numpy(batch["repeat_mask_numpy"]).to(device),
        "base_scores": base,
        "item_only_scores": item_only,
    }
    if "labels_numpy" in batch:
        output["labels"] = torch.from_numpy(batch["labels_numpy"]).to(device)
    return output


def iter_batches(
    data: PackedStructure,
    indices: Sequence[int],
    *,
    seed: int,
    shuffle: bool,
    max_requests: int,
    max_candidate_sequences: int,
    max_sequence_tokens: int,
) -> Iterator[np.ndarray]:
    order = np.asarray([int(value) for value in indices], dtype=np.int64)
    if shuffle:
        order = order[np.random.default_rng(seed).permutation(len(order))]
    current: list[int] = []
    max_candidates = 0
    max_history = 1
    for raw_index in order:
        index = int(raw_index)
        candidates = int(data.candidate_offsets[index + 1] - data.candidate_offsets[index])
        history = max(1, int(data.history_offsets[index + 1] - data.history_offsets[index]))
        prospective_requests = len(current) + 1
        prospective_candidates = max(max_candidates, candidates)
        prospective_history = max(max_history, history)
        candidate_sequences = prospective_requests * prospective_candidates
        sequence_tokens = candidate_sequences * (prospective_history + 2)
        if current and (
            prospective_requests > max_requests
            or candidate_sequences > max_candidate_sequences
            or sequence_tokens > max_sequence_tokens
        ):
            yield np.asarray(current, dtype=np.int64)
            current = []
            max_candidates = 0
            max_history = 1
        current.append(index)
        max_candidates = max(max_candidates, candidates)
        max_history = max(max_history, history)
    if current:
        yield np.asarray(current, dtype=np.int64)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
