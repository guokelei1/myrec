"""Frozen C29 token features, causal authentication, and staged label access."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from train.structure import PackedStructure, candidate_key_sha256, read_json, sha256_file


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("C29 compact-label offsets differ")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("C29 compact-label values differ")
        if len(set(int(value) for value in self.request_indices)) != len(self.request_indices):
            raise ValueError("C29 compact-label requests overlap")

    @property
    def positions(self) -> dict[int, int]:
        return {int(index): row for row, index in enumerate(self.request_indices)}

    def rows(self, indices: Sequence[int], counts: Sequence[int]) -> list[np.ndarray]:
        positions = self.positions
        output: list[np.ndarray] = []
        for raw_index, raw_count in zip(indices, counts):
            index, count = int(raw_index), int(raw_count)
            if index not in positions:
                raise PermissionError(f"C29 label unavailable for request {index}")
            row = positions[index]
            start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
            if stop - start != count:
                raise ValueError("C29 label candidate count differs")
            output.append(np.asarray(self.values[start:stop], dtype=np.float32).copy())
        return output


def open_original_labels(
    *,
    data: PackedStructure,
    indices: Sequence[int],
    path: str | Path,
    selection_path: str | Path,
    selection_sha256: str,
) -> CompactLabels:
    if sha256_file(selection_path) != selection_sha256:
        raise RuntimeError("C29 selection changed before label access")
    values = np.load(path, mmap_mode="r")
    rows: list[np.ndarray] = []
    offsets = [0]
    for raw_index in indices:
        index = int(raw_index)
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        row = np.asarray(values[start:stop], dtype=np.float32).copy()
        if not np.isfinite(row).all():
            raise ValueError("C29 nonfinite label row")
        rows.append(row)
        offsets.append(offsets[-1] + len(row))
    return CompactLabels(
        request_indices=np.asarray(indices, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        values=np.concatenate(rows).astype(np.float32, copy=False),
    )


def zscore_row(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    scale = float(np.asarray(values, dtype=np.float64).std()) if len(values) else 0.0
    if scale <= 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - float(np.asarray(values, dtype=np.float64).mean())) / scale).astype(
        np.float32
    )


class FrozenMediationStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.data = PackedStructure(config["paths"]["packed_train_root"])
        self.selection = read_json(config["paths"]["selection"])
        root = Path(config["paths"]["artifact_root"])
        self.feature_indices = np.load(root / "feature_request_indices.npy", mmap_mode="r")
        self.score_offsets = np.load(root / "feature_candidate_offsets.npy", mmap_mode="r")
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self.query_ids = np.load(root / "query_token_ids.npy", mmap_mode="r")
        self.query_attention = np.load(root / "query_attention_mask.npy", mmap_mode="r")
        self.query_content = np.load(root / "query_content_mask.npy", mmap_mode="r")
        self.item_indices = np.load(root / "item_embedding_indices.npy", mmap_mode="r")
        self.item_ids = np.load(root / "item_token_ids.npy", mmap_mode="r")
        self.item_attention = np.load(root / "item_attention_mask.npy", mmap_mode="r")
        self.item_content = np.load(root / "item_content_mask.npy", mmap_mode="r")
        self.auth_request_indices = np.load(
            root / "authentication_request_indices.npy", mmap_mode="r"
        )
        self.auth_true_offsets = np.load(root / "auth_true_offsets.npy", mmap_mode="r")
        self.auth_true_items = np.load(root / "auth_true_items.npy", mmap_mode="r")
        self.auth_wrong_offsets = np.load(root / "auth_wrong_offsets.npy", mmap_mode="r")
        self.auth_wrong_items = np.load(root / "auth_wrong_items.npy", mmap_mode="r")
        self.auth_profile_sizes = np.load(root / "auth_profile_sizes.npy", mmap_mode="r")
        self.feature_position = {int(index): row for row, index in enumerate(self.feature_indices)}
        self.auth_position = {
            int(index): row for row, index in enumerate(self.auth_request_indices)
        }
        if len(self.feature_position) != len(self.feature_indices):
            raise ValueError("C29 feature request structure differs")
        if set(self.feature_position) != set(self.auth_position):
            raise ValueError("C29 feature/authentication request structure differs")
        if not np.all(np.diff(self.item_indices) > 0):
            raise ValueError("C29 item-token indices are not strictly ordered")
        self.donor_by_index: dict[int, int] = {}
        for role, row in self.selection["wrong_history_donors"].items():
            for recipient, donor in zip(self.selection["roles"][role]["indices"], row["indices"]):
                self.donor_by_index[int(recipient)] = int(donor)

    def role_indices(self, role: str) -> list[int]:
        return [int(value) for value in self.selection["roles"][role]["indices"]]

    def candidate_hash(self, indices: Sequence[int]) -> str:
        return candidate_key_sha256(self.data, indices)

    def candidate_count(self, index: int) -> int:
        return int(self.data.candidate_offsets[index + 1] - self.data.candidate_offsets[index])

    def candidate_embedding_indices(self, index: int) -> np.ndarray:
        return self.data.candidate_indices(index).astype(np.int64, copy=False)

    def candidate_item_ids(self, index: int) -> np.ndarray:
        start, stop = int(self.data.candidate_offsets[index]), int(
            self.data.candidate_offsets[index + 1]
        )
        return np.asarray(self.data.candidate_item_ids[start:stop]).copy()

    def base_row(self, index: int) -> np.ndarray:
        feature = self.feature_position[int(index)]
        start, stop = int(self.score_offsets[feature]), int(self.score_offsets[feature + 1])
        if stop - start != self.candidate_count(int(index)):
            raise ValueError("C29 base-score candidate count differs")
        return zscore_row(np.asarray(self.base_scores[start:stop], dtype=np.float32))

    def item_only_row(self, index: int) -> np.ndarray:
        candidates = self.candidate_embedding_indices(index)
        history = self.data.history_indices(index).astype(np.int64, copy=False)
        weights_start, weights_stop = int(self.data.history_offsets[index]), int(
            self.data.history_offsets[index + 1]
        )
        weights = np.asarray(
            self.data.history_event_weights[weights_start:weights_stop], dtype=np.float32
        )
        if len(history):
            reverse = np.maximum(len(history) - np.arange(len(history)), 1).astype(np.float32)
            recurrence = weights / np.sqrt(reverse)
            component = 3.0 * (
                (candidates[:, None] == history[None, :]).astype(np.float32)
                * recurrence[None, :]
            ).sum(axis=1)
        else:
            component = np.zeros(len(candidates), dtype=np.float32)
        beta = float(self.config["base"]["item_only_beta"])
        return beta * self.base_row(index) + (1.0 - beta) * zscore_row(component)

    def query_tokens(self, index: int, maximum: int) -> list[int]:
        row = self.feature_position[int(index)]
        return [
            int(value)
            for value in np.asarray(self.query_ids[row])[np.asarray(self.query_content[row])]
        ][:maximum]

    def item_tokens(self, item_index: int, maximum: int) -> list[int]:
        position = int(np.searchsorted(self.item_indices, int(item_index)))
        if position >= len(self.item_indices) or int(self.item_indices[position]) != int(item_index):
            raise ValueError(f"C29 item token missing: {item_index}")
        return [
            int(value)
            for value in np.asarray(self.item_ids[position])[
                np.asarray(self.item_content[position], dtype=bool)
            ]
        ][:maximum]

    def authenticated_history(self, index: int, source: str) -> np.ndarray:
        row = self.auth_position[int(index)]
        if source == "true":
            start, stop = int(self.auth_true_offsets[row]), int(self.auth_true_offsets[row + 1])
            return np.asarray(self.auth_true_items[start:stop], dtype=np.int64)
        if source == "wrong":
            start, stop = int(self.auth_wrong_offsets[row]), int(self.auth_wrong_offsets[row + 1])
            return np.asarray(self.auth_wrong_items[start:stop], dtype=np.int64)
        if source == "none":
            return np.empty(0, dtype=np.int64)
        raise ValueError(f"unexpected C29 history source: {source}")

    def raw_history(self, index: int, source: str) -> np.ndarray:
        if source == "none":
            return np.empty(0, dtype=np.int64)
        source_index = int(index) if source == "true" else self.donor_by_index[int(index)]
        return self.data.history_indices(source_index).astype(np.int64, copy=False)

    def has_repeat(self, index: int) -> bool:
        return bool(
            set(int(value) for value in self.candidate_embedding_indices(index))
            & set(int(value) for value in self.data.history_indices(index))
        )


class MediationSequenceBuilder:
    def __init__(
        self,
        store: FrozenMediationStore,
        *,
        cls_token_id: int,
        sep_token_id: int,
        pad_token_id: int,
    ) -> None:
        self.store = store
        self.cls = int(cls_token_id)
        self.sep = int(sep_token_id)
        self.pad = int(pad_token_id)
        sequence = store.config["sequence"]
        self.max_length = int(sequence["max_length"])
        self.max_query = int(sequence["max_query_content"])
        self.max_candidate = int(sequence["max_candidate_content"])
        self.max_history_items = int(sequence["max_history_items"])
        self.max_history_content = int(sequence["max_history_content"])

    def pair(
        self,
        request_index: int,
        candidate_position: int,
        *,
        history_source: str,
        authenticated: bool,
    ) -> tuple[list[int], list[int]]:
        query = self.store.query_tokens(request_index, self.max_query)
        candidates = self.store.candidate_embedding_indices(request_index)
        candidate = int(candidates[int(candidate_position)])
        candidate_tokens = self.store.item_tokens(candidate, self.max_candidate)
        history = (
            self.store.authenticated_history(request_index, history_source)
            if authenticated
            else self.store.raw_history(request_index, history_source)
        )[-self.max_history_items :]
        prefix = [self.cls, *query, self.sep, *candidate_tokens, self.sep]
        factual = list(prefix)
        for item in history:
            factual.extend(self.store.item_tokens(int(item), self.max_history_content))
            factual.append(self.sep)
        return factual[: self.max_length], prefix[: self.max_length]

    def batch(
        self,
        examples: Sequence[tuple[int, int]],
        *,
        history_source: str,
        authenticated: bool,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rows: list[list[int]] = []
        for request_index, candidate_position in examples:
            factual, null = self.pair(
                int(request_index),
                int(candidate_position),
                history_source=history_source,
                authenticated=authenticated,
            )
            rows.extend((factual, null))
        if not rows:
            raise ValueError("empty C29 sequence batch")
        length = max(len(row) for row in rows)
        ids = np.full((len(rows), length), self.pad, dtype=np.int64)
        attention = np.zeros((len(rows), length), dtype=np.int64)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = row
            attention[index, : len(row)] = 1
        return torch.from_numpy(ids).to(device), torch.from_numpy(attention).to(device)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
