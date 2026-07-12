"""Frozen C24 features and staged compact-label access."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from train.structure import PackedStructure, candidate_key_sha256, read_json, sha256_file


FEATURE_ROLES = (
    "fit",
    "internal_A",
    "structural_single_repeat",
    "structural_nohistory",
    "structural_nonrepeat",
)


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("C24 compact label offsets differ")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("C24 compact label values differ")

    @property
    def positions(self) -> dict[int, int]:
        return {int(index): row for row, index in enumerate(self.request_indices)}

    def rows(self, indices: Sequence[int], counts: Sequence[int]) -> list[np.ndarray]:
        positions = self.positions
        output: list[np.ndarray] = []
        for raw_index, raw_count in zip(indices, counts):
            index, count = int(raw_index), int(raw_count)
            if index not in positions:
                raise PermissionError(f"C24 label unavailable for request {index}")
            row = positions[index]
            start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
            if stop - start != count:
                raise ValueError("C24 compact label candidate count differs")
            output.append(np.asarray(self.values[start:stop], dtype=np.float32).copy())
        return output

    def padded(self, indices: Sequence[int], mask: np.ndarray) -> np.ndarray:
        rows = self.rows(indices, [int(row.sum()) for row in mask])
        output = np.zeros(mask.shape, dtype=np.float32)
        for position, values in enumerate(rows):
            output[position, : len(values)] = values
        return output


def slice_compact_labels(source_root: str | Path, indices: Sequence[int]) -> CompactLabels:
    root = Path(source_root)
    source = CompactLabels(
        request_indices=np.load(root / "fit_request_indices.npy", allow_pickle=False),
        offsets=np.load(root / "fit_label_offsets.npy", allow_pickle=False),
        values=np.load(root / "fit_labels.npy", allow_pickle=False),
    )
    rows = source.rows(indices, [
        int(source.offsets[source.positions[int(index)] + 1] - source.offsets[source.positions[int(index)]])
        for index in indices
    ])
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return CompactLabels(
        request_indices=np.asarray(indices, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        values=np.concatenate(rows).astype(np.float32, copy=False),
    )


def open_original_labels(
    *,
    data: PackedStructure,
    indices: Sequence[int],
    path: str | Path,
    selection_path: str | Path,
    selection_sha256: str,
) -> CompactLabels:
    if sha256_file(selection_path) != selection_sha256:
        raise RuntimeError("C24 selection changed before delayed label access")
    values = np.load(path, mmap_mode="r")
    if len(values) != int(data.candidate_offsets[-1]):
        raise ValueError("C24 original label rows differ")
    rows: list[np.ndarray] = []
    offsets = [0]
    for raw_index in indices:
        index = int(raw_index)
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        row = np.asarray(values[start:stop], dtype=np.float32).copy()
        rows.append(row)
        offsets.append(offsets[-1] + len(row))
    return CompactLabels(
        request_indices=np.asarray(indices, dtype=np.int64),
        offsets=np.asarray(offsets, dtype=np.int64),
        values=np.concatenate(rows),
    )


class FrozenFeatureStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.data = PackedStructure(config["paths"]["packed_train_root"])
        self.selection = read_json(config["paths"]["selection"])
        root = Path(config["paths"]["artifact_root"])
        self.feature_indices = np.load(root / "feature_request_indices.npy", mmap_mode="r")
        self.query = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.item_indices = np.load(root / "item_embedding_indices.npy", mmap_mode="r")
        self.items = np.load(root / "item_embeddings.npy", mmap_mode="r")
        self.score_offsets = np.load(root / "feature_candidate_offsets.npy", mmap_mode="r")
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self.feature_position = {int(index): row for row, index in enumerate(self.feature_indices)}
        if len(self.feature_position) != len(self.feature_indices):
            raise ValueError("C24 feature requests duplicate")
        if not np.all(np.diff(self.item_indices) > 0):
            raise ValueError("C24 item indices not sorted")

    def role_indices(self, role: str) -> list[int]:
        return [int(value) for value in self.selection["roles"][role]["indices"]]

    def candidate_hash(self, indices: Sequence[int]) -> str:
        return candidate_key_sha256(self.data, indices)

    def _states(self, indices: np.ndarray) -> np.ndarray:
        positions = np.searchsorted(self.item_indices, indices)
        if bool((positions >= len(self.item_indices)).any()) or not np.array_equal(
            self.item_indices[positions], indices
        ):
            raise ValueError("C24 item state missing")
        return np.asarray(self.items[positions], dtype=np.float32)

    def collate(
        self, indices: Sequence[int], *, labels: CompactLabels | None = None
    ) -> dict[str, Any]:
        request_indices = np.asarray(indices, dtype=np.int64)
        candidate_counts = np.asarray(
            [int(self.data.candidate_offsets[i + 1] - self.data.candidate_offsets[i]) for i in request_indices]
        )
        history_counts = np.asarray(
            [int(self.data.history_offsets[i + 1] - self.data.history_offsets[i]) for i in request_indices]
        )
        max_candidates = int(candidate_counts.max())
        max_history = max(1, int(history_counts.max()))
        batch = len(request_indices)
        candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_ids = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
        history_indices = np.zeros((batch, max_history), dtype=np.int64)
        history_mask = np.zeros((batch, max_history), dtype=bool)
        event_weights = np.zeros((batch, max_history), dtype=np.float32)
        base = np.zeros((batch, max_candidates), dtype=np.float32)
        request_ids: list[str] = []
        for row, raw_index in enumerate(request_indices):
            index = int(raw_index)
            cs, ce = int(self.data.candidate_offsets[index]), int(self.data.candidate_offsets[index + 1])
            hs, he = int(self.data.history_offsets[index]), int(self.data.history_offsets[index + 1])
            cc, hc = ce - cs, he - hs
            candidate_indices[row, :cc] = self.data.candidate_embedding_indices[cs:ce]
            candidate_ids[row, :cc] = self.data.candidate_item_ids[cs:ce]
            candidate_mask[row, :cc] = True
            if hc:
                history_indices[row, :hc] = self.data.history_embedding_indices[hs:he]
                history_mask[row, :hc] = True
                event_weights[row, :hc] = self.data.history_event_weights[hs:he]
            feature_row = self.feature_position[index]
            start, stop = int(self.score_offsets[feature_row]), int(self.score_offsets[feature_row + 1])
            if stop - start != cc:
                raise ValueError("C24 base score count differs")
            base[row, :cc] = self.base_scores[start:stop]
            request_ids.append(self.data.request_ids[index])
        repeat = (
            candidate_indices[:, :, None] == history_indices[:, None, :]
        ) & candidate_mask[:, :, None] & history_mask[:, None, :]
        reverse = np.maximum(
            history_counts[:, None] - np.arange(max_history)[None, :], 1
        ).astype(np.float32)
        recurrence = event_weights / np.sqrt(reverse)
        item_component = 3.0 * (repeat.astype(np.float32) * recurrence[:, None, :]).sum(-1)
        result: dict[str, Any] = {
            "request_indices": request_indices,
            "request_ids": request_ids,
            "query_numpy": np.asarray(
                self.query[[self.feature_position[int(i)] for i in request_indices]], dtype=np.float32
            ),
            "candidate_item_ids": candidate_ids,
            "candidate_numpy": self._states(candidate_indices),
            "candidate_mask_numpy": candidate_mask,
            "history_mask_numpy": history_mask,
            "repeat_mask_numpy": repeat,
            "event_weights_numpy": event_weights,
            "base_scores_numpy": base,
            "item_component_numpy": item_component,
        }
        if labels is not None:
            result["labels_numpy"] = labels.padded(request_indices, candidate_mask)
        return result


def zscore_numpy(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for row in range(len(values)):
        selected = np.asarray(values[row, mask[row]], dtype=np.float64)
        scale = float(selected.std()) if len(selected) else 0.0
        if scale > 1e-8:
            output[row, mask[row]] = ((selected - selected.mean()) / scale).astype(np.float32)
    return output


def to_device(batch: Mapping[str, Any], device: torch.device, *, beta: float) -> dict[str, torch.Tensor]:
    mask = batch["candidate_mask_numpy"]
    base_z = zscore_numpy(batch["base_scores_numpy"], mask)
    item_z = zscore_numpy(batch["item_component_numpy"], mask)
    output = {
        "query": torch.from_numpy(batch["query_numpy"]).to(device),
        "candidates": torch.from_numpy(batch["candidate_numpy"]).to(device),
        "candidate_mask": torch.from_numpy(mask).to(device),
        "history_mask": torch.from_numpy(batch["history_mask_numpy"]).to(device),
        "repeat_mask": torch.from_numpy(batch["repeat_mask_numpy"]).to(device),
        "event_weights": torch.from_numpy(batch["event_weights_numpy"]).to(device),
        "base_scores": torch.from_numpy(batch["base_scores_numpy"]).to(device),
        "item_only_scores": torch.from_numpy(beta * base_z + (1.0 - beta) * item_z).to(device),
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
    max_padded_candidates: int,
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64)
    if shuffle:
        order = order[np.random.default_rng(seed).permutation(len(order))]
    current: list[int] = []
    maximum = 0
    for raw_index in order:
        index = int(raw_index)
        count = int(data.candidate_offsets[index + 1] - data.candidate_offsets[index])
        prospective = max(maximum, count)
        if current and (
            len(current) + 1 > max_requests
            or (len(current) + 1) * prospective > max_padded_candidates
        ):
            yield np.asarray(current, dtype=np.int64)
            current, maximum = [], 0
        current.append(index)
        maximum = max(maximum, count)
    if current:
        yield np.asarray(current, dtype=np.int64)


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()
