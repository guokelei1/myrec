"""Frozen token-level C28 features and staged label access."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from train.structure import PackedStructure, candidate_key_sha256, read_json, sha256_file


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1 or int(self.offsets[-1]) != len(
            self.values
        ):
            raise ValueError("C28 compact labels differ")

    @property
    def positions(self) -> dict[int, int]:
        return {int(index): row for row, index in enumerate(self.request_indices)}

    def rows(self, indices: Sequence[int], counts: Sequence[int]) -> list[np.ndarray]:
        positions = self.positions
        output: list[np.ndarray] = []
        for raw_index, raw_count in zip(indices, counts):
            index, count = int(raw_index), int(raw_count)
            if index not in positions:
                raise PermissionError(f"C28 label unavailable for request {index}")
            row = positions[index]
            start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
            if stop - start != count:
                raise ValueError("C28 label candidate count differs")
            output.append(np.asarray(self.values[start:stop], dtype=np.float32).copy())
        return output

    def padded(self, indices: Sequence[int], mask: np.ndarray) -> np.ndarray:
        rows = self.rows(indices, [int(row.sum()) for row in mask])
        output = np.zeros(mask.shape, dtype=np.float32)
        for position, values in enumerate(rows):
            output[position, : len(values)] = values
        return output


def copy_compact_labels(source_root: str | Path, indices: Sequence[int]) -> CompactLabels:
    root = Path(source_root)
    source = CompactLabels(
        request_indices=np.load(root / "fit_request_indices.npy", allow_pickle=False),
        offsets=np.load(root / "fit_label_offsets.npy", allow_pickle=False),
        values=np.load(root / "fit_labels.npy", allow_pickle=False),
    )
    rows = source.rows(
        indices,
        [
            int(source.offsets[source.positions[int(index)] + 1] - source.offsets[source.positions[int(index)]])
            for index in indices
        ],
    )
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
        raise RuntimeError("C28 selection changed before label access")
    values = np.load(path, mmap_mode="r")
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
        values=np.concatenate(rows).astype(np.float32, copy=False),
    )


class FrozenTokenStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        self.data = PackedStructure(config["paths"]["packed_train_root"])
        self.selection = read_json(config["paths"]["selection"])
        feature_root = Path(config["paths"]["artifact_root"])
        token_root = feature_root
        self.feature_indices = np.load(
            feature_root / "feature_request_indices.npy", mmap_mode="r"
        )
        self.score_offsets = np.load(
            feature_root / "feature_candidate_offsets.npy", mmap_mode="r"
        )
        self.base_scores = np.load(feature_root / "base_scores.npy", mmap_mode="r")
        self.query_ids = np.load(feature_root / "query_token_ids.npy", mmap_mode="r")
        self.query_attention = np.load(
            feature_root / "query_attention_mask.npy", mmap_mode="r"
        )
        self.query_content = np.load(
            feature_root / "query_content_mask.npy", mmap_mode="r"
        )
        self.item_indices = np.load(token_root / "item_embedding_indices.npy", mmap_mode="r")
        self.item_ids = np.load(token_root / "item_token_ids.npy", mmap_mode="r")
        self.item_attention = np.load(token_root / "item_attention_mask.npy", mmap_mode="r")
        self.item_content = np.load(token_root / "item_content_mask.npy", mmap_mode="r")
        self.word_embeddings = np.load(token_root / "word_embeddings.npy", mmap_mode="r")
        self.feature_position = {int(index): row for row, index in enumerate(self.feature_indices)}
        if len(self.feature_position) != len(self.feature_indices) or not np.all(
            np.diff(self.item_indices) > 0
        ):
            raise ValueError("C28 feature/item index structure differs")
        self.donor_by_index: dict[int, int] = {}
        for role, row in self.selection["wrong_history_donors"].items():
            for recipient, donor in zip(self.selection["roles"][role]["indices"], row["indices"]):
                self.donor_by_index[int(recipient)] = int(donor)

    def role_indices(self, role: str) -> list[int]:
        return [int(value) for value in self.selection["roles"][role]["indices"]]

    def candidate_hash(self, indices: Sequence[int]) -> str:
        return candidate_key_sha256(self.data, indices)

    def _token_rows(self, embedding_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positions = np.searchsorted(self.item_indices, embedding_indices)
        if bool((positions >= len(self.item_indices)).any()) or not np.array_equal(
            self.item_indices[positions], embedding_indices
        ):
            raise ValueError("C28 item token row missing")
        return (
            np.asarray(self.item_ids[positions], dtype=np.int64),
            np.asarray(self.item_attention[positions], dtype=bool),
            np.asarray(self.item_content[positions], dtype=bool),
        )

    def collate(
        self,
        indices: Sequence[int],
        *,
        labels: CompactLabels | None = None,
        history_source: str = "true",
    ) -> dict[str, Any]:
        if history_source not in {"true", "wrong"}:
            raise ValueError("C28 history source differs")
        request_indices = np.asarray(indices, dtype=np.int64)
        source_indices = np.asarray(
            [
                int(index) if history_source == "true" else self.donor_by_index[int(index)]
                for index in request_indices
            ],
            dtype=np.int64,
        )
        candidate_counts = np.asarray(
            [int(self.data.candidate_offsets[i + 1] - self.data.candidate_offsets[i]) for i in request_indices]
        )
        history_counts = np.asarray(
            [int(self.data.history_offsets[i + 1] - self.data.history_offsets[i]) for i in source_indices]
        )
        batch = len(request_indices)
        max_candidates, max_history = int(candidate_counts.max()), max(1, int(history_counts.max()))
        candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_item_ids = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
        history_indices = np.zeros((batch, max_history), dtype=np.int64)
        history_mask = np.zeros((batch, max_history), dtype=bool)
        event_weights = np.zeros((batch, max_history), dtype=np.float32)
        base = np.zeros((batch, max_candidates), dtype=np.float32)
        request_ids: list[str] = []
        for row, (raw_index, raw_source) in enumerate(zip(request_indices, source_indices)):
            index, source = int(raw_index), int(raw_source)
            cs, ce = int(self.data.candidate_offsets[index]), int(self.data.candidate_offsets[index + 1])
            hs, he = int(self.data.history_offsets[source]), int(self.data.history_offsets[source + 1])
            cc, hc = ce - cs, he - hs
            candidate_indices[row, :cc] = self.data.candidate_embedding_indices[cs:ce]
            candidate_item_ids[row, :cc] = self.data.candidate_item_ids[cs:ce]
            candidate_mask[row, :cc] = True
            if hc:
                history_indices[row, :hc] = self.data.history_embedding_indices[hs:he]
                history_mask[row, :hc] = True
                event_weights[row, :hc] = self.data.history_event_weights[hs:he]
            feature_row = self.feature_position[index]
            start, stop = int(self.score_offsets[feature_row]), int(self.score_offsets[feature_row + 1])
            base[row, :cc] = self.base_scores[start:stop]
            request_ids.append(self.data.request_ids[index])
        candidate_tokens = self._token_rows(candidate_indices)
        history_tokens = self._token_rows(history_indices)
        item_length = max(
            1,
            int(candidate_tokens[1].sum(axis=-1).max()),
            int(history_tokens[1].sum(axis=-1).max()),
        )
        candidate_tokens = tuple(value[..., :item_length] for value in candidate_tokens)
        history_tokens = tuple(value[..., :item_length] for value in history_tokens)
        repeat = (
            candidate_indices[:, :, None] == history_indices[:, None, :]
        ) & candidate_mask[:, :, None] & history_mask[:, None, :]
        reverse = np.maximum(history_counts[:, None] - np.arange(max_history)[None, :], 1).astype(
            np.float32
        )
        recurrence = event_weights / np.sqrt(reverse)
        item_component = 3.0 * (repeat.astype(np.float32) * recurrence[:, None, :]).sum(-1)
        query_rows = [self.feature_position[int(index)] for index in request_indices]
        query_ids = np.asarray(self.query_ids[query_rows], dtype=np.int64)
        query_attention = np.asarray(self.query_attention[query_rows], dtype=bool)
        query_content = np.asarray(self.query_content[query_rows], dtype=bool)
        query_length = max(1, int(query_attention.sum(axis=-1).max()))
        result: dict[str, Any] = {
            "request_indices": request_indices,
            "request_ids": request_ids,
            "candidate_item_ids": candidate_item_ids,
            "query_token_ids_numpy": query_ids[..., :query_length],
            "query_attention_mask_numpy": query_attention[..., :query_length],
            "query_content_mask_numpy": query_content[..., :query_length],
            "candidate_token_ids_numpy": candidate_tokens[0],
            "candidate_attention_mask_numpy": candidate_tokens[1],
            "candidate_content_mask_numpy": candidate_tokens[2],
            "history_token_ids_numpy": history_tokens[0],
            "history_attention_mask_numpy": history_tokens[1],
            "history_content_mask_numpy": history_tokens[2],
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
    candidate_mask = batch["candidate_mask_numpy"]
    base_z = zscore_numpy(batch["base_scores_numpy"], candidate_mask)
    item_z = zscore_numpy(batch["item_component_numpy"], candidate_mask)
    names = (
        "query_token_ids",
        "query_attention_mask",
        "query_content_mask",
        "candidate_token_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "history_token_ids",
        "history_attention_mask",
        "history_content_mask",
        "candidate_mask",
        "history_mask",
        "repeat_mask",
        "event_weights",
        "base_scores",
    )
    output = {
        name: torch.from_numpy(batch[f"{name}_numpy"]).to(device)
        for name in names
    }
    output["base_scores"] = torch.from_numpy(base_z).to(device)
    output["item_only_scores"] = torch.from_numpy(beta * base_z + (1.0 - beta) * item_z).to(
        device
    )
    if "labels_numpy" in batch:
        output["labels"] = torch.from_numpy(batch["labels_numpy"]).to(device)
    return output


def iter_batches(
    store: FrozenTokenStore,
    indices: Sequence[int],
    *,
    seed: int,
    shuffle: bool,
    max_requests: int,
    max_bridge_cells: int,
    max_pair_cells: int,
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64)
    if shuffle:
        order = order[np.random.default_rng(seed).permutation(len(order))]
    current: list[int] = []
    max_candidates = max_history = max_query = 0
    for raw_index in order:
        index = int(raw_index)
        candidates = int(store.data.candidate_offsets[index + 1] - store.data.candidate_offsets[index])
        history = max(1, int(store.data.history_offsets[index + 1] - store.data.history_offsets[index]))
        query_row = store.feature_position[index]
        query = max(1, int(np.asarray(store.query_attention[query_row]).sum()))
        next_c, next_h, next_q = max(max_candidates, candidates), max(max_history, history), max(
            max_query, query
        )
        if current and (
            len(current) + 1 > max_requests
            or (len(current) + 1) * next_c * next_h * next_q > max_bridge_cells
            or (len(current) + 1) * next_c * next_c > max_pair_cells
        ):
            yield np.asarray(current, dtype=np.int64)
            current = [index]
            max_candidates, max_history, max_query = candidates, history, query
            continue
        current.append(index)
        max_candidates, max_history, max_query = next_c, next_h, next_q
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
