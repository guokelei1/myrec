"""Read-only C21 access to C06 fit features and compact fit labels.

The module has no path or argument for the original packed train label array.
Compact labels are opened only by ``FrozenFitData.open_labels`` after the C21
proposal lock has been verified by the formal runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_request_ids(path: Path) -> list[str]:
    output: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            output.append(str(row["request_id"]))
    if len(output) != len(set(output)):
        raise ValueError("packed request IDs are not unique")
    return output


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.request_indices.ndim != 1 or self.offsets.ndim != 1 or self.values.ndim != 1:
            raise ValueError("compact label arrays must be one-dimensional")
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("compact label offset mismatch")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("compact label row mismatch")

    @property
    def positions(self) -> dict[int, int]:
        return {int(index): position for position, index in enumerate(self.request_indices)}

    def rows(self, request_indices: Sequence[int], candidate_counts: Sequence[int]) -> list[np.ndarray]:
        positions = self.positions
        output: list[np.ndarray] = []
        for raw_index, expected_count in zip(request_indices, candidate_counts):
            index = int(raw_index)
            if index not in positions:
                raise PermissionError(f"C21 label request is outside C06 fit: {index}")
            position = positions[index]
            start = int(self.offsets[position])
            stop = int(self.offsets[position + 1])
            row = np.asarray(self.values[start:stop], dtype=np.float32).copy()
            if len(row) != int(expected_count):
                raise ValueError("compact label/candidate count mismatch")
            output.append(row)
        return output

    def padded(self, request_indices: Sequence[int], candidate_mask: np.ndarray) -> np.ndarray:
        counts = candidate_mask.sum(axis=1).astype(np.int64)
        rows = self.rows(request_indices, counts)
        output = np.zeros(candidate_mask.shape, dtype=np.float32)
        for row_index, row in enumerate(rows):
            output[row_index, : len(row)] = row
        return output


class FrozenFitData:
    """Structural train arrays plus the frozen D2p state coordinate."""

    def __init__(self, config: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
        paths = config["paths"]
        root = Path(paths["c06_artifact_root"])
        packed = Path(paths["packed_train_root"])
        registered = config["registered_inputs"]
        names = {
            "feature_request_indices": "feature_request_indices.npy",
            "feature_candidate_offsets": "feature_candidate_offsets.npy",
            "base_scores": "base_scores.npy",
            "query_embeddings": "query_embeddings.npy",
            "item_embedding_indices": "item_embedding_indices.npy",
            "item_embeddings": "item_embeddings.npy",
            "fit_request_indices": "fit_request_indices.npy",
        }
        for key, filename in names.items():
            expected = str(registered[f"{key}_sha256"])
            actual = sha256_file(root / filename)
            if actual != expected:
                raise ValueError(f"registered C06 feature changed: {filename}")

        self.request_ids = load_request_ids(packed / "request_ids.jsonl")
        self.candidate_offsets = np.load(packed / "candidate_offsets.npy", mmap_mode="r", allow_pickle=False)
        self.candidate_embedding_indices = np.load(
            packed / "candidate_embedding_indices.npy", mmap_mode="r", allow_pickle=False
        )
        self.candidate_item_ids = np.load(
            packed / "candidate_item_ids.npy", mmap_mode="r", allow_pickle=False
        )
        self.history_offsets = np.load(packed / "history_offsets.npy", mmap_mode="r", allow_pickle=False)
        self.history_embedding_indices = np.load(
            packed / "history_embedding_indices.npy", mmap_mode="r", allow_pickle=False
        )
        if len(self.candidate_offsets) != len(self.request_ids) + 1:
            raise ValueError("packed candidate offsets changed")
        if len(self.history_offsets) != len(self.request_ids) + 1:
            raise ValueError("packed history offsets changed")
        if int(self.candidate_offsets[-1]) != len(self.candidate_embedding_indices):
            raise ValueError("packed candidate embedding rows changed")
        if len(self.candidate_item_ids) != len(self.candidate_embedding_indices):
            raise ValueError("packed candidate item rows changed")
        if int(self.history_offsets[-1]) != len(self.history_embedding_indices):
            raise ValueError("packed history rows changed")

        self.feature_request_indices = np.load(root / "feature_request_indices.npy", allow_pickle=False)
        self.feature_positions = {
            int(index): position for position, index in enumerate(self.feature_request_indices)
        }
        self.query_embeddings = np.load(root / "query_embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.item_embedding_indices = np.load(
            root / "item_embedding_indices.npy", mmap_mode="r", allow_pickle=False
        )
        self.item_embeddings = np.load(root / "item_embeddings.npy", mmap_mode="r", allow_pickle=False)
        self.feature_candidate_offsets = np.load(
            root / "feature_candidate_offsets.npy", mmap_mode="r", allow_pickle=False
        )
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r", allow_pickle=False)
        if len(self.query_embeddings) != len(self.feature_request_indices):
            raise ValueError("frozen query-state rows changed")
        if len(self.item_embeddings) != len(self.item_embedding_indices):
            raise ValueError("frozen item-state rows changed")
        if len(self.feature_candidate_offsets) != len(self.feature_request_indices) + 1:
            raise ValueError("frozen base offsets changed")
        if int(self.feature_candidate_offsets[-1]) != len(self.base_scores):
            raise ValueError("frozen base-score rows changed")
        if not bool(np.all(np.diff(self.item_embedding_indices) > 0)):
            raise ValueError("frozen item indices are not sorted unique")

        c06 = read_json(paths["c06_selection"])
        expected_features = np.asarray(
            [
                int(index)
                for role in ("fit", "internal_A", "nohistory")
                for index in c06["roles"][role]["indices"]
            ],
            dtype=np.int64,
        )
        if not np.array_equal(self.feature_request_indices, expected_features):
            raise ValueError("C06 feature-role order changed")
        selected = {
            int(index)
            for role in ("train_fit", "internal_probe")
            for index in selection["roles"][role]["indices"]
        }
        c06_fit = {int(index) for index in c06["roles"]["fit"]["indices"]}
        if selected != c06_fit:
            raise PermissionError("C21 selection is not exactly the C06 fit role")
        self.nohistory_indices = np.asarray(c06["roles"]["nohistory"]["indices"], dtype=np.int64)

    def open_labels(self, config: Mapping[str, Any]) -> CompactLabels:
        """Open only C06's already materialized compact fit-label artifact."""

        root = Path(config["paths"]["c06_artifact_root"])
        registered = config["registered_inputs"]
        paths = {
            "request_indices": root / "fit_request_indices.npy",
            "offsets": root / "fit_label_offsets.npy",
            "values": root / "fit_labels.npy",
        }
        expected = {
            "request_indices": registered["fit_request_indices_sha256"],
            "offsets": registered["fit_label_offsets_sha256"],
            "values": registered["fit_labels_sha256"],
        }
        for name, path in paths.items():
            if sha256_file(path) != str(expected[name]):
                raise ValueError(f"registered compact fit label changed: {name}")
        labels = CompactLabels(
            np.load(paths["request_indices"], allow_pickle=False),
            np.load(paths["offsets"], allow_pickle=False),
            np.load(paths["values"], mmap_mode="r", allow_pickle=False),
        )
        if set(int(value) for value in labels.request_indices) != set(self.feature_request_indices[:12_000].tolist()):
            raise PermissionError("compact labels are not exactly C06 fit")
        return labels

    def candidate_count(self, index: int) -> int:
        return int(self.candidate_offsets[index + 1] - self.candidate_offsets[index])

    def candidate_key_sha256(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for raw_index in indices:
            index = int(raw_index)
            start = int(self.candidate_offsets[index])
            stop = int(self.candidate_offsets[index + 1])
            payload = json.dumps(
                [
                    self.request_ids[index],
                    [str(value) for value in self.candidate_item_ids[start:stop]],
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def history_count(self, index: int, history_limit: int) -> int:
        return min(int(self.history_offsets[index + 1] - self.history_offsets[index]), history_limit)

    def _item_states(self, indices: np.ndarray) -> np.ndarray:
        raw = np.asarray(indices, dtype=np.int64)
        positions = np.searchsorted(self.item_embedding_indices, raw)
        if bool((positions >= len(self.item_embedding_indices)).any()):
            raise ValueError("request references an unmaterialized C06 item state")
        if not np.array_equal(np.asarray(self.item_embedding_indices[positions], dtype=np.int64), raw):
            raise ValueError("request references an unmaterialized C06 item state")
        return np.asarray(self.item_embeddings[positions], dtype=np.float32).copy()

    def collate(
        self,
        request_indices: Sequence[int] | np.ndarray,
        *,
        history_limit: int,
        labels: CompactLabels | None,
        history_sources: Mapping[int, int] | None = None,
        shuffle_history: bool = False,
        shuffle_payload: str = "c21-shuffle-v1\\0<request_id>",
    ) -> dict[str, Any]:
        indices = np.asarray(request_indices, dtype=np.int64)
        if not len(indices):
            raise ValueError("cannot collate an empty C21 batch")
        candidate_counts = [self.candidate_count(int(index)) for index in indices]
        source_indices = [
            int(history_sources.get(int(index), int(index))) if history_sources is not None else int(index)
            for index in indices
        ]
        history_counts = [self.history_count(index, history_limit) for index in source_indices]
        batch = len(indices)
        max_candidates = max(candidate_counts)
        max_history = max(1, max(history_counts))
        candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_item_ids = np.full((batch, max_candidates), -1, dtype=np.int64)
        candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
        history_indices = np.zeros((batch, max_history), dtype=np.int64)
        history_mask = np.zeros((batch, max_history), dtype=bool)
        base = np.zeros((batch, max_candidates), dtype=np.float32)
        query = np.zeros((batch, self.query_embeddings.shape[1]), dtype=np.float32)

        for row, (raw_index, history_index) in enumerate(zip(indices, source_indices)):
            index = int(raw_index)
            if index not in self.feature_positions or history_index not in self.feature_positions:
                raise PermissionError("C21 request is outside materialized C06 features")
            position = self.feature_positions[index]
            query[row] = self.query_embeddings[position]
            cs, ce = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
            count = ce - cs
            candidate_indices[row, :count] = self.candidate_embedding_indices[cs:ce]
            candidate_item_ids[row, :count] = self.candidate_item_ids[cs:ce]
            candidate_mask[row, :count] = True
            bs, be = int(self.feature_candidate_offsets[position]), int(self.feature_candidate_offsets[position + 1])
            if be - bs != count:
                raise ValueError("C21 frozen candidate/base alignment changed")
            base[row, :count] = self.base_scores[bs:be]

            hs, he = int(self.history_offsets[history_index]), int(self.history_offsets[history_index + 1])
            start = max(hs, he - history_limit)
            history_row = np.asarray(self.history_embedding_indices[start:he], dtype=np.int64).copy()
            if shuffle_history and len(history_row) >= 2:
                request_id = self.request_ids[index]
                payload = shuffle_payload.replace("<request_id>", request_id).replace("\\0", "\0")
                seed = int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")
                permutation = np.random.default_rng(seed).permutation(len(history_row))
                if np.array_equal(permutation, np.arange(len(history_row))):
                    permutation = np.roll(permutation, 1)
                history_row = history_row[permutation]
            history_indices[row, : len(history_row)] = history_row
            history_mask[row, : len(history_row)] = True

        result: dict[str, Any] = {
            "request_indices": indices,
            "request_ids": [self.request_ids[int(index)] for index in indices],
            "candidate_item_ids": candidate_item_ids,
            "candidate_mask_numpy": candidate_mask,
            "query_numpy": query,
            "candidate_states_numpy": self._item_states(candidate_indices),
            "history_states_numpy": self._item_states(history_indices),
            "history_mask_numpy": history_mask,
            "base_scores_numpy": base,
        }
        if labels is not None:
            result["labels_numpy"] = labels.padded(indices, candidate_mask)
        return result


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    output = {
        "query": torch.from_numpy(batch["query_numpy"]).to(device),
        "candidates": torch.from_numpy(batch["candidate_states_numpy"]).to(device),
        "history": torch.from_numpy(batch["history_states_numpy"]).to(device),
        "candidate_mask": torch.from_numpy(batch["candidate_mask_numpy"]).to(device),
        "history_mask": torch.from_numpy(batch["history_mask_numpy"]).to(device),
        "base_scores": torch.from_numpy(batch["base_scores_numpy"]).to(device),
    }
    if "labels_numpy" in batch:
        output["labels"] = torch.from_numpy(batch["labels_numpy"]).to(device)
    return output


def iter_batches(
    data: FrozenFitData,
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
        candidates = data.candidate_count(index)
        history = data.history_count(index, history_limit)
        next_size = len(batch) + 1
        if batch and (
            next_size > max_requests
            or next_size * max(max_candidates, candidates) > max_padded_candidates
            or next_size * max(max_history, history, 1) > max_padded_history
        ):
            yield np.asarray(batch, dtype=np.int64)
            batch = []
            max_candidates = 0
            max_history = 0
        batch.append(index)
        max_candidates = max(max_candidates, candidates)
        max_history = max(max_history, history, 1)
    if batch:
        yield np.asarray(batch, dtype=np.int64)


def wrong_history_sources(
    data: FrozenFitData,
    indices: Sequence[int],
    *,
    payload_template: str,
    history_limit: int,
) -> dict[int, int]:
    """Create an exact-length-matched deterministic derangement."""

    values = [int(index) for index in indices]
    groups: dict[int, list[int]] = {}
    for index in values:
        groups.setdefault(data.history_count(index, history_limit), []).append(index)
    mapping: dict[int, int] = {}
    for count, group in sorted(groups.items()):
        if len(group) < 2:
            raise ValueError(f"cannot derange singleton C21 history-length group: {count}")
        ordered = sorted(
            group,
            key=lambda index: (
                hashlib.sha256(
                    payload_template.replace("<request_id>", data.request_ids[index])
                    .replace("\\0", "\0")
                    .encode("utf-8")
                ).digest(),
                index,
            ),
        )
        mapping.update(zip(ordered, ordered[1:] + ordered[:1]))
    if any(index == donor for index, donor in mapping.items()):
        raise ValueError("wrong-history donor permutation contains a self donor")
    if set(mapping) != set(mapping.values()):
        raise ValueError("wrong-history donors are not bijective")
    if any(
        data.history_count(index, history_limit) != data.history_count(donor, history_limit)
        for index, donor in mapping.items()
    ):
        raise ValueError("wrong-history donor changed clipped history length")
    return mapping
