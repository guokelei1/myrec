"""Locked C56 packed data and contextual-token collation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class CompactLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if len(self.offsets) != len(self.request_indices) + 1:
            raise ValueError("C56 compact label offsets differ")
        if int(self.offsets[-1]) != len(self.values):
            raise ValueError("C56 compact label values differ")

    @property
    def positions(self) -> dict[int, int]:
        return {int(value): row for row, value in enumerate(self.request_indices)}

    def row(self, index: int, count: int) -> np.ndarray:
        position = self.positions
        if int(index) not in position:
            raise PermissionError(f"C56 label unavailable: {index}")
        row = position[int(index)]
        start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
        if stop - start != int(count):
            raise ValueError("C56 label candidate count differs")
        return np.asarray(self.values[start:stop], dtype=np.float32).copy()


class PackedData:
    def __init__(self, root: Path) -> None:
        self.root = root
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
        self.history_event_weights = np.load(root / "history_event_weights.npy", mmap_mode="r")

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
            np.asarray(self.history_event_weights[start:stop], dtype=np.float32),
        )


class ContextualTokens:
    def __init__(self, manifest: Mapping[str, Any], repo_root: Path) -> None:
        self.item_indices: list[np.ndarray] = []
        self.item_states: list[np.ndarray] = []
        self.item_content: list[np.ndarray] = []
        self.query_indices: list[np.ndarray] = []
        self.query_states: list[np.ndarray] = []
        self.query_content: list[np.ndarray] = []
        self.item_location: dict[int, tuple[int, int]] = {}
        self.query_location: dict[int, tuple[int, int]] = {}
        for shard_id, report in enumerate(manifest["shards"]):
            outputs = report["outputs"]
            item_indices = np.load(repo_root / outputs["item_indices"]["path"], mmap_mode="r")
            item_states = np.load(repo_root / outputs["item_states"]["path"], mmap_mode="r")
            item_content = np.load(repo_root / outputs["item_content"]["path"], mmap_mode="r")
            query_indices = np.load(repo_root / outputs["query_indices"]["path"], mmap_mode="r")
            query_states = np.load(repo_root / outputs["query_states"]["path"], mmap_mode="r")
            query_content = np.load(repo_root / outputs["query_content"]["path"], mmap_mode="r")
            if item_states.shape[:-1] != item_content.shape or query_states.shape[:-1] != query_content.shape:
                raise ValueError("C56 contextual state/mask shape differs")
            self.item_indices.append(item_indices)
            self.item_states.append(item_states)
            self.item_content.append(item_content)
            self.query_indices.append(query_indices)
            self.query_states.append(query_states)
            self.query_content.append(query_content)
            for row, value in enumerate(item_indices):
                index = int(value)
                if index in self.item_location:
                    raise ValueError("C56 duplicate contextual item")
                self.item_location[index] = (shard_id, row)
            for row, value in enumerate(query_indices):
                index = int(value)
                if index in self.query_location:
                    raise ValueError("C56 duplicate contextual query")
                self.query_location[index] = (shard_id, row)
        first = self.item_states[0]
        self.item_length = int(first.shape[1])
        self.input_dim = int(first.shape[2])
        self.query_length = int(self.query_states[0].shape[1])

    def query(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        shard, row = self.query_location[int(index)]
        return (
            np.asarray(self.query_states[shard][row], dtype=np.float32),
            np.asarray(self.query_content[shard][row], dtype=bool),
        )

    def items(self, indices: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        states = np.zeros((*indices.shape, self.item_length, self.input_dim), dtype=np.float32)
        content = np.zeros((*indices.shape, self.item_length), dtype=bool)
        for position in zip(*np.where(mask)):
            index = int(indices[position])
            if index not in self.item_location:
                raise KeyError(f"C56 contextual item unavailable: {index}")
            shard, row = self.item_location[index]
            states[position] = np.asarray(self.item_states[shard][row], dtype=np.float32)
            content[position] = np.asarray(self.item_content[shard][row], dtype=bool)
        return states, content


def zscore(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    for row in range(len(values)):
        selected = np.asarray(values[row, mask[row]], dtype=np.float64)
        if not len(selected):
            continue
        scale = float(selected.std())
        if scale > 1e-8:
            output[row, mask[row]] = ((selected - selected.mean()) / scale).astype(np.float32)
    return output


class C56Store:
    def __init__(self, config: Mapping[str, Any], repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.data = PackedData(repo_root / config["paths"]["packed_train_root"])
        self.selection = json.loads(
            (repo_root / config["paths"]["selection"]).read_text(encoding="utf-8")
        )
        if sha256_file(repo_root / config["paths"]["selection"]) != config["integrity"]["c56_v2_selection_sha256"]:
            raise RuntimeError("C56 frozen selection changed")
        if self.selection.get("status") != "label_blind_split_frozen":
            raise RuntimeError("C56 frozen selection did not pass")
        manifest = json.loads(
            (repo_root / config["paths"]["contextual_manifest"]).read_text(encoding="utf-8")
        )
        if manifest.get("status") != "passed":
            raise RuntimeError("C56 contextual manifest did not pass")
        self.tokens = ContextualTokens(manifest, repo_root)
        source_root = repo_root / config["paths"]["c26_artifact_root"]
        self.feature_indices = np.load(source_root / "feature_request_indices.npy", mmap_mode="r")
        self.feature_position = {int(value): row for row, value in enumerate(self.feature_indices)}
        self.score_offsets = np.load(source_root / "feature_candidate_offsets.npy", mmap_mode="r")
        self.base_scores = np.load(source_root / "base_scores.npy", mmap_mode="r")
        self.labels = CompactLabels(
            request_indices=np.load(source_root / "fit_request_indices.npy", mmap_mode="r"),
            offsets=np.load(source_root / "fit_label_offsets.npy", mmap_mode="r"),
            values=np.load(source_root / "fit_labels.npy", mmap_mode="r"),
        )
        self.label_position = {
            int(value): row for row, value in enumerate(self.labels.request_indices)
        }
        self.donor: dict[int, int] = {}
        for role in ("train", "holdout"):
            self.donor.update(
                zip(
                    map(int, self.selection["roles"][role]),
                    map(int, self.selection["wrong_history_donors"][role]),
                )
            )

    def role(self, name: str) -> list[int]:
        return [int(value) for value in self.selection["roles"][name]]

    def candidate_hash(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for raw_index in indices:
            index = int(raw_index)
            start, stop = int(self.data.candidate_offsets[index]), int(self.data.candidate_offsets[index + 1])
            payload = json.dumps(
                [
                    self.data.request_ids[index],
                    [str(value) for value in self.data.candidate_item_ids[start:stop]],
                ],
                separators=(",", ":"),
            ).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def label(self, index: int) -> np.ndarray:
        if int(index) not in self.label_position:
            raise PermissionError(f"C56 label unavailable: {index}")
        row = self.label_position[int(index)]
        start, stop = int(self.labels.offsets[row]), int(self.labels.offsets[row + 1])
        expected = len(self.data.candidate_ids(index))
        if stop - start != expected:
            raise ValueError("C56 label candidate count differs")
        return np.asarray(self.labels.values[start:stop], dtype=np.float32).copy()

    def collate(
        self,
        indices: Sequence[int],
        *,
        history_source: str = "true",
        with_labels: bool = False,
    ) -> dict[str, Any]:
        if history_source not in {"true", "wrong"}:
            raise ValueError("C56 history source differs")
        request_indices = np.asarray(indices, dtype=np.int64)
        source_indices = np.asarray(
            [
                int(index)
                if history_source == "true"
                else self.donor[int(index)]
                for index in request_indices
            ],
            dtype=np.int64,
        )
        candidate_counts = np.asarray(
            [len(self.data.candidates(int(index))) for index in request_indices], dtype=np.int64
        )
        history_counts = np.asarray(
            [len(self.data.history(int(index))[0]) for index in source_indices], dtype=np.int64
        )
        batch = len(request_indices)
        max_candidates = int(candidate_counts.max())
        max_history = max(1, int(history_counts.max()))
        candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
        candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
        history_indices = np.zeros((batch, max_history), dtype=np.int64)
        history_mask = np.zeros((batch, max_history), dtype=bool)
        event_weights = np.zeros((batch, max_history), dtype=np.float32)
        candidate_item_ids = np.empty((batch, max_candidates), dtype=object)
        base = np.zeros((batch, max_candidates), dtype=np.float32)
        item_component = np.zeros((batch, max_candidates), dtype=np.float32)
        repeat_request = np.zeros(batch, dtype=bool)
        labels = np.zeros((batch, max_candidates), dtype=np.float32) if with_labels else None
        request_ids: list[str] = []
        query_states = np.zeros(
            (batch, self.tokens.query_length, self.tokens.input_dim), dtype=np.float32
        )
        query_mask = np.zeros((batch, self.tokens.query_length), dtype=bool)
        for row, (raw_index, raw_source) in enumerate(zip(request_indices, source_indices)):
            index, source = int(raw_index), int(raw_source)
            candidates = self.data.candidates(index)
            history, weights = self.data.history(source)
            cc, hc = len(candidates), len(history)
            candidate_indices[row, :cc] = candidates
            candidate_mask[row, :cc] = True
            history_indices[row, :hc] = history
            history_mask[row, :hc] = True
            event_weights[row, :hc] = weights
            ids = self.data.candidate_ids(index)
            candidate_item_ids[row, :cc] = ids
            feature_row = self.feature_position[index]
            start, stop = int(self.score_offsets[feature_row]), int(self.score_offsets[feature_row + 1])
            if stop - start != cc:
                raise ValueError("C56 base candidate count differs")
            base[row, :cc] = self.base_scores[start:stop]
            q, qm = self.tokens.query(index)
            query_states[row], query_mask[row] = q, qm
            repeat = candidates[:, None] == history[None, :] if hc else np.zeros((cc, 0), dtype=bool)
            repeat_request[row] = bool(repeat.any())
            if hc:
                reverse = np.maximum(hc - np.arange(hc), 1).astype(np.float32)
                recurrence = weights / np.sqrt(reverse)
                item_component[row, :cc] = 3.0 * (repeat.astype(np.float32) * recurrence[None]).sum(-1)
            if with_labels and labels is not None:
                labels[row, :cc] = self.label(index)
            request_ids.append(self.data.request_ids[index])
        candidate_states, candidate_token_mask = self.tokens.items(candidate_indices, candidate_mask)
        history_states, history_token_mask = self.tokens.items(history_indices, history_mask)
        base_z = zscore(base, candidate_mask)
        item_z = zscore(item_component, candidate_mask)
        beta = 0.3
        item_only = beta * base_z + (1.0 - beta) * item_z
        result: dict[str, Any] = {
            "request_indices": request_indices,
            "request_ids": request_ids,
            "candidate_item_ids": candidate_item_ids,
            "query_tokens": query_states,
            "query_token_mask": query_mask,
            "candidate_tokens": candidate_states,
            "candidate_token_mask": candidate_token_mask,
            "history_tokens": history_states,
            "history_token_mask": history_token_mask,
            "event_weights": event_weights,
            "candidate_mask": candidate_mask,
            "base_scores": base_z,
            "item_only_scores": item_only,
            "repeat_request": repeat_request,
        }
        if labels is not None:
            result["labels"] = labels
        return result


TENSOR_NAMES = (
    "query_tokens",
    "query_token_mask",
    "candidate_tokens",
    "candidate_token_mask",
    "history_tokens",
    "history_token_mask",
    "event_weights",
    "candidate_mask",
    "base_scores",
    "item_only_scores",
    "repeat_request",
)


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    output = {
        name: torch.from_numpy(np.asarray(batch[name])).to(device)
        for name in TENSOR_NAMES
    }
    if "labels" in batch:
        output["labels"] = torch.from_numpy(np.asarray(batch["labels"])).to(device)
    return output


def iter_batches(
    store: C56Store,
    indices: Sequence[int],
    *,
    seed: int,
    shuffle: bool,
    max_requests: int,
    max_transport_cells: int,
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64)
    if shuffle:
        order = order[np.random.default_rng(seed).permutation(len(order))]
    current: list[int] = []
    max_candidates = max_history = 0
    token_length = store.tokens.item_length
    for raw_index in order:
        index = int(raw_index)
        candidates = len(store.data.candidates(index))
        history = max(1, len(store.data.history(index)[0]))
        next_c, next_h = max(max_candidates, candidates), max(max_history, history)
        cells = (len(current) + 1) * next_c * token_length * next_h * token_length
        if current and (len(current) + 1 > max_requests or cells > max_transport_cells):
            yield np.asarray(current, dtype=np.int64)
            current, max_candidates, max_history = [], 0, 0
        current.append(index)
        max_candidates, max_history = max(max_candidates, candidates), max(max_history, history)
    if current:
        yield np.asarray(current, dtype=np.int64)
