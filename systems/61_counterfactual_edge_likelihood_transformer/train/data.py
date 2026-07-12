"""C61 staged packed-data and contextual-token interface."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch

from myrec.eval.metrics import ScoredCandidate, sort_candidates


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

    @property
    def positions(self) -> dict[int, int]:
        return {int(value): row for row, value in enumerate(self.request_indices)}

    def row(self, index: int, count: int) -> np.ndarray:
        position = self.positions
        if int(index) not in position:
            raise PermissionError(f"C61 fit label unavailable: {index}")
        row = position[int(index)]
        start, stop = int(self.offsets[row]), int(self.offsets[row + 1])
        if stop - start != count:
            raise ValueError("C61 fit label candidate count differs")
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
        self.item_states: list[np.ndarray] = []
        self.item_content: list[np.ndarray] = []
        self.query_states: list[np.ndarray] = []
        self.query_content: list[np.ndarray] = []
        self.item_location: dict[int, tuple[int, int]] = {}
        self.query_location: dict[int, tuple[int, int]] = {}
        for shard_id, report in enumerate(manifest["shards"]):
            output = report["outputs"]
            item_indices = np.load(repo_root / output["item_indices"]["path"], mmap_mode="r")
            item_states = np.load(repo_root / output["item_states"]["path"], mmap_mode="r")
            item_content = np.load(repo_root / output["item_content"]["path"], mmap_mode="r")
            query_indices = np.load(repo_root / output["query_indices"]["path"], mmap_mode="r")
            query_states = np.load(repo_root / output["query_states"]["path"], mmap_mode="r")
            query_content = np.load(repo_root / output["query_content"]["path"], mmap_mode="r")
            self.item_states.append(item_states)
            self.item_content.append(item_content)
            self.query_states.append(query_states)
            self.query_content.append(query_content)
            for row, value in enumerate(item_indices):
                if int(value) in self.item_location:
                    raise ValueError("C61 duplicate contextual item")
                self.item_location[int(value)] = (shard_id, row)
            for row, value in enumerate(query_indices):
                if int(value) in self.query_location:
                    raise ValueError("C61 duplicate contextual query")
                self.query_location[int(value)] = (shard_id, row)
        self.item_length = int(self.item_states[0].shape[1])
        self.query_length = int(self.query_states[0].shape[1])
        self.input_dim = int(self.item_states[0].shape[2])

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
                raise KeyError(f"C61 contextual item unavailable: {index}")
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


def role_indices(selection: Mapping[str, Any], role: str) -> list[int]:
    return [int(value) for value in selection["roles"][role]["indices"]]


class C61Store:
    def __init__(self, config: Mapping[str, Any], repo_root: Path) -> None:
        self.config = config
        self.repo_root = repo_root
        self.data = PackedData(repo_root / config["paths"]["packed_train_root"])
        self.selection = json.loads(
            (repo_root / config["paths"]["c26_selection"]).read_text(encoding="utf-8")
        )
        if sha256_file(repo_root / config["paths"]["c26_selection"]) != config["integrity"]["c26_selection_sha256"]:
            raise RuntimeError("C61 C26 selection changed")
        manifest = json.loads(
            (repo_root / config["paths"]["contextual_manifest"]).read_text(encoding="utf-8")
        )
        if manifest.get("status") != "passed":
            raise RuntimeError("C61 contextual manifest did not pass")
        self.tokens = ContextualTokens(manifest, repo_root)
        source_root = repo_root / config["paths"]["c26_artifact_root"]
        self.feature_indices = np.load(source_root / "feature_request_indices.npy", mmap_mode="r")
        self.feature_position = {int(value): row for row, value in enumerate(self.feature_indices)}
        self.score_offsets = np.load(source_root / "feature_candidate_offsets.npy", mmap_mode="r")
        self.base_scores = np.load(source_root / "base_scores.npy", mmap_mode="r")
        self._source_root = source_root
        self._fit_labels_cache: CompactLabels | None = None
        self._all_labels_cache: np.ndarray | None = None
        self.donor: dict[int, int] = {}
        for role in ("fit", "internal_A"):
            requests = role_indices(self.selection, role)
            donors = [
                int(value)
                for value in self.selection["wrong_history_donors"][role]["indices"]
            ]
            if len(requests) != len(donors):
                raise ValueError("C61 wrong-history donor length differs")
            self.donor.update(zip(requests, donors))

    def role(self, name: str) -> list[int]:
        return role_indices(self.selection, name)

    def candidate_hash(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for raw_index in indices:
            index = int(raw_index)
            payload = json.dumps(
                [self.data.request_ids[index], self.data.candidate_ids(index)],
                separators=(",", ":"),
            ).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    def _fit_labels(self) -> CompactLabels:
        if self._fit_labels_cache is None:
            self._fit_labels_cache = CompactLabels(
                request_indices=np.load(self._source_root / "fit_request_indices.npy", mmap_mode="r"),
                offsets=np.load(self._source_root / "fit_label_offsets.npy", mmap_mode="r"),
                values=np.load(self._source_root / "fit_labels.npy", mmap_mode="r"),
            )
        return self._fit_labels_cache

    def labels(self, index: int, *, role: str) -> np.ndarray:
        count = len(self.data.candidates(index))
        if role == "fit":
            return self._fit_labels().row(index, count)
        if role != "internal_A":
            raise PermissionError(f"C61 label role forbidden: {role}")
        if self._all_labels_cache is None:
            self._all_labels_cache = np.load(
                self.repo_root / self.config["paths"]["packed_train_root"] / "candidate_labels.npy",
                mmap_mode="r",
            )
        start, stop = int(self.data.candidate_offsets[index]), int(self.data.candidate_offsets[index + 1])
        return np.asarray(self._all_labels_cache[start:stop], dtype=np.float32).copy()

    @staticmethod
    def _canonical_order(request_id: str, items: Sequence[str], scores: np.ndarray) -> np.ndarray:
        ranked = sort_candidates(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(items, scores)],
        )
        position = {str(item): row for row, item in enumerate(items)}
        if len(position) != len(items):
            raise ValueError("C61 duplicate candidate item")
        return np.asarray([position[row.item_id] for row in ranked], dtype=np.int64)

    def collate(
        self,
        indices: Sequence[int],
        *,
        history_source: str = "true",
        label_role: str | None = None,
    ) -> dict[str, Any]:
        if history_source not in {"true", "wrong"}:
            raise ValueError("C61 history source differs")
        request_indices = np.asarray(indices, dtype=np.int64)
        source_indices = np.asarray(
            [int(index) if history_source == "true" else self.donor[int(index)] for index in request_indices],
            dtype=np.int64,
        )
        candidate_counts = np.asarray([len(self.data.candidates(int(index))) for index in request_indices])
        history_counts = np.asarray([len(self.data.history(int(index))[0]) for index in source_indices])
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
        labels = np.zeros((batch, max_candidates), dtype=np.float32) if label_role else None
        canonical_order = np.full((batch, max_candidates), -1, dtype=np.int64)
        query_states = np.zeros((batch, self.tokens.query_length, self.tokens.input_dim), dtype=np.float32)
        query_mask = np.zeros((batch, self.tokens.query_length), dtype=bool)
        request_ids: list[str] = []
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
            items = self.data.candidate_ids(index)
            candidate_item_ids[row, :cc] = items
            feature_row = self.feature_position[index]
            start, stop = int(self.score_offsets[feature_row]), int(self.score_offsets[feature_row + 1])
            if stop - start != cc:
                raise ValueError("C61 base candidate count differs")
            base[row, :cc] = self.base_scores[start:stop]
            query_states[row], query_mask[row] = self.tokens.query(index)
            repeat = candidates[:, None] == history[None, :] if hc else np.zeros((cc, 0), dtype=bool)
            repeat_request[row] = bool(repeat.any())
            if hc:
                reverse = np.maximum(hc - np.arange(hc), 1).astype(np.float32)
                recurrence = weights / np.sqrt(reverse)
                item_component[row, :cc] = 3.0 * (repeat.astype(np.float32) * recurrence[None]).sum(-1)
            request_id = self.data.request_ids[index]
            request_ids.append(request_id)
            if labels is not None and label_role is not None:
                labels[row, :cc] = self.labels(index, role=label_role)
        candidate_states, candidate_token_mask = self.tokens.items(candidate_indices, candidate_mask)
        history_states, history_token_mask = self.tokens.items(history_indices, history_mask)
        base_z = zscore(base, candidate_mask)
        item_z = zscore(item_component, candidate_mask)
        item_only = 0.3 * base_z + 0.7 * item_z
        for row, request_id in enumerate(request_ids):
            count = int(candidate_counts[row])
            canonical_order[row, :count] = self._canonical_order(
                request_id,
                [str(value) for value in candidate_item_ids[row, :count]],
                base_z[row, :count],
            )
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
            "canonical_order": canonical_order,
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
    "canonical_order",
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
    store: C61Store,
    indices: Sequence[int],
    *,
    seed: int,
    shuffle: bool,
    max_requests: int,
    max_edge_cells: int,
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64)
    if shuffle:
        order = order[np.random.default_rng(seed).permutation(len(order))]
    current: list[int] = []
    max_candidates = max_history = 0
    for raw_index in order:
        index = int(raw_index)
        candidates = len(store.data.candidates(index))
        history = max(1, len(store.data.history(index)[0]))
        next_c, next_h = max(max_candidates, candidates), max(max_history, history)
        cells = (len(current) + 1) * max(1, next_c - 1) * next_h * int(store.tokens.input_dim)
        if current and (len(current) + 1 > max_requests or cells > max_edge_cells):
            yield np.asarray(current, dtype=np.int64)
            current, max_candidates, max_history = [], 0, 0
        current.append(index)
        max_candidates, max_history = max(max_candidates, candidates), max(max_history, history)
    if current:
        yield np.asarray(current, dtype=np.int64)
