"""Read-only unified-interface adapter for the C01 probe.

Candidate-local arrays are merely caches of label-free categories/user keys and
registered baseline scores.  They are written under the C01 artifact prefix and
never alter the shared standardized data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error


def stable_bucket(value: str, buckets: int) -> int:
    if buckets < 2:
        raise ValueError("category buckets must be at least two")
    value = str(value or "").strip()
    if not value or value.upper() == "UNKNOWN":
        return 0
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (buckets - 1)


def deepest_category(row: dict[str, Any]) -> str:
    categories = [str(value) for value in row.get("cat", [])]
    for value in reversed(categories):
        if value and value.upper() != "UNKNOWN":
            return value
    return ""


def stable_user_key(user_id: str) -> np.uint64:
    digest = hashlib.sha256(str(user_id).encode("utf-8")).digest()
    return np.uint64(int.from_bytes(digest[:8], "big"))


def _zscore(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    return (values - values.mean()) / math.sqrt(float(values.var()) + eps)


@dataclass
class LocalArrayPaths:
    candidate_categories: Path
    history_categories: Path
    user_keys: Path


def assert_candidate_manifest(path: str | Path, expected_sha256: str) -> str:
    """Fail closed when the frozen candidate universe changes."""

    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(
            f"candidate manifest hash mismatch: {actual} != {expected_sha256}"
        )
    return actual


def prepare_local_arrays(config: dict[str, Any]) -> dict[str, Any]:
    """Materialize category hashes and aligned registered dev score arrays."""

    paths = config["paths"]
    data_config = config["data"]
    artifact_root = Path(paths["c01_artifacts"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    packed_root = Path(paths["packed_data"])
    standardized_dir = Path(paths["standardized_dir"])
    buckets = int(data_config["category_buckets"])

    split_results: dict[str, Any] = {}
    for split in ("train", "dev"):
        destination = artifact_root / split
        destination.mkdir(parents=True, exist_ok=True)
        local_paths = LocalArrayPaths(
            candidate_categories=destination / "candidate_categories.npy",
            history_categories=destination / "history_categories.npy",
            user_keys=destination / "user_keys.npy",
        )
        if not all(
            path.exists()
            for path in (
                local_paths.candidate_categories,
                local_paths.history_categories,
                local_paths.user_keys,
            )
        ):
            _materialize_split_metadata(
                records_path=standardized_dir / f"records_{split}.jsonl",
                packed_dir=packed_root / split,
                output_paths=local_paths,
                category_buckets=buckets,
            )
        split_results[split] = {
            "candidate_categories": _file_record(local_paths.candidate_categories),
            "history_categories": _file_record(local_paths.history_categories),
            "user_keys": _file_record(local_paths.user_keys),
        }

    dev_base = artifact_root / "dev_d2p_scores.npy"
    dev_item_only = artifact_root / "dev_item_only_scores.npy"
    if not dev_base.exists():
        _materialize_aligned_scores(
            Path(paths["d2p_dev_scores"]), packed_root / "dev", dev_base
        )
    if not dev_item_only.exists():
        _materialize_aligned_scores(
            Path(paths["item_only_dev_scores"]), packed_root / "dev", dev_item_only
        )

    candidate_manifest_sha256 = assert_candidate_manifest(
        paths["candidate_manifest"], paths["candidate_manifest_sha256"]
    )
    manifest = {
        "candidate_id": config["candidate_id"],
        "candidate_manifest": {
            "path": paths["candidate_manifest"],
            "sha256": candidate_manifest_sha256,
        },
        "category_buckets": buckets,
        "dev_d2p_scores": _file_record(dev_base),
        "dev_item_only_scores": _file_record(dev_item_only),
        "qrel_files_read": False,
        "splits": split_results,
        "test_files_read": False,
    }
    manifest_path = artifact_root / "data_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    return manifest


def _materialize_split_metadata(
    records_path: Path,
    packed_dir: Path,
    output_paths: LocalArrayPaths,
    category_buckets: int,
) -> None:
    request_ids = [
        str(row["request_id"]) for row in iter_jsonl(packed_dir / "request_ids.jsonl")
    ]
    request_to_index = {request_id: index for index, request_id in enumerate(request_ids)}
    if len(request_to_index) != len(request_ids):
        raise ValueError(f"duplicate packed request id under {packed_dir}")
    candidate_offsets = np.load(packed_dir / "candidate_offsets.npy", mmap_mode="r")
    history_offsets = np.load(packed_dir / "history_offsets.npy", mmap_mode="r")
    candidate_categories = np.lib.format.open_memmap(
        output_paths.candidate_categories,
        mode="w+",
        dtype=np.int32,
        shape=(int(candidate_offsets[-1]),),
    )
    history_categories = np.lib.format.open_memmap(
        output_paths.history_categories,
        mode="w+",
        dtype=np.int32,
        shape=(int(history_offsets[-1]),),
    )
    user_keys = np.lib.format.open_memmap(
        output_paths.user_keys,
        mode="w+",
        dtype=np.uint64,
        shape=(len(request_ids),),
    )
    seen = np.zeros(len(request_ids), dtype=np.bool_)
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        request_index = request_to_index.get(request_id)
        if request_index is None:
            # The packed train interface intentionally drops requests without a
            # clicked positive.  Dev is expected to have no skipped request.
            continue
        if seen[request_index]:
            raise ValueError(f"duplicate standardized request: {request_id}")
        candidates = list(record["candidates"])
        candidate_start = int(candidate_offsets[request_index])
        candidate_end = int(candidate_offsets[request_index + 1])
        if len(candidates) != candidate_end - candidate_start:
            raise ValueError(f"candidate alignment mismatch for {request_id}")
        candidate_categories[candidate_start:candidate_end] = np.asarray(
            [
                stable_bucket(deepest_category(candidate), category_buckets)
                for candidate in candidates
            ],
            dtype=np.int32,
        )

        # Shared packed data uses the frozen last-50 history contract.
        history = list(record.get("history") or [])[-50:]
        history_start = int(history_offsets[request_index])
        history_end = int(history_offsets[request_index + 1])
        if len(history) != history_end - history_start:
            raise ValueError(f"history alignment mismatch for {request_id}")
        history_categories[history_start:history_end] = np.asarray(
            [stable_bucket(deepest_category(event), category_buckets) for event in history],
            dtype=np.int32,
        )
        user_keys[request_index] = stable_user_key(str(record["user_id"]))
        seen[request_index] = True

    missing = np.flatnonzero(~seen)
    if missing.size:
        raise ValueError(
            f"standardized records missing {missing.size} packed requests; first={missing[0]}"
        )
    candidate_categories.flush()
    history_categories.flush()
    user_keys.flush()


def _materialize_aligned_scores(
    scores_path: Path, packed_dir: Path, output_path: Path
) -> None:
    request_ids = [
        str(row["request_id"]) for row in iter_jsonl(packed_dir / "request_ids.jsonl")
    ]
    offsets = np.load(packed_dir / "candidate_offsets.npy", mmap_mode="r")
    item_ids = np.load(packed_dir / "candidate_item_ids.npy", mmap_mode="r")
    request_to_index = {request_id: index for index, request_id in enumerate(request_ids)}
    if len(request_to_index) != len(request_ids):
        raise ValueError(f"duplicate packed request id under {packed_dir}")
    temporary = output_path.with_name(output_path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    output = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(item_ids),)
    )
    seen = np.zeros(len(item_ids), dtype=np.bool_)
    row_count = 0
    cached_request = None
    cached_positions: dict[str, int] = {}
    try:
        for row in iter_jsonl(scores_path):
            request_id = str(row["request_id"])
            request_index = request_to_index.get(request_id)
            if request_index is None:
                raise ValueError(f"unknown score request id: {request_id}")
            if cached_request != request_id:
                start = int(offsets[request_index])
                end = int(offsets[request_index + 1])
                cached_positions = {
                    str(int(item_ids[position])): position
                    for position in range(start, end)
                }
                if len(cached_positions) != end - start:
                    raise ValueError(f"duplicate packed candidate for {request_id}")
                cached_request = request_id
            item_id = str(row["candidate_item_id"])
            position = cached_positions.get(item_id)
            if position is None:
                raise ValueError(
                    f"score candidate does not match packed data: {request_id}/{item_id}"
                )
            if seen[position]:
                raise ValueError(f"duplicate score row: {request_id}/{item_id}")
            value = float(row["score"])
            if not math.isfinite(value):
                raise ValueError(f"non-finite registered score at row {row_count}")
            output[position] = value
            seen[position] = True
            row_count += 1
        missing = np.flatnonzero(~seen)
        if row_count != len(item_ids) or missing.size:
            raise ValueError(
                f"score coverage mismatch: rows={row_count}/{len(item_ids)}, "
                f"missing={missing.size}"
            )
        output.flush()
        del output
        os.replace(temporary, output_path)
    except Exception:
        del output
        if temporary.exists():
            temporary.unlink()
        raise


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


class PackedSplit:
    """Memory-mapped train/dev records with candidate-local metadata caches."""

    def __init__(self, config: dict[str, Any], split: str) -> None:
        if split not in {"train", "dev"}:
            raise ValueError("C01 may load only train or dev")
        self.config = config
        self.split = split
        paths = config["paths"]
        self.root = Path(paths["packed_data"]) / split
        self.local_root = Path(paths["c01_artifacts"]) / split
        self.request_ids = [
            str(row["request_id"]) for row in iter_jsonl(self.root / "request_ids.jsonl")
        ]
        self.query_indices = np.load(self.root / "query_indices.npy", mmap_mode="r")
        self.candidate_offsets = np.load(
            self.root / "candidate_offsets.npy", mmap_mode="r"
        )
        self.candidate_indices = np.load(
            self.root / "candidate_embedding_indices.npy", mmap_mode="r"
        )
        self.candidate_item_ids = np.load(
            self.root / "candidate_item_ids.npy", mmap_mode="r"
        )
        self.labels = (
            np.load(self.root / "candidate_labels.npy", mmap_mode="r")
            if split == "train"
            else None
        )
        self.history_offsets = np.load(
            self.root / "history_offsets.npy", mmap_mode="r"
        )
        self.history_indices = np.load(
            self.root / "history_embedding_indices.npy", mmap_mode="r"
        )
        self.history_event_weights = np.load(
            self.root / "history_event_weights.npy", mmap_mode="r"
        )
        self.candidate_categories = np.load(
            self.local_root / "candidate_categories.npy", mmap_mode="r"
        )
        self.history_categories = np.load(
            self.local_root / "history_categories.npy", mmap_mode="r"
        )
        self.user_keys = np.load(self.local_root / "user_keys.npy", mmap_mode="r")
        self.query_embeddings = np.load(paths["query_embeddings"], mmap_mode="r")
        self.item_embeddings = np.load(paths["item_embeddings"], mmap_mode="r")
        self.item_popularity = np.load(paths["item_popularity"], mmap_mode="r")
        self.dev_base_scores = (
            np.load(Path(paths["c01_artifacts"]) / "dev_d2p_scores.npy", mmap_mode="r")
            if split == "dev"
            else None
        )
        self.dev_item_only_scores = (
            np.load(
                Path(paths["c01_artifacts"]) / "dev_item_only_scores.npy",
                mmap_mode="r",
            )
            if split == "dev"
            else None
        )
        self.max_history = int(config["data"]["max_history"])
        self.max_negatives = int(config["data"]["max_negatives_per_request"])
        self.seed = int(config["seed"])
        self._validate()

    def _validate(self) -> None:
        requests = len(self.request_ids)
        if len(self.query_indices) != requests:
            raise ValueError(f"query/request mismatch for {self.split}")
        if len(self.candidate_offsets) != requests + 1:
            raise ValueError(f"candidate offset mismatch for {self.split}")
        if len(self.history_offsets) != requests + 1:
            raise ValueError(f"history offset mismatch for {self.split}")
        if len(self.candidate_categories) != len(self.candidate_indices):
            raise ValueError(f"candidate category mismatch for {self.split}")
        if len(self.history_categories) != len(self.history_indices):
            raise ValueError(f"history category mismatch for {self.split}")
        if len(self.user_keys) != requests:
            raise ValueError(f"user key mismatch for {self.split}")
        if self.split == "train" and len(self) != int(
            self.config["data"]["retained_train_requests"]
        ):
            raise ValueError("frozen retained train count mismatch")
        if self.split == "dev" and len(self) != int(
            self.config["screening"]["expected_requests"]
        ):
            raise ValueError("frozen dev request count mismatch")

    def __len__(self) -> int:
        return len(self.request_ids)

    def history_length(self, request_index: int) -> int:
        start = int(self.history_offsets[request_index])
        end = int(self.history_offsets[request_index + 1])
        return min(end - start, self.max_history)

    def has_exact_candidate(self, request_index: int) -> bool:
        candidate_slice = self._candidate_global_positions(request_index)
        history_slice = self._history_global_positions(request_index)
        if history_slice.size == 0:
            return False
        return bool(
            np.intersect1d(
                np.asarray(self.candidate_indices[candidate_slice]),
                np.asarray(self.history_indices[history_slice]),
                assume_unique=False,
            ).size
        )

    def default_wrong_indices(
        self,
        request_indices: Iterable[int],
        *,
        donor_start: int = 0,
        donor_end: int | None = None,
    ) -> list[int]:
        """Choose the next history-present request from a different user."""

        indices = [int(value) for value in request_indices]
        result: list[int] = []
        donor_end = len(self) if donor_end is None else int(donor_end)
        donor_start = int(donor_start)
        if not 0 <= donor_start < donor_end <= len(self):
            raise ValueError("invalid wrong-history donor bounds")
        size = donor_end - donor_start
        for target in indices:
            donor = target + 1
            if donor < donor_start or donor >= donor_end:
                donor = donor_start
            searched = 0
            while (
                self.history_length(donor) == 0
                or self.user_keys[donor] == self.user_keys[target]
            ):
                donor += 1
                if donor >= donor_end:
                    donor = donor_start
                searched += 1
                if searched >= size:
                    raise RuntimeError("no valid different-user history donor")
            result.append(donor)
        return result

    def build_batch(
        self,
        request_indices: Iterable[int],
        *,
        all_candidates: bool,
        wrong_indices: Iterable[int] | None = None,
        include_wrong_history: bool = True,
    ) -> dict[str, Any]:
        indices = [int(value) for value in request_indices]
        if not indices:
            raise ValueError("empty request batch")
        if self.split == "dev" and not all_candidates:
            raise ValueError("dev scoring must retain all candidates")
        selections = [
            self._select_candidates(index, all_candidates=all_candidates)
            for index in indices
        ]
        candidate_width = max(len(selection) for selection in selections)
        batch_size = len(indices)
        text_dim = int(self.config["model"]["frozen_text_dim"])
        history_width = self.max_history

        query = np.zeros((batch_size, text_dim), dtype=np.float32)
        candidates = np.zeros(
            (batch_size, candidate_width, text_dim), dtype=np.float32
        )
        candidate_indices = np.full(
            (batch_size, candidate_width), -1, dtype=np.int64
        )
        candidate_item_ids = np.full(
            (batch_size, candidate_width), -1, dtype=np.int64
        )
        candidate_categories = np.zeros(
            (batch_size, candidate_width), dtype=np.int64
        )
        candidate_mask = np.zeros((batch_size, candidate_width), dtype=np.bool_)
        labels = np.zeros((batch_size, candidate_width), dtype=np.float32)
        base_scores = np.zeros((batch_size, candidate_width), dtype=np.float32)
        item_only_scores = np.zeros(
            (batch_size, candidate_width), dtype=np.float32
        )

        history = np.zeros(
            (batch_size, history_width, text_dim), dtype=np.float32
        )
        history_indices = np.full(
            (batch_size, history_width), -1, dtype=np.int64
        )
        history_categories = np.zeros(
            (batch_size, history_width), dtype=np.int64
        )
        history_event_weights = np.zeros(
            (batch_size, history_width), dtype=np.float32
        )
        history_mask = np.zeros((batch_size, history_width), dtype=np.bool_)

        if not include_wrong_history and wrong_indices is not None:
            raise ValueError("wrong donors cannot be supplied for true-only scoring")
        if not include_wrong_history:
            wrong = []
        elif wrong_indices is None:
            wrong = self.default_wrong_indices(indices)
        else:
            wrong = [int(value) for value in wrong_indices]
            if len(wrong) != batch_size:
                raise ValueError("wrong donor count mismatch")
        wrong_history = np.zeros_like(history)
        wrong_history_indices = np.full_like(history_indices, -1)
        wrong_history_categories = np.zeros_like(history_categories)
        wrong_history_event_weights = np.zeros_like(history_event_weights)
        wrong_history_mask = np.zeros_like(history_mask)

        item_id_rows: list[list[str]] = []
        for row, (request_index, selected) in enumerate(zip(indices, selections)):
            query[row] = np.asarray(
                self.query_embeddings[int(self.query_indices[request_index])],
                dtype=np.float32,
            )
            global_positions = self._candidate_global_positions(request_index)[selected]
            count = len(global_positions)
            embedding_indices = np.asarray(
                self.candidate_indices[global_positions], dtype=np.int64
            )
            candidates[row, :count] = np.asarray(
                self.item_embeddings[embedding_indices], dtype=np.float32
            )
            candidate_indices[row, :count] = embedding_indices
            ids = np.asarray(self.candidate_item_ids[global_positions], dtype=np.int64)
            candidate_item_ids[row, :count] = ids
            item_id_rows.append([str(int(value)) for value in ids])
            candidate_categories[row, :count] = np.asarray(
                self.candidate_categories[global_positions], dtype=np.int64
            )
            candidate_mask[row, :count] = True
            if self.labels is not None:
                labels[row, :count] = np.asarray(
                    self.labels[global_positions], dtype=np.float32
                )
            if self.dev_base_scores is not None:
                base_scores[row, :count] = np.asarray(
                    self.dev_base_scores[global_positions], dtype=np.float32
                )
                item_only_scores[row, :count] = np.asarray(
                    self.dev_item_only_scores[global_positions], dtype=np.float32
                )
            else:
                q = query[row]
                c = candidates[row, :count]
                cosine = (c @ q) / np.maximum(
                    np.linalg.norm(c, axis=1) * np.linalg.norm(q), 1e-6
                )
                popularity = np.asarray(
                    self.item_popularity[embedding_indices], dtype=np.float32
                )
                base_scores[row, :count] = 0.6 * _zscore(cosine) + 0.4 * _zscore(
                    popularity
                )

            self._fill_history(
                request_index,
                history[row],
                history_indices[row],
                history_categories[row],
                history_event_weights[row],
                history_mask[row],
            )
            if include_wrong_history:
                self._fill_history(
                    wrong[row],
                    wrong_history[row],
                    wrong_history_indices[row],
                    wrong_history_categories[row],
                    wrong_history_event_weights[row],
                    wrong_history_mask[row],
                )

        arrays = {
            "query": query,
            "candidates": candidates,
            "candidate_indices": candidate_indices,
            "candidate_item_ids": candidate_item_ids,
            "candidate_categories": candidate_categories,
            "candidate_mask": candidate_mask,
            "labels": labels,
            "base_scores": base_scores,
            "item_only_scores": item_only_scores,
            "history": history,
            "history_indices": history_indices,
            "history_categories": history_categories,
            "history_event_weights": history_event_weights,
            "history_mask": history_mask,
            "wrong_history": wrong_history,
            "wrong_history_indices": wrong_history_indices,
            "wrong_history_categories": wrong_history_categories,
            "wrong_history_event_weights": wrong_history_event_weights,
            "wrong_history_mask": wrong_history_mask,
        }
        batch: dict[str, Any] = {
            key: torch.from_numpy(value) for key, value in arrays.items()
        }
        batch.update(
            {
                "request_indices": indices,
                "request_ids": [self.request_ids[index] for index in indices],
                "candidate_item_id_rows": item_id_rows,
                "wrong_request_indices": wrong,
            }
        )
        return batch

    def _candidate_global_positions(self, request_index: int) -> np.ndarray:
        start = int(self.candidate_offsets[request_index])
        end = int(self.candidate_offsets[request_index + 1])
        return np.arange(start, end, dtype=np.int64)

    def _history_global_positions(self, request_index: int) -> np.ndarray:
        start = int(self.history_offsets[request_index])
        end = int(self.history_offsets[request_index + 1])
        start = max(start, end - self.max_history)
        return np.arange(start, end, dtype=np.int64)

    def _select_candidates(
        self, request_index: int, *, all_candidates: bool
    ) -> np.ndarray:
        positions = self._candidate_global_positions(request_index)
        local = np.arange(len(positions), dtype=np.int64)
        if all_candidates:
            return local
        if self.labels is None:
            raise ValueError("candidate sampling requires train labels")
        labels = np.asarray(self.labels[positions], dtype=np.uint8)
        positive = local[labels > 0]
        negative = local[labels == 0]
        if positive.size == 0:
            raise ValueError(f"train request {request_index} has no positive")
        request_id = self.request_ids[request_index]
        ranked_negative = sorted(
            (int(value) for value in negative),
            key=lambda value: hashlib.sha256(
                f"{request_id}|{int(self.candidate_item_ids[positions[value]])}|{self.seed}".encode(
                    "utf-8"
                )
            ).hexdigest(),
        )[: self.max_negatives]
        selected = np.concatenate(
            [positive, np.asarray(ranked_negative, dtype=np.int64)]
        )
        return np.sort(selected)

    def _fill_history(
        self,
        request_index: int,
        destination_embeddings: np.ndarray,
        destination_indices: np.ndarray,
        destination_categories: np.ndarray,
        destination_event_weights: np.ndarray,
        destination_mask: np.ndarray,
    ) -> None:
        positions = self._history_global_positions(request_index)
        length = len(positions)
        if length == 0:
            return
        embedding_indices = np.asarray(self.history_indices[positions], dtype=np.int64)
        destination_embeddings[:length] = np.asarray(
            self.item_embeddings[embedding_indices], dtype=np.float32
        )
        destination_indices[:length] = embedding_indices
        destination_categories[:length] = np.asarray(
            self.history_categories[positions], dtype=np.int64
        )
        destination_event_weights[:length] = np.asarray(
            self.history_event_weights[positions], dtype=np.float32
        )
        destination_mask[:length] = True


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def request_batches(
    indices: Iterable[int], batch_size: int, *, shuffle: bool, seed: int
) -> Iterator[list[int]]:
    values = np.asarray(list(indices), dtype=np.int64)
    if shuffle:
        generator = np.random.default_rng(seed)
        generator.shuffle(values)
    for start in range(0, len(values), batch_size):
        yield [int(value) for value in values[start : start + batch_size]]
