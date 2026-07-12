"""Label-safe selection, all-candidate batching, and frozen C05 feature access."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import yaml
from torch.nn import functional as F


FORBIDDEN_TRAIN_BASENAMES = {
    "qrels_dev.jsonl",
    "qrels_test.jsonl",
    "records_dev.jsonl",
    "records_test.jsonl",
    "metrics.json",
    "per_request_metrics.jsonl",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config.get("candidate_id") != "c05":
        raise ValueError("configuration is not C05")
    if int(config.get("seed", -1)) != 20260708:
        raise ValueError("unexpected C05 seed")
    if int(config.get("physical_gpu", -1)) != 0:
        raise ValueError("C05 G0/G2a is assigned to physical GPU 0")
    if not str(config.get("run_id", "")).startswith(
        "20260711_kuaisearch_c05_"
    ):
        raise ValueError("invalid C05 run prefix")
    if not str(config.get("g0_run_id", "")).startswith(
        "20260711_kuaisearch_c05_"
    ):
        raise ValueError("invalid C05 G0 run prefix")
    assert_training_path_firewall(config)
    return config


def assert_training_path_firewall(config: dict[str, Any]) -> None:
    for name, raw in config.get("paths", {}).items():
        path = Path(str(raw))
        if path.name in FORBIDDEN_TRAIN_BASENAMES:
            raise ValueError(f"forbidden C05 training path {name}: {path}")
        lowered = str(path).lower()
        if "qrels" in lowered or lowered.endswith("/metrics.json"):
            raise ValueError(f"forbidden C05 training path {name}: {path}")
    checkpoint = str(config["paths"]["calibration_checkpoint"])
    if "_final_" in checkpoint:
        raise ValueError("G0/G2a must not use the final D2 checkpoint")
    popularity = str(config["paths"]["internal_train_popularity"])
    if "full_train" in popularity:
        raise ValueError("G0/G2a must not use full-train popularity")


def assert_candidate_manifest(config: dict[str, Any]) -> str:
    actual = sha256_file(config["paths"]["candidate_manifest"])
    expected = str(config["paths"]["candidate_manifest_sha256"])
    if actual != expected:
        raise ValueError(f"candidate manifest hash mismatch: {actual} != {expected}")
    return actual


def assert_proposal_lock(config: dict[str, Any]) -> str:
    """Verify the review-amended pre-outcome source/design lock."""

    lock_path = Path(config["paths"]["proposal_lock"])
    lock = read_json(lock_path)
    if lock.get("lock_id") != "c05_review1_g0_g2a_preoutcome":
        raise ValueError("unexpected C05 proposal lock")
    if lock.get("status") != "locked_before_any_c05_data_fit_outcome":
        raise ValueError("C05 proposal is not pre-outcome locked")
    if lock.get("outcomes_observed_before_lock") is not False:
        raise ValueError("C05 lock does not make the required pre-outcome declaration")
    source_root = Path(config["paths"]["candidate_source_root"])
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("C05 proposal lock has no file manifest")
    for relative, expected in files.items():
        actual = sha256_file(source_root / relative)
        if actual != expected:
            raise ValueError(
                f"C05 locked file changed: {relative}: {actual} != {expected}"
            )
    for relative, expected in lock.get("repo_files", {}).items():
        actual = sha256_file(Path(relative))
        if actual != expected:
            raise ValueError(
                f"C05 locked shared file changed: {relative}: {actual} != {expected}"
            )
    return sha256_file(lock_path)


def validate_gpu(device: str) -> None:
    if device != "cuda:0":
        raise ValueError("C05 code must address the sole visible GPU as cuda:0")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C05 must see exactly one CUDA device")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("C05 must be launched with CUDA_VISIBLE_DEVICES=0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("deterministic CUBLAS_WORKSPACE_CONFIG is required")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def request_ids(path: str | Path) -> list[str]:
    rows: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(str(json.loads(line)["request_id"]))
    return rows


def selected_candidate_key_sha256(data: Any, indices: Sequence[int]) -> str:
    """Hash ordered request/candidate identities for an exact eval cohort."""

    digest = hashlib.sha256()
    for raw_index in indices:
        index = int(raw_index)
        start = int(data.candidate_offsets[index])
        end = int(data.candidate_offsets[index + 1])
        payload = json.dumps(
            [
                str(data.request_ids[index]),
                [str(value) for value in data.candidate_item_ids[start:end]],
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(payload).to_bytes(8, byteorder="big"))
        digest.update(payload)
    return digest.hexdigest()


def _stable_key(seed: int, split: str, request_id: str) -> str:
    payload = f"c05_g2a:{seed}:{split}:{request_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _nonrepeat_indices(
    packed_train: Path, start: int, stop: int
) -> list[int]:
    candidate_offsets = np.load(
        packed_train / "candidate_offsets.npy", mmap_mode="r"
    )
    candidate_indices = np.load(
        packed_train / "candidate_embedding_indices.npy", mmap_mode="r"
    )
    history_offsets = np.load(packed_train / "history_offsets.npy", mmap_mode="r")
    history_indices = np.load(
        packed_train / "history_embedding_indices.npy", mmap_mode="r"
    )
    result: list[int] = []
    for index in range(start, stop):
        hs = int(history_offsets[index])
        he = int(history_offsets[index + 1])
        if hs == he:
            continue
        cs = int(candidate_offsets[index])
        ce = int(candidate_offsets[index + 1])
        if np.intersect1d(
            np.asarray(candidate_indices[cs:ce]),
            np.asarray(history_indices[hs:he]),
            assume_unique=False,
        ).size:
            continue
        result.append(index)
    return result


def freeze_selection(config: dict[str, Any]) -> dict[str, Any]:
    """Freeze request IDs before any label-shaped array is opened."""

    packed_root = Path(config["paths"]["packed_train_root"])
    train_root = packed_root / "train"
    ids = request_ids(train_root / "request_ids.jsonl")
    expected_requests = int(config["integrity"]["packed_train_requests"])
    if len(ids) != expected_requests:
        raise ValueError(f"packed request count mismatch: {len(ids)}")
    if len(set(ids)) != len(ids):
        raise ValueError("packed train request IDs are not unique")
    selection = config["selection"]
    fit_start, fit_stop = (int(value) for value in selection["fit_range"])
    internal_start, internal_stop = (
        int(value) for value in selection["internal_range"]
    )
    cut = int(config["integrity"]["packed_cut_request_index"])
    if (fit_start, fit_stop, internal_start, internal_stop) != (
        0,
        cut,
        cut,
        expected_requests,
    ):
        raise ValueError("selection ranges do not match the frozen D2 boundary")

    fit_pool = _nonrepeat_indices(train_root, fit_start, fit_stop)
    internal_pool = _nonrepeat_indices(train_root, internal_start, internal_stop)
    seed = int(config["seed"])

    def select(pool: Sequence[int], count: int, name: str) -> list[int]:
        ranked = sorted(
            pool,
            key=lambda index: (_stable_key(seed, name, ids[index]), ids[index]),
        )
        if len(ranked) < count:
            raise ValueError(f"not enough {name} non-repeat requests")
        return sorted(ranked[:count])

    fit = select(fit_pool, int(selection["fit_nonrepeat_requests"]), "fit")
    internal = select(
        internal_pool,
        int(selection["internal_nonrepeat_requests"]),
        "internal",
    )
    if set(fit) & set(internal):
        raise AssertionError("fit/internal selections overlap")
    output = {
        "candidate_id": config["candidate_id"],
        "labels_opened_before_selection": False,
        "packed_request_ids_path": str(train_root / "request_ids.jsonl"),
        "packed_request_ids_sha256": sha256_file(
            train_root / "request_ids.jsonl"
        ),
        "pool_counts": {
            "fit_nonrepeat_history_present": len(fit_pool),
            "internal_nonrepeat_history_present": len(internal_pool),
        },
        "rule": (
            "lowest sha256(c05_g2a, seed, split, request_id), then store in "
            "packed index order"
        ),
        "seed": seed,
        "fit": {"indices": fit, "request_ids": [ids[index] for index in fit]},
        "internal": {
            "indices": internal,
            "request_ids": [ids[index] for index in internal],
        },
    }
    path = Path(config["paths"]["artifact_root"]) / "selection.json"
    write_json(path, output)
    output["path"] = str(path)
    output["sha256"] = sha256_file(path)
    return output


def iter_request_batches(
    data: Any,
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
        candidates = int(
            data.candidate_offsets[index + 1] - data.candidate_offsets[index]
        )
        full_history = int(
            data.history_offsets[index + 1] - data.history_offsets[index]
        )
        history = min(full_history, history_limit)
        next_size = len(batch) + 1
        next_candidates = max(max_candidates, candidates)
        next_history = max(max_history, max(history, 1))
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
        max_candidates = max(max_candidates, candidates)
        max_history = max(max_history, max(history, 1))
    if batch:
        yield np.asarray(batch, dtype=np.int64)


class FrozenG2AFeatures:
    """Read-only G0 outputs plus frozen raw item states."""

    def __init__(self, config: dict[str, Any], selection: dict[str, Any]) -> None:
        root = Path(config["paths"]["artifact_root"])
        self.selected_indices = np.load(root / "selected_request_indices.npy")
        expected = np.asarray(
            selection["fit"]["indices"] + selection["internal"]["indices"],
            dtype=np.int64,
        )
        if not np.array_equal(self.selected_indices, expected):
            raise ValueError("G0 selected-request array differs from selection lock")
        self.position = {
            int(request_index): row
            for row, request_index in enumerate(self.selected_indices)
        }
        self.query = np.load(root / "query_embeddings.npy", mmap_mode="r")
        self.base_offsets = np.load(
            root / "selected_candidate_offsets.npy", mmap_mode="r"
        )
        self.base_scores = np.load(root / "base_scores.npy", mmap_mode="r")
        self.raw_items = np.load(
            config["paths"]["raw_item_embeddings"], mmap_mode="r"
        )
        self.item_adapter = np.load(root / "item_adapter_weight.npy", mmap_mode="r")
        if len(self.query) != len(self.selected_indices):
            raise ValueError("query-state row mismatch")
        if len(self.base_offsets) != len(self.selected_indices) + 1:
            raise ValueError("base offset row mismatch")
        if int(self.base_offsets[-1]) != len(self.base_scores):
            raise ValueError("base score row mismatch")
        self._adapter_by_device: dict[str, torch.Tensor] = {}

    def tensors(
        self, data: Any, batch: dict[str, np.ndarray], device: str
    ) -> dict[str, torch.Tensor]:
        request_indices = np.asarray(batch["request_indices"], dtype=np.int64)
        positions = np.asarray(
            [self.position[int(index)] for index in request_indices], dtype=np.int64
        )
        query = torch.from_numpy(
            np.asarray(self.query[positions], dtype=np.float32).copy()
        ).to(device)
        candidates = self._adapt(batch["candidate_indices"], device)
        history = self._adapt(batch["history_indices"], device)
        max_candidates = batch["candidate_indices"].shape[1]
        base = np.zeros((len(positions), max_candidates), dtype=np.float32)
        for row, position in enumerate(positions):
            start = int(self.base_offsets[position])
            end = int(self.base_offsets[position + 1])
            count = end - start
            base[row, :count] = self.base_scores[start:end]
        return {
            "query": query,
            "candidates": candidates,
            "history": history,
            "candidate_mask": torch.from_numpy(batch["candidate_mask"]).to(device),
            "history_mask": torch.from_numpy(batch["history_mask"]).to(device),
            "history_event_weights": torch.from_numpy(
                batch["history_event_weights"]
            ).to(device),
            "base_scores": torch.from_numpy(base).to(device),
            "labels": torch.from_numpy(batch["candidate_labels"]).to(device),
        }

    def _adapt(self, indices: np.ndarray, device: str) -> torch.Tensor:
        raw = torch.from_numpy(
            np.asarray(self.raw_items[np.asarray(indices, dtype=np.int64)], dtype=np.float32).copy()
        ).to(device)
        if device not in self._adapter_by_device:
            self._adapter_by_device[device] = torch.from_numpy(
                np.asarray(self.item_adapter, dtype=np.float32).copy()
            ).to(device)
        return F.normalize(
            F.linear(raw, self._adapter_by_device[device]), dim=-1, eps=1e-6
        )


def collate_g2a(
    data: Any, request_indices: np.ndarray, *, history_limit: int
) -> dict[str, np.ndarray]:
    batch = len(request_indices)
    candidate_counts = [
        int(data.candidate_offsets[index + 1] - data.candidate_offsets[index])
        for index in request_indices
    ]
    full_history_counts = [
        int(data.history_offsets[index + 1] - data.history_offsets[index])
        for index in request_indices
    ]
    max_candidates = max(candidate_counts)
    max_history = max(1, min(max(full_history_counts), history_limit))
    candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
    candidate_item_ids = np.full((batch, max_candidates), -1, dtype=np.int64)
    candidate_labels = np.zeros((batch, max_candidates), dtype=np.float32)
    candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
    history_indices = np.zeros((batch, max_history), dtype=np.int64)
    history_event_weights = np.zeros((batch, max_history), dtype=np.float32)
    history_mask = np.zeros((batch, max_history), dtype=bool)

    for row, raw_index in enumerate(request_indices):
        index = int(raw_index)
        cs = int(data.candidate_offsets[index])
        ce = int(data.candidate_offsets[index + 1])
        candidate_count = ce - cs
        candidate_indices[row, :candidate_count] = data.candidate_embedding_indices[cs:ce]
        candidate_item_ids[row, :candidate_count] = data.candidate_item_ids[cs:ce]
        candidate_labels[row, :candidate_count] = data.candidate_labels[cs:ce]
        candidate_mask[row, :candidate_count] = True

        hs = int(data.history_offsets[index])
        he = int(data.history_offsets[index + 1])
        start = max(hs, he - history_limit)
        history_count = he - start
        if history_count:
            history_indices[row, :history_count] = data.history_embedding_indices[start:he]
            event_multiplier = np.asarray(
                data.history_event_weights[start:he], dtype=np.float32
            )
            reverse_age = np.arange(history_count, 0, -1, dtype=np.float32)
            history_event_weights[row, :history_count] = (
                event_multiplier / np.sqrt(reverse_age)
            )
            history_mask[row, :history_count] = True
    return {
        "request_indices": np.asarray(request_indices, dtype=np.int64),
        "candidate_indices": candidate_indices,
        "candidate_item_ids": candidate_item_ids,
        "candidate_labels": candidate_labels,
        "candidate_mask": candidate_mask,
        "history_indices": history_indices,
        "history_event_weights": history_event_weights,
        "history_mask": history_mask,
    }
