"""Supervised query and history-residual diagnostics for PPS motivation."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from myrec.baselines.core import recent_behavior_scores
from myrec.eval.metrics import ScoredCandidate, request_metrics
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


VARIANTS = {"d1q", "d1m", "d1a"}


@dataclass
class PackedRequestData:
    root: Path
    split: str
    request_ids: list[str]
    query_indices: np.ndarray
    timestamps: np.ndarray
    candidate_offsets: np.ndarray
    candidate_embedding_indices: np.ndarray
    candidate_item_ids: np.ndarray
    candidate_labels: np.ndarray
    candidate_purchase_labels: np.ndarray
    candidate_b0b_scores: np.ndarray
    history_offsets: np.ndarray
    history_embedding_indices: np.ndarray
    history_event_weights: np.ndarray

    @classmethod
    def load(cls, root: str | Path, split: str) -> "PackedRequestData":
        root = Path(root)
        prefix = root / split
        request_ids = [
            str(row["request_id"]) for row in iter_jsonl(prefix / "request_ids.jsonl")
        ]
        data = cls(
            root=root,
            split=split,
            request_ids=request_ids,
            query_indices=np.load(prefix / "query_indices.npy", mmap_mode="r"),
            timestamps=np.load(prefix / "timestamps.npy", mmap_mode="r"),
            candidate_offsets=np.load(prefix / "candidate_offsets.npy", mmap_mode="r"),
            candidate_embedding_indices=np.load(
                prefix / "candidate_embedding_indices.npy", mmap_mode="r"
            ),
            candidate_item_ids=np.load(prefix / "candidate_item_ids.npy", mmap_mode="r"),
            candidate_labels=np.load(prefix / "candidate_labels.npy", mmap_mode="r"),
            candidate_purchase_labels=np.load(
                prefix / "candidate_purchase_labels.npy", mmap_mode="r"
            ),
            candidate_b0b_scores=np.load(
                prefix / "candidate_b0b_scores.npy", mmap_mode="r"
            ),
            history_offsets=np.load(prefix / "history_offsets.npy", mmap_mode="r"),
            history_embedding_indices=np.load(
                prefix / "history_embedding_indices.npy", mmap_mode="r"
            ),
            history_event_weights=np.load(
                prefix / "history_event_weights.npy", mmap_mode="r"
            ),
        )
        data.validate()
        return data

    def validate(self) -> None:
        requests = len(self.request_ids)
        if len(self.query_indices) != requests or len(self.timestamps) != requests:
            raise ValueError(f"request array length mismatch for {self.split}")
        if len(self.candidate_offsets) != requests + 1:
            raise ValueError(f"candidate offsets mismatch for {self.split}")
        if len(self.history_offsets) != requests + 1:
            raise ValueError(f"history offsets mismatch for {self.split}")
        candidate_rows = int(self.candidate_offsets[-1])
        for values in [
            self.candidate_embedding_indices,
            self.candidate_item_ids,
            self.candidate_labels,
            self.candidate_purchase_labels,
            self.candidate_b0b_scores,
        ]:
            if len(values) != candidate_rows:
                raise ValueError(f"candidate row mismatch for {self.split}")
        history_rows = int(self.history_offsets[-1])
        if len(self.history_embedding_indices) != history_rows:
            raise ValueError(f"history row mismatch for {self.split}")
        if len(self.history_event_weights) != history_rows:
            raise ValueError(f"history weight mismatch for {self.split}")

    def __len__(self) -> int:
        return len(self.request_ids)


def materialize_supervised_diagnostic_data(
    config: dict[str, Any],
    config_path: str | Path,
) -> dict[str, Any]:
    validate_config(config)
    standardized_dir = Path(config["standardized_dir"])
    embedding_root = Path(config["embedding_artifact"]["root"])
    output_dir = Path(config["materialized_data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(config_path)

    with (embedding_root / config["embedding_artifact"]["query_index"]).open(
        "r", encoding="utf-8"
    ) as handle:
        query_index = json.load(handle)
    with (embedding_root / config["embedding_artifact"]["item_index"]).open(
        "r", encoding="utf-8"
    ) as handle:
        item_index = json.load(handle)

    split_manifests = {}
    for split in config["materialized_data"]["source_splits"]:
        split_manifests[split] = _materialize_split(
            records_path=standardized_dir / f"records_{split}.jsonl",
            output_dir=output_dir / split,
            split=split,
            query_index=query_index,
            item_index=item_index,
            history_limit=int(config["materialized_data"]["history_limit"]),
        )

    train = PackedRequestData.load(output_dir, "train")
    internal_cut = int(
        math.floor(len(train) * float(config["internal_calibration"]["train_fraction"]))
    )
    if not 0 < internal_cut < len(train):
        raise ValueError("invalid internal calibration split")
    item_count = len(item_index)
    popularity_files = {}
    for name, request_indices in {
        "internal_train": np.arange(internal_cut, dtype=np.int64),
        "full_train": np.arange(len(train), dtype=np.int64),
    }.items():
        values = compute_item_log_click_counts(train, request_indices, item_count)
        path = output_dir / f"item_log_click_{name}.npy"
        np.save(path, values)
        popularity_files[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "nonzero_items": int(np.count_nonzero(values)),
        }

    files = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path != output_dir / "manifest.json":
            files[str(path.relative_to(output_dir))] = {
                "path": str(path),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    manifest = {
        "analysis_id": config["analysis_id"],
        "candidate_manifest": {
            "path": str(standardized_dir / "candidate_manifest.json"),
            "sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        },
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "embedding_files": {
            name: {
                "path": str(embedding_root / config["embedding_artifact"][name]),
                "sha256": sha256_file(
                    embedding_root / config["embedding_artifact"][name]
                ),
            }
            for name in [
                "query_embeddings",
                "item_embeddings",
                "query_index",
                "item_index",
            ]
        },
        "files": files,
        "internal_calibration": {
            "cut_request_index": internal_cut,
            "internal_train_requests": internal_cut,
            "internal_validation_requests": len(train) - internal_cut,
            "rule": config["internal_calibration"]["split"],
        },
        "item_count": item_count,
        "popularity_files": popularity_files,
        "qrels_read": False,
        "source_files": {
            split: {
                "path": str(standardized_dir / f"records_{split}.jsonl"),
                "sha256": sha256_file(standardized_dir / f"records_{split}.jsonl"),
            }
            for split in config["materialized_data"]["source_splits"]
        },
        "splits": split_manifests,
        "status": "passed",
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _materialize_split(
    records_path: Path,
    output_dir: Path,
    split: str,
    query_index: dict[str, int],
    item_index: dict[str, int],
    history_limit: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    request_ids = []
    query_indices = []
    timestamps = []
    candidate_offsets = [0]
    candidate_embedding_indices = []
    candidate_item_ids = []
    candidate_labels = []
    candidate_purchase_labels = []
    candidate_b0b_scores = []
    history_offsets = [0]
    history_embedding_indices = []
    history_event_weights = []
    missing_query = 0
    missing_candidate = 0
    missing_history = 0
    nonnumeric_item_ids = 0
    skipped_train_without_positive = 0

    request_ids_path = output_dir / "request_ids.jsonl"
    with request_ids_path.open("w", encoding="utf-8") as request_handle:
        for record in iter_jsonl(records_path):
            if split == "train" and not any(
                int(candidate.get("clicked", 0) or 0) > 0
                for candidate in record["candidates"]
            ):
                skipped_train_without_positive += 1
                continue
            request_id = str(record["request_id"])
            if request_id not in query_index:
                missing_query += 1
                continue
            request_ids.append(request_id)
            request_handle.write(
                json.dumps({"request_id": request_id}, sort_keys=True) + "\n"
            )
            query_indices.append(int(query_index[request_id]))
            timestamps.append(int(record["ts"]))

            b0b_scores = recent_behavior_scores(record)
            for candidate in record["candidates"]:
                item_id = str(candidate["item_id"])
                if item_id not in item_index:
                    missing_candidate += 1
                    raise ValueError(f"candidate embedding missing for {item_id}")
                try:
                    numeric_item_id = int(item_id)
                except ValueError as error:
                    nonnumeric_item_ids += 1
                    raise ValueError(f"non-numeric item id {item_id}") from error
                candidate_embedding_indices.append(int(item_index[item_id]))
                candidate_item_ids.append(numeric_item_id)
                candidate_labels.append(int(candidate.get("clicked", 0) or 0))
                candidate_purchase_labels.append(int(candidate.get("purchased", 0) or 0))
                candidate_b0b_scores.append(float(b0b_scores[item_id]))
            candidate_offsets.append(len(candidate_embedding_indices))

            history = list(record.get("history") or [])[-history_limit:]
            for event in history:
                item_id = str(event["item_id"])
                if item_id not in item_index:
                    missing_history += 1
                    raise ValueError(f"history embedding missing for {item_id}")
                history_embedding_indices.append(int(item_index[item_id]))
                history_event_weights.append(
                    1.5 if str(event.get("event") or "click") == "purchase" else 1.0
                )
            history_offsets.append(len(history_embedding_indices))

    if not request_ids:
        raise ValueError(f"no records materialized for {split}")
    if split == "train" and not all(
        sum(candidate_labels[start:end]) > 0
        for start, end in zip(candidate_offsets[:-1], candidate_offsets[1:])
    ):
        raise AssertionError("train request without clicked positive survived filtering")

    arrays = {
        "query_indices": np.asarray(query_indices, dtype=np.int32),
        "timestamps": np.asarray(timestamps, dtype=np.int64),
        "candidate_offsets": np.asarray(candidate_offsets, dtype=np.int64),
        "candidate_embedding_indices": np.asarray(
            candidate_embedding_indices, dtype=np.int32
        ),
        "candidate_item_ids": np.asarray(candidate_item_ids, dtype=np.int64),
        "candidate_labels": np.asarray(candidate_labels, dtype=np.uint8),
        "candidate_purchase_labels": np.asarray(
            candidate_purchase_labels, dtype=np.uint8
        ),
        "candidate_b0b_scores": np.asarray(candidate_b0b_scores, dtype=np.float32),
        "history_offsets": np.asarray(history_offsets, dtype=np.int64),
        "history_embedding_indices": np.asarray(
            history_embedding_indices, dtype=np.int32
        ),
        "history_event_weights": np.asarray(history_event_weights, dtype=np.float16),
    }
    for name, values in arrays.items():
        np.save(output_dir / f"{name}.npy", values)

    candidate_counts = np.diff(arrays["candidate_offsets"])
    history_counts = np.diff(arrays["history_offsets"])
    return {
        "candidate_count_max": int(candidate_counts.max()),
        "candidate_count_mean": float(candidate_counts.mean()),
        "candidate_rows": int(len(candidate_embedding_indices)),
        "clicked_rows": int(sum(candidate_labels)),
        "history_count_max": int(history_counts.max()),
        "history_count_mean": float(history_counts.mean()),
        "history_present_requests": int(np.count_nonzero(history_counts)),
        "history_rows": int(len(history_embedding_indices)),
        "missing_candidate_embeddings": missing_candidate,
        "missing_history_embeddings": missing_history,
        "missing_query_embeddings": missing_query,
        "nonnumeric_item_ids": nonnumeric_item_ids,
        "purchase_rows": int(sum(candidate_purchase_labels)),
        "requests": len(request_ids),
        "skipped_train_requests_without_clicked_positive": skipped_train_without_positive,
        "split": split,
    }


def compute_item_log_click_counts(
    data: PackedRequestData,
    request_indices: np.ndarray,
    item_count: int,
) -> np.ndarray:
    counts = np.zeros(item_count, dtype=np.float32)
    for request_index in request_indices:
        start = int(data.candidate_offsets[request_index])
        end = int(data.candidate_offsets[request_index + 1])
        labels = np.asarray(data.candidate_labels[start:end], dtype=bool)
        if labels.any():
            np.add.at(
                counts,
                np.asarray(data.candidate_embedding_indices[start:end])[labels],
                1.0,
            )
    np.log1p(counts, out=counts)
    return counts


def iter_request_batches(
    data: PackedRequestData,
    request_indices: np.ndarray,
    batch_size: int,
    max_padded_candidates: int,
    max_padded_history: int,
    seed: int,
    shuffle: bool,
) -> Iterator[dict[str, np.ndarray]]:
    indices = np.asarray(request_indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    batch: list[int] = []
    max_candidates = 0
    max_history = 0
    for request_index in indices:
        request_index = int(request_index)
        candidate_count = int(
            data.candidate_offsets[request_index + 1]
            - data.candidate_offsets[request_index]
        )
        history_count = max(
            1,
            int(
                data.history_offsets[request_index + 1]
                - data.history_offsets[request_index]
            ),
        )
        next_size = len(batch) + 1
        next_candidates = max(max_candidates, candidate_count)
        next_history = max(max_history, history_count)
        exceeds = (
            next_size > batch_size
            or next_size * next_candidates > max_padded_candidates
            or next_size * next_history > max_padded_history
        )
        if batch and exceeds:
            yield collate_request_batch(data, np.asarray(batch, dtype=np.int64))
            batch = []
            max_candidates = 0
            max_history = 0
        batch.append(request_index)
        max_candidates = max(max_candidates, candidate_count)
        max_history = max(max_history, history_count)
    if batch:
        yield collate_request_batch(data, np.asarray(batch, dtype=np.int64))


def collate_request_batch(
    data: PackedRequestData,
    request_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate_counts = np.asarray(
        [
            int(data.candidate_offsets[index + 1] - data.candidate_offsets[index])
            for index in request_indices
        ],
        dtype=np.int32,
    )
    history_counts = np.asarray(
        [
            int(data.history_offsets[index + 1] - data.history_offsets[index])
            for index in request_indices
        ],
        dtype=np.int32,
    )
    max_candidates = int(candidate_counts.max())
    max_history = max(1, int(history_counts.max()))
    batch = len(request_indices)
    candidate_indices = np.zeros((batch, max_candidates), dtype=np.int64)
    candidate_item_ids = np.zeros((batch, max_candidates), dtype=np.int64)
    candidate_labels = np.zeros((batch, max_candidates), dtype=np.float32)
    candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
    candidate_b0b = np.zeros((batch, max_candidates), dtype=np.float32)
    history_indices = np.zeros((batch, max_history), dtype=np.int64)
    history_event_weights = np.zeros((batch, max_history), dtype=np.float32)
    history_mask = np.zeros((batch, max_history), dtype=bool)

    for row, request_index in enumerate(request_indices):
        c_start = int(data.candidate_offsets[request_index])
        c_end = int(data.candidate_offsets[request_index + 1])
        c_count = c_end - c_start
        candidate_indices[row, :c_count] = data.candidate_embedding_indices[c_start:c_end]
        candidate_item_ids[row, :c_count] = data.candidate_item_ids[c_start:c_end]
        candidate_labels[row, :c_count] = data.candidate_labels[c_start:c_end]
        candidate_b0b[row, :c_count] = data.candidate_b0b_scores[c_start:c_end]
        candidate_mask[row, :c_count] = True

        h_start = int(data.history_offsets[request_index])
        h_end = int(data.history_offsets[request_index + 1])
        h_count = h_end - h_start
        if h_count:
            history_indices[row, :h_count] = data.history_embedding_indices[h_start:h_end]
            event_weights = np.asarray(
                data.history_event_weights[h_start:h_end], dtype=np.float32
            )
            reverse_ages = np.arange(h_count, 0, -1, dtype=np.float32)
            history_event_weights[row, :h_count] = event_weights / np.sqrt(reverse_ages)
            history_mask[row, :h_count] = True

    return {
        "request_indices": request_indices,
        "query_indices": np.asarray(data.query_indices[request_indices], dtype=np.int64),
        "candidate_indices": candidate_indices,
        "candidate_item_ids": candidate_item_ids,
        "candidate_labels": candidate_labels,
        "candidate_mask": candidate_mask,
        "candidate_b0b": candidate_b0b,
        "history_indices": history_indices,
        "history_event_weights": history_event_weights,
        "history_mask": history_mask,
    }


class SupervisedDiagnosticRanker(nn.Module):
    def __init__(
        self,
        query_embeddings: torch.Tensor,
        item_embeddings: torch.Tensor,
        item_log_click: torch.Tensor,
        projection_dim: int,
        dropout: float,
        variant: str,
    ) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"unknown variant {variant}")
        self.variant = variant
        self.query_embeddings = query_embeddings
        self.item_embeddings = item_embeddings
        self.item_log_click = item_log_click
        embedding_dim = int(query_embeddings.shape[1])
        if int(item_embeddings.shape[1]) != embedding_dim:
            raise ValueError("query/item embedding dimensions differ")
        self.base_q_proj = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.base_i_proj = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.att_q_proj = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.hist_key_proj = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.hist_value_proj = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.res_i_proj = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.base_weights = nn.Parameter(torch.tensor([1.0, 0.25, 0.10]))
        self.residual_weights = nn.Parameter(torch.tensor([0.50, 0.10]))
        self.projection_dim = projection_dim

    def freeze_base(self) -> None:
        for module in [self.base_q_proj, self.base_i_proj]:
            for parameter in module.parameters():
                parameter.requires_grad = False
        self.base_weights.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        query = safe_normalize(
            self.query_embeddings[batch["query_indices"]].float()
        )
        candidates = safe_normalize(
            self.item_embeddings[batch["candidate_indices"]].float()
        )
        candidate_mask = batch["candidate_mask"]

        cosine = torch.einsum("bd,bcd->bc", query, candidates)
        query_low = safe_normalize(self.base_q_proj(self.dropout(query)))
        candidate_low = safe_normalize(
            self.base_i_proj(self.dropout(candidates))
        )
        learned = torch.einsum("bd,bcd->bc", query_low, candidate_low)
        popularity = self.item_log_click[batch["candidate_indices"]]
        base_features = torch.stack(
            [
                masked_zscore(cosine, candidate_mask),
                masked_zscore(popularity, candidate_mask),
                masked_zscore(learned, candidate_mask),
            ],
            dim=-1,
        )
        scores = torch.einsum("bcf,f->bc", base_features, self.base_weights)
        if self.variant == "d1q":
            return scores.masked_fill(~candidate_mask, _mask_value(scores))

        history_mask = batch["history_mask"]
        history_present = history_mask.any(dim=1)
        history = safe_normalize(
            self.item_embeddings[batch["history_indices"]].float()
        )
        history_values = self.hist_value_proj(self.dropout(history))
        prior_weights = batch["history_event_weights"].masked_fill(~history_mask, 0.0)
        if self.variant == "d1m":
            weights = prior_weights / prior_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            query_key = self.att_q_proj(self.dropout(query))
            history_keys = self.hist_key_proj(self.dropout(history))
            logits = torch.einsum("bd,bhd->bh", query_key, history_keys)
            logits = logits / math.sqrt(self.projection_dim)
            logits = logits + torch.log(prior_weights.clamp_min(1e-8))
            logits = logits.masked_fill(~history_mask, _mask_value(logits))
            weights = torch.softmax(logits, dim=1)
            weights = torch.where(history_present[:, None], weights, torch.zeros_like(weights))
        summary = torch.einsum("bh,bhd->bd", weights, history_values)
        summary = safe_normalize(summary)
        residual_items = safe_normalize(self.res_i_proj(self.dropout(candidates)))
        residual_interaction = torch.einsum("bd,bcd->bc", summary, residual_items)
        b0b = batch["candidate_b0b"]
        residual_features = torch.stack(
            [
                masked_zscore(b0b, candidate_mask),
                masked_zscore(residual_interaction, candidate_mask),
            ],
            dim=-1,
        )
        residual = torch.einsum(
            "bcf,f->bc", residual_features, self.residual_weights
        )
        residual = residual * history_present[:, None]
        combined = scores + residual
        return combined.masked_fill(~candidate_mask, _mask_value(combined))


def masked_zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_float = mask.to(values.dtype)
    count = mask_float.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (values * mask_float).sum(dim=1, keepdim=True) / count
    variance = (((values - mean) ** 2) * mask_float).sum(dim=1, keepdim=True) / count
    normalized = (values - mean) / (variance + 1e-6).sqrt()
    return normalized.masked_fill(~mask, 0.0)


def safe_normalize(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values, dim=-1, eps=1e-6)


def multi_positive_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    positives = (labels > 0) & candidate_mask
    missing = int((~positives.any(dim=1)).sum().item())
    if missing:
        raise ValueError(f"batch contains {missing} requests without positives")
    mask_value = _mask_value(scores)
    all_logsumexp = torch.logsumexp(
        scores.masked_fill(~candidate_mask, mask_value), dim=1
    )
    positive_logsumexp = torch.logsumexp(
        scores.masked_fill(~positives, mask_value), dim=1
    )
    return (all_logsumexp - positive_logsumexp).mean()


def _mask_value(values: torch.Tensor) -> float:
    return -float(torch.finfo(values.dtype).max)


def train_diagnostic(
    config: dict[str, Any],
    config_path: str | Path,
    variant: str,
    seed: int,
    mode: str,
    device: str,
    base_checkpoint: str | Path | None = None,
    epochs: int | None = None,
) -> dict[str, Any]:
    validate_config(config)
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    if mode not in {"calibrate", "final"}:
        raise ValueError("mode must be calibrate or final")
    if variant != "d1q" and base_checkpoint is None:
        raise ValueError("residual variants require a base checkpoint")
    set_reproducible_seed(seed)
    started = time.time()
    data_root = Path(config["materialized_data"]["output_dir"])
    train_data = PackedRequestData.load(data_root, "train")
    manifest = _read_json(data_root / "manifest.json")
    cut = int(manifest["internal_calibration"]["cut_request_index"])
    if mode == "calibrate":
        train_indices = np.arange(cut, dtype=np.int64)
        validation_indices = np.arange(cut, len(train_data), dtype=np.int64)
        popularity_path = data_root / "item_log_click_internal_train.npy"
        max_epochs = int(config["internal_calibration"]["max_epochs"])
    else:
        train_indices = np.arange(len(train_data), dtype=np.int64)
        validation_indices = None
        popularity_path = data_root / "item_log_click_full_train.npy"
        if epochs is None or epochs < 1:
            raise ValueError("final mode requires positive epochs")
        max_epochs = int(epochs)

    model = build_model(config, variant, popularity_path, device)
    if base_checkpoint is not None:
        checkpoint = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.variant = variant
        model.freeze_base()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config["model"]["learning_rate"]),
        weight_decay=float(config["model"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    amp_dtype = torch.float16
    batch_size = int(config["model"]["batch_size_requests"])
    max_padded_candidates = int(
        config["model"]["max_padded_candidates_per_batch"]
    )
    max_padded_history = int(
        config["model"]["max_padded_history_events_per_batch"]
    )
    patience = int(config["internal_calibration"]["patience"])
    min_delta = float(config["internal_calibration"]["min_delta"])
    best_metric = -math.inf
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    epoch_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for numpy_batch in iter_request_batches(
            train_data,
            train_indices,
            batch_size=batch_size,
            max_padded_candidates=max_padded_candidates,
            max_padded_history=max_padded_history,
            seed=seed + epoch,
            shuffle=True,
        ):
            batch = batch_to_device(numpy_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda" if device.startswith("cuda") else "cpu",
                dtype=amp_dtype,
                enabled=device.startswith("cuda"),
            ):
                scores = model(batch)
                loss = multi_positive_listwise_loss(
                    scores, batch["candidate_labels"], batch["candidate_mask"]
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                trainable, float(config["model"]["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach().cpu())
            batches += 1

        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, batches),
        }
        if validation_indices is not None:
            validation_metric = evaluate_internal_ndcg(
                model,
                train_data,
                validation_indices,
                batch_size=batch_size,
                max_padded_candidates=max_padded_candidates,
                max_padded_history=max_padded_history,
                device=device,
            )
            row["internal_validation_ndcg@10"] = validation_metric
            if validation_metric > best_metric + min_delta:
                best_metric = validation_metric
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        epoch_rows.append(row)
        if validation_indices is not None and epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    run_id = diagnostic_training_run_id(variant, mode, seed)
    model_dir = Path("models/diagnostics") / run_id
    run_dir = Path("runs") / run_id
    model_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "model.pt"
    torch.save(
        {
            "analysis_id": config["analysis_id"],
            "best_epoch": best_epoch,
            "config_sha256": sha256_file(config_path),
            "mode": mode,
            "model_state": best_state,
            "seed": seed,
            "variant": variant,
        },
        checkpoint_path,
    )
    summary = {
        "analysis_id": config["analysis_id"],
        "base_checkpoint": str(base_checkpoint) if base_checkpoint else None,
        "base_checkpoint_sha256": (
            sha256_file(base_checkpoint) if base_checkpoint else None
        ),
        "best_epoch": best_epoch,
        "best_internal_validation_ndcg@10": (
            best_metric if validation_indices is not None else None
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "device": device,
        "elapsed_seconds": time.time() - started,
        "epochs": epoch_rows,
        "final_epochs_requested": epochs,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "internal_validation_requests": (
            len(validation_indices) if validation_indices is not None else 0
        ),
        "mode": mode,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "qrels_read": False,
        "run_id": run_id,
        "seed": seed,
        "test_read": False,
        "train_requests": len(train_indices),
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "variant": variant,
    }
    _copy_config(config_path, run_dir)
    write_json(run_dir / "train_summary.json", summary)
    return summary


def build_model(
    config: dict[str, Any],
    variant: str,
    popularity_path: str | Path,
    device: str,
) -> SupervisedDiagnosticRanker:
    embedding_root = Path(config["embedding_artifact"]["root"])
    query_embeddings = torch.from_numpy(
        np.array(
            np.load(
                embedding_root / config["embedding_artifact"]["query_embeddings"],
                mmap_mode="r",
            ),
            copy=True,
        )
    ).to(device)
    item_embeddings = torch.from_numpy(
        np.array(
            np.load(
                embedding_root / config["embedding_artifact"]["item_embeddings"],
                mmap_mode="r",
            ),
            copy=True,
        )
    ).to(device)
    item_log_click = torch.from_numpy(
        np.array(np.load(popularity_path, mmap_mode="r"), dtype=np.float32, copy=True)
    ).to(device)
    return SupervisedDiagnosticRanker(
        query_embeddings=query_embeddings,
        item_embeddings=item_embeddings,
        item_log_click=item_log_click,
        projection_dim=int(config["model"]["projection_dim"]),
        dropout=float(config["model"]["dropout"]),
        variant=variant,
    ).to(device)


def evaluate_internal_ndcg(
    model: SupervisedDiagnosticRanker,
    data: PackedRequestData,
    request_indices: np.ndarray,
    batch_size: int,
    max_padded_candidates: int,
    max_padded_history: int,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    requests = 0
    with torch.inference_mode():
        for numpy_batch in iter_request_batches(
            data,
            request_indices,
            batch_size=batch_size,
            max_padded_candidates=max_padded_candidates,
            max_padded_history=max_padded_history,
            seed=0,
            shuffle=False,
        ):
            batch = batch_to_device(numpy_batch, device)
            scores = model(batch).float().cpu().numpy()
            labels = numpy_batch["candidate_labels"]
            mask = numpy_batch["candidate_mask"]
            item_ids = numpy_batch["candidate_item_ids"]
            for row, request_index in enumerate(numpy_batch["request_indices"]):
                count = int(mask[row].sum())
                request_id = data.request_ids[int(request_index)]
                metrics = request_metrics(
                    request_id=request_id,
                    scored_candidates=[
                        ScoredCandidate(str(item_ids[row, index]), float(scores[row, index]))
                        for index in range(count)
                    ],
                    clicked_item_ids={
                        str(item_ids[row, index])
                        for index in range(count)
                        if labels[row, index] > 0
                    },
                    purchased_item_ids=set(),
                )
                total += float(metrics["ndcg@10"])
                requests += 1
    return total / requests


def evaluate_internal_reference_features(
    config: dict[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    data_root = Path(config["materialized_data"]["output_dir"])
    data = PackedRequestData.load(data_root, "train")
    manifest = _read_json(data_root / "manifest.json")
    cut = int(manifest["internal_calibration"]["cut_request_index"])
    indices = np.arange(cut, len(data), dtype=np.int64)
    embedding_root = Path(config["embedding_artifact"]["root"])
    queries = np.load(
        embedding_root / config["embedding_artifact"]["query_embeddings"],
        mmap_mode="r",
    )
    items = np.load(
        embedding_root / config["embedding_artifact"]["item_embeddings"],
        mmap_mode="r",
    )
    popularity = np.load(data_root / "item_log_click_internal_train.npy", mmap_mode="r")
    alpha_grid = [value / 10 for value in range(11)]
    totals = {"bge_cosine": 0.0, "item_popularity": 0.0, "b0b": 0.0}
    query_mix_totals = {str(alpha): 0.0 for alpha in alpha_grid}
    static_mix_totals = {str(alpha): 0.0 for alpha in alpha_grid}
    for request_index in indices:
        request_index = int(request_index)
        start = int(data.candidate_offsets[request_index])
        end = int(data.candidate_offsets[request_index + 1])
        embedding_indices = np.asarray(
            data.candidate_embedding_indices[start:end], dtype=np.int64
        )
        query = np.asarray(queries[int(data.query_indices[request_index])], dtype=np.float32)
        query /= max(float(np.linalg.norm(query)), 1e-8)
        candidate_embeddings = np.asarray(items[embedding_indices], dtype=np.float32)
        candidate_embeddings /= np.linalg.norm(
            candidate_embeddings, axis=1, keepdims=True
        ).clip(min=1e-8)
        cosine = candidate_embeddings @ query
        pop = np.asarray(popularity[embedding_indices], dtype=np.float32)
        b0b = np.asarray(data.candidate_b0b_scores[start:end], dtype=np.float32)
        item_ids = np.asarray(data.candidate_item_ids[start:end])
        labels = np.asarray(data.candidate_labels[start:end])
        request_id = data.request_ids[request_index]
        positives = {str(item_id) for item_id, label in zip(item_ids, labels) if label > 0}
        feature_values = {
            "bge_cosine": cosine,
            "item_popularity": pop,
            "b0b": b0b,
        }
        for name, values in feature_values.items():
            totals[name] += _ndcg_for_values(request_id, item_ids, values, positives)
        cosine_z = numpy_zscore(cosine)
        pop_z = numpy_zscore(pop)
        b0b_z = numpy_zscore(b0b)
        for alpha in alpha_grid:
            query_mix_totals[str(alpha)] += _ndcg_for_values(
                request_id,
                item_ids,
                alpha * cosine_z + (1.0 - alpha) * pop_z,
                positives,
            )
            static_mix_totals[str(alpha)] += _ndcg_for_values(
                request_id,
                item_ids,
                alpha * cosine_z + (1.0 - alpha) * b0b_z,
                positives,
            )
    count = len(indices)
    result = {
        "analysis_id": config["analysis_id"],
        "internal_validation_requests": count,
        "metrics": {name: value / count for name, value in totals.items()},
        "query_cosine_popularity_grid": {
            alpha: value / count for alpha, value in query_mix_totals.items()
        },
        "query_cosine_b0b_grid": {
            alpha: value / count for alpha, value in static_mix_totals.items()
        },
        "qrels_read": False,
        "scope": "train internal validation only",
    }
    result["best_query_mix"] = max(
        result["query_cosine_popularity_grid"].items(), key=lambda row: row[1]
    )
    result["best_static_mix"] = max(
        result["query_cosine_b0b_grid"].items(), key=lambda row: row[1]
    )
    write_json(output_path, result)
    return result


def numpy_zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    std = float(values.std())
    if std == 0.0:
        return np.zeros_like(values)
    return (values - float(values.mean())) / std


def _ndcg_for_values(
    request_id: str,
    item_ids: np.ndarray,
    values: np.ndarray,
    positives: set[str],
) -> float:
    row = request_metrics(
        request_id=request_id,
        scored_candidates=[
            ScoredCandidate(str(item_id), float(value))
            for item_id, value in zip(item_ids, values)
        ],
        clicked_item_ids=positives,
        purchased_item_ids=set(),
    )
    return float(row["ndcg@10"])


def score_diagnostic(
    config: dict[str, Any],
    config_path: str | Path,
    variant: str,
    seed: int,
    checkpoint_path: str | Path,
    device: str,
    data_override: PackedRequestData | None = None,
    run_id_override: str | None = None,
    method_id_override: str | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_config(config)
    data_root = Path(config["materialized_data"]["output_dir"])
    data = data_override or PackedRequestData.load(data_root, "dev")
    if data.split != "dev":
        raise ValueError(f"diagnostic scoring requires dev data, got {data.split}")
    popularity_path = data_root / "item_log_click_full_train.npy"
    model = build_model(config, variant, popularity_path, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.variant = variant
    model.eval()
    run_id = run_id_override or diagnostic_dev_run_id(variant, seed)
    method_id = method_id_override or diagnostic_method_id(variant)
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    scores_path = run_dir / "scores.jsonl"
    score_rows = 0
    started = time.time()
    with scores_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for numpy_batch in iter_request_batches(
            data,
            np.arange(len(data), dtype=np.int64),
            batch_size=int(config["model"]["batch_size_requests"]),
            max_padded_candidates=int(
                config["model"]["max_padded_candidates_per_batch"]
            ),
            max_padded_history=int(
                config["model"]["max_padded_history_events_per_batch"]
            ),
            seed=0,
            shuffle=False,
        ):
            batch = batch_to_device(numpy_batch, device)
            scores = model(batch).float().cpu().numpy()
            mask = numpy_batch["candidate_mask"]
            item_ids = numpy_batch["candidate_item_ids"]
            for row, request_index in enumerate(numpy_batch["request_indices"]):
                request_id = data.request_ids[int(request_index)]
                count = int(mask[row].sum())
                for index in range(count):
                    handle.write(
                        json.dumps(
                            {
                                "candidate_item_id": str(item_ids[row, index]),
                                "method_id": method_id,
                                "request_id": request_id,
                                "score": float(scores[row, index]),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    score_rows += 1

    standardized_dir = Path(config["standardized_dir"])
    data_manifest = _read_json(data_root / "manifest.json")
    metadata = {
        "analysis_id": config["analysis_id"],
        "candidate_manifest_path": str(standardized_dir / "candidate_manifest.json"),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "data_manifest_path": str(data_root / "manifest.json"),
        "data_manifest_sha256": sha256_file(data_root / "manifest.json"),
        "dataset_id": config["dataset_id"],
        "dataset_version": config["dataset_version"],
        "device": device,
        "elapsed_seconds": time.time() - started,
        "embedding_files": data_manifest["embedding_files"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_fields_used": (
            ["query", "candidate text embedding", "train-only item click count"]
            if variant == "d1q"
            else [
                "query",
                "candidate text embedding",
                "train-only item click count",
                "causal history",
                "frozen B0b score",
            ]
        ),
        "method_id": method_id,
        "qrels_read": False,
        "request_count": len(data),
        "run_id": run_id,
        "score_rows": score_rows,
        "seed": seed,
        "split": "dev",
        "test_read": False,
        "training_labels": "records_train.candidates.clicked only",
        "variant": variant,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    _copy_config(config_path, run_dir)
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def materialize_wrong_history_donor_bank(
    train_records_path: str | Path,
    assignments_paths: Iterable[str | Path],
    item_index_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Extract all fixed C3-R train donors in one scan of train records."""
    assignments_paths = [Path(path) for path in assignments_paths]
    donor_ids = {
        str(row["donor_request_id"])
        for path in assignments_paths
        for row in iter_jsonl(path)
        if row.get("donor_request_id") is not None
    }
    with Path(item_index_path).open("r", encoding="utf-8") as handle:
        item_index = json.load(handle)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    found = set()
    history_rows = 0
    with output_path.open("w", encoding="utf-8") as output:
        for record in iter_jsonl(train_records_path):
            request_id = str(record["request_id"])
            if request_id not in donor_ids:
                continue
            history = [
                {
                    "cat": [str(value) for value in event.get("cat", [])],
                    "event": str(event.get("event") or "click"),
                    "item_id": str(event["item_id"]),
                    "ts": int(event["ts"]),
                }
                for event in record.get("history") or []
            ]
            embedding_indices = []
            event_weights = []
            for event in history:
                item_id = event["item_id"]
                if item_id not in item_index:
                    raise ValueError(f"donor history embedding missing: {item_id}")
                embedding_indices.append(int(item_index[item_id]))
                event_weights.append(1.5 if event["event"] == "purchase" else 1.0)
            output.write(
                json.dumps(
                    {
                        "history": history,
                        "history_embedding_indices": embedding_indices,
                        "history_event_weights": event_weights,
                        "request_id": request_id,
                        "user_id": str(record["user_id"]),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            found.add(request_id)
            history_rows += len(history)
    if found != donor_ids:
        missing = sorted(donor_ids - found)[:5]
        raise ValueError(f"missing donor requests after train scan: {missing}")
    return {
        "assignment_files": {
            str(path): sha256_file(path) for path in assignments_paths
        },
        "donor_requests": len(found),
        "history_rows": history_rows,
        "item_index_path": str(item_index_path),
        "item_index_sha256": sha256_file(item_index_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "qrels_read": False,
        "test_read": False,
        "train_records_path": str(train_records_path),
        "train_records_sha256": sha256_file(train_records_path),
    }


def build_permuted_history_data(
    data: PackedRequestData,
    records_path: str | Path,
    assignments_path: str | Path,
    donor_bank_path: str | Path,
) -> tuple[PackedRequestData, dict[str, Any]]:
    """Replace every non-empty target history with its fixed matched donor."""
    if data.split != "dev":
        raise ValueError("matched history permutation is defined only for dev")
    assignments = {
        str(row["request_id"]): row for row in iter_jsonl(assignments_path)
    }
    if set(assignments) != set(data.request_ids):
        raise ValueError("assignment request IDs do not match packed dev data")
    expected_donor_ids = {
        str(row["donor_request_id"])
        for row in assignments.values()
        if row.get("donor_request_id") is not None
    }
    donor_bank = {
        str(row["request_id"]): row for row in iter_jsonl(donor_bank_path)
    }
    if not expected_donor_ids.issubset(donor_bank):
        raise ValueError("one or more donor histories are unavailable")
    history_offsets = [0]
    history_embedding_indices = []
    history_event_weights = []
    candidate_b0b_scores = np.empty(
        len(data.candidate_b0b_scores), dtype=np.float32
    )
    tier_counts: dict[str, int] = {}
    donor_requests = 0
    absent_requests = 0

    records_seen = 0
    for target_index, record in enumerate(iter_jsonl(records_path)):
        target_id = str(record["request_id"])
        if (
            target_index >= len(data.request_ids)
            or data.request_ids[target_index] != target_id
        ):
            raise ValueError(f"record order mismatch at dev index {target_index}")
        assignment = assignments[target_id]
        tier = str(assignment["match_tier"])
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        target_start = int(data.history_offsets[target_index])
        target_end = int(data.history_offsets[target_index + 1])
        donor_id = assignment.get("donor_request_id")
        if target_start == target_end:
            if donor_id is not None:
                raise ValueError(f"empty-history target has donor: {target_id}")
            selected_history = []
            absent_requests += 1
        else:
            if donor_id is None or str(donor_id) == target_id:
                raise ValueError(
                    f"invalid donor for history-present target: {target_id}"
                )
            if str(assignment["target_user_id"]) == str(assignment["donor_user_id"]):
                raise ValueError(f"donor user equals target user: {target_id}")
            donor = donor_bank[str(donor_id)]
            if str(donor["user_id"]) != str(assignment["donor_user_id"]):
                raise ValueError(f"donor bank user mismatch: {donor_id}")
            selected_history = donor["history"]
            donor_indices = donor["history_embedding_indices"]
            donor_weights = donor["history_event_weights"]
            if not selected_history or len(selected_history) != len(donor_indices):
                raise ValueError(f"invalid donor history: {donor_id}")
            if len(donor_weights) != len(donor_indices):
                raise ValueError(f"invalid donor event weights: {donor_id}")
            if int(assignment["target_history_length"]) != target_end - target_start:
                raise ValueError(f"target history length mismatch: {target_id}")
            history_embedding_indices.extend(donor_indices)
            history_event_weights.extend(donor_weights)
            donor_requests += 1
        history_offsets.append(len(history_embedding_indices))

        score_record = {
            "candidates": record["candidates"],
            "history": selected_history,
        }
        b0b_scores = recent_behavior_scores(score_record)
        candidate_start = int(data.candidate_offsets[target_index])
        candidate_end = int(data.candidate_offsets[target_index + 1])
        if len(record["candidates"]) != candidate_end - candidate_start:
            raise ValueError(f"candidate count mismatch: {target_id}")
        for offset, candidate in enumerate(record["candidates"]):
            item_id = str(candidate["item_id"])
            packed_item_id = str(data.candidate_item_ids[candidate_start + offset])
            if item_id != packed_item_id:
                raise ValueError(f"candidate order mismatch: {target_id}")
            candidate_b0b_scores[candidate_start + offset] = b0b_scores[item_id]
        records_seen += 1

    if records_seen != len(data):
        raise ValueError(f"dev record count mismatch: {records_seen} != {len(data)}")
    permuted = PackedRequestData(
        root=data.root,
        split=data.split,
        request_ids=data.request_ids,
        query_indices=data.query_indices,
        timestamps=data.timestamps,
        candidate_offsets=data.candidate_offsets,
        candidate_embedding_indices=data.candidate_embedding_indices,
        candidate_item_ids=data.candidate_item_ids,
        candidate_labels=data.candidate_labels,
        candidate_purchase_labels=data.candidate_purchase_labels,
        candidate_b0b_scores=candidate_b0b_scores,
        history_offsets=np.asarray(history_offsets, dtype=np.int64),
        history_embedding_indices=np.asarray(
            history_embedding_indices, dtype=np.int32
        ),
        history_event_weights=np.asarray(history_event_weights, dtype=np.float16),
    )
    permuted.validate()
    return permuted, {
        "assignment_sha256": sha256_file(assignments_path),
        "absent_history_requests": absent_requests,
        "donor_requests": donor_requests,
        "history_rows": len(history_embedding_indices),
        "match_tier_counts": tier_counts,
        "donor_bank_path": str(donor_bank_path),
        "donor_bank_sha256": sha256_file(donor_bank_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
    }


def batch_to_device(
    batch: dict[str, np.ndarray],
    device: str,
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in batch.items():
        if key == "request_indices" or key == "candidate_item_ids":
            continue
        tensor = torch.from_numpy(np.asarray(value))
        result[key] = tensor.to(device, non_blocking=True)
    return result


def diagnostic_training_run_id(variant: str, mode: str, seed: int) -> str:
    return f"20260710_kuaisearch_{variant}_{mode}_s{seed}"


def diagnostic_dev_run_id(variant: str, seed: int) -> str:
    names = {
        "d1q": "d1q_supervised_query_dev",
        "d1m": "d1m_mean_history_residual_dev",
        "d1a": "d1a_query_attn_residual_dev",
    }
    return f"20260710_kuaisearch_{names[variant]}_s{seed}"


def diagnostic_method_id(variant: str) -> str:
    return {
        "d1q": "d1q_supervised_query",
        "d1m": "d1m_mean_history_residual",
        "d1a": "d1a_query_attn_residual",
    }[variant]


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def validate_config(config: dict[str, Any]) -> None:
    if config["materialized_data"].get("qrels_read") is not False:
        raise ValueError("materializer must not read qrels")
    if config["internal_calibration"].get("dev_evaluations") != 0:
        raise ValueError("internal calibration cannot evaluate dev")
    if config["materialized_data"].get(
        "drop_train_requests_without_clicked_positive"
    ) is not True:
        raise ValueError("listwise training must drop train requests without positives")
    if config["final_evaluation"].get("test_read") is not False:
        raise ValueError("test access must remain disabled")
    configured = {row["id"] for row in config["variants"]}
    if configured != VARIANTS:
        raise ValueError(f"configured variants differ: {configured}")


def _copy_config(config_path: str | Path, destination: Path) -> None:
    config_path = Path(config_path)
    if config_path.exists():
        shutil.copyfile(config_path, destination / f"config_snapshot{config_path.suffix}")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_hash_int(*parts: str) -> int:
    payload = "|".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
