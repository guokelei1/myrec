"""Train-internal observability diagnostic for nonrepeat history signal.

The module deliberately contains no dev/test/qrels reader.  Held-out fold
labels are loaded lazily and only by the final summarization stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml


MODES = ("full", "text", "id", "null")
TENSOR_NAMES = (
    "query_semantic",
    "candidate_semantic",
    "candidate_indices",
    "candidate_mask",
    "candidate_popularity",
    "history_semantic",
    "history_indices",
    "history_mask",
    "history_weight",
    "labels",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("history observability config must be a mapping")
    if tuple(value["model"]["modes"]) != MODES:
        raise ValueError("history observability modes differ from protocol")
    forbidden = ("records_dev", "records_test", "qrels_dev", "qrels_test")
    for key in forbidden:
        if bool(value["authorization"].get(key, True)):
            raise PermissionError(f"forbidden history observability authority: {key}")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(value: np.ndarray) -> str:
    view = np.ascontiguousarray(value).view(np.uint8)
    return hashlib.sha256(view).hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)


class PackedObservabilityData:
    """Memory-mapped positive train requests with lazy click-label access."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.request_ids = [
            str(json.loads(line)["request_id"])
            for line in (self.root / "request_ids.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line
        ]
        self.query_indices = np.load(self.root / "query_indices.npy", mmap_mode="r")
        self.timestamps = np.load(self.root / "timestamps.npy", mmap_mode="r")
        self.candidate_offsets = np.load(
            self.root / "candidate_offsets.npy", mmap_mode="r"
        )
        self.candidate_indices = np.load(
            self.root / "candidate_embedding_indices.npy", mmap_mode="r"
        )
        self.candidate_item_ids = np.load(
            self.root / "candidate_item_ids.npy", mmap_mode="r"
        )
        self.history_offsets = np.load(
            self.root / "history_offsets.npy", mmap_mode="r"
        )
        self.history_indices = np.load(
            self.root / "history_embedding_indices.npy", mmap_mode="r"
        )
        self.history_weights = np.load(
            self.root / "history_event_weights.npy", mmap_mode="r"
        )
        count = len(self.request_ids)
        if not (
            len(self.query_indices) == count
            and len(self.timestamps) == count
            and len(self.candidate_offsets) == count + 1
            and len(self.history_offsets) == count + 1
        ):
            raise ValueError("packed history observability cardinality differs")
        self._labels: np.ndarray | None = None

    @property
    def labels_opened(self) -> bool:
        return self._labels is not None

    def open_train_labels(self) -> np.ndarray:
        if self._labels is None:
            self._labels = np.load(
                self.root / "candidate_labels.npy", mmap_mode="r"
            )
        return self._labels

    def candidates(self, index: int) -> np.ndarray:
        start, stop = (
            int(self.candidate_offsets[index]),
            int(self.candidate_offsets[index + 1]),
        )
        return np.asarray(self.candidate_indices[start:stop], dtype=np.int64)

    def candidate_ids(self, index: int) -> np.ndarray:
        start, stop = (
            int(self.candidate_offsets[index]),
            int(self.candidate_offsets[index + 1]),
        )
        return np.asarray(self.candidate_item_ids[start:stop]).copy()

    def label_row(self, index: int) -> np.ndarray:
        labels = self.open_train_labels()
        start, stop = (
            int(self.candidate_offsets[index]),
            int(self.candidate_offsets[index + 1]),
        )
        return np.asarray(labels[start:stop], dtype=np.float32)

    def history(self, index: int, max_history: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        start, stop = (
            int(self.history_offsets[index]),
            int(self.history_offsets[index + 1]),
        )
        if max_history is not None:
            start = max(start, stop - int(max_history))
        return (
            np.asarray(self.history_indices[start:stop], dtype=np.int64),
            np.asarray(self.history_weights[start:stop], dtype=np.float32),
        )

    def strict_nonrepeat_indices(self) -> np.ndarray:
        selected: list[int] = []
        for index in range(len(self.request_ids)):
            history, _ = self.history(index)
            if not len(history):
                continue
            candidates = self.candidates(index)
            if not np.intersect1d(candidates, history, assume_unique=False).size:
                selected.append(index)
        return np.asarray(selected, dtype=np.int64)

    def candidate_hash(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for index_value in indices:
            index = int(index_value)
            payload = json.dumps(
                [
                    self.request_ids[index],
                    *[str(value) for value in self.candidate_ids(index).tolist()],
                ],
                separators=(",", ":"),
            ).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()


@dataclass(frozen=True)
class CompactFoldLabels:
    request_indices: np.ndarray
    offsets: np.ndarray
    values: np.ndarray
    _positions: dict[int, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_positions",
            {int(value): row for row, value in enumerate(self.request_indices)},
        )

    @property
    def positions(self) -> dict[int, int]:
        return self._positions

    def row(self, index: int, count: int) -> np.ndarray:
        position = self.positions.get(int(index))
        if position is None:
            raise PermissionError(f"HSO fold label unavailable: {index}")
        start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
        if stop - start != int(count):
            raise ValueError("HSO compact fold label candidate count differs")
        return np.asarray(self.values[start:stop], dtype=np.float32).copy()

    @classmethod
    def load(cls, root: str | Path, fold: int) -> "CompactFoldLabels":
        root = Path(root)
        return cls(
            request_indices=np.load(
                root / f"fit_fold{fold}_request_indices.npy", mmap_mode="r"
            ),
            offsets=np.load(root / f"fit_fold{fold}_label_offsets.npy", mmap_mode="r"),
            values=np.load(root / f"fit_fold{fold}_labels.npy", mmap_mode="r"),
        )


class FrozenSemanticStore:
    """Label-free frozen query and item embeddings."""

    def __init__(
        self,
        data: PackedObservabilityData,
        query_path: str | Path,
        item_path: str | Path,
    ) -> None:
        self.data = data
        self.queries = np.load(query_path, mmap_mode="r")
        self.items = np.load(item_path, mmap_mode="r")
        if self.queries.ndim != 2 or self.items.ndim != 2:
            raise ValueError("history observability embeddings must be matrices")
        if self.queries.shape[1] != self.items.shape[1]:
            raise ValueError("query and item embedding widths differ")
        if int(np.max(data.query_indices)) >= len(self.queries):
            raise ValueError("packed query index exceeds frozen table")
        if int(np.max(data.candidate_indices)) >= len(self.items):
            raise ValueError("packed candidate index exceeds frozen table")
        if int(np.max(data.history_indices)) >= len(self.items):
            raise ValueError("packed history index exceeds frozen table")

    @property
    def width(self) -> int:
        return int(self.items.shape[1])

    def query(self, request_indices: Sequence[int]) -> np.ndarray:
        rows = np.asarray(self.data.query_indices[np.asarray(request_indices)], dtype=np.int64)
        return np.asarray(self.queries[rows], dtype=np.float32)

    def item(self, indices: np.ndarray) -> np.ndarray:
        return np.asarray(self.items[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


def fold_for_user(user_id: str, namespace: str, folds: int) -> int:
    payload = f"{namespace}:{user_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % int(folds)


def masked_zscore(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.to(values.dtype)
    count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * weight).sum(dim=-1, keepdim=True) / count
    variance = ((values - mean).square() * weight).sum(dim=-1, keepdim=True) / count
    output = (values - mean) / variance.sqrt().clamp_min(1e-6)
    return output * weight


def compose_fold_anchor(
    query: torch.Tensor,
    candidates: torch.Tensor,
    popularity: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    query = F.normalize(query.float(), dim=-1, eps=1e-6)
    candidates = F.normalize(candidates.float(), dim=-1, eps=1e-6)
    text = torch.einsum("bd,bcd->bc", query, candidates)
    popularity = torch.log1p(popularity.float().clamp_min(0.0))
    return 0.6 * masked_zscore(text, candidate_mask) + 0.4 * masked_zscore(
        popularity, candidate_mask
    )


@dataclass(frozen=True)
class ObservabilityOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    base_scores: torch.Tensor


class CandidateCrossBlock(nn.Module):
    def __init__(self, width: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            width, heads, dropout=dropout, batch_first=True
        )
        self.norm_attention = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, width),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(width)

    def forward(
        self,
        candidates: torch.Tensor,
        context: torch.Tensor,
        context_padding: torch.Tensor,
    ) -> torch.Tensor:
        attended, _ = self.attention(
            candidates,
            context,
            context,
            key_padding_mask=context_padding,
            need_weights=False,
        )
        candidates = self.norm_attention(candidates + attended)
        return self.norm_ffn(candidates + self.ffn(candidates))


class HistorySignalTransformer(nn.Module):
    """Matched diagnostic Transformer; only its history carrier changes by mode."""

    def __init__(
        self,
        *,
        mode: str,
        input_dim: int,
        width: int,
        heads: int,
        context_layers: int,
        candidate_layers: int,
        ffn_dim: int,
        dropout: float,
        id_buckets: int,
        id_dim: int,
        max_history: int,
        zero_initial_output: bool,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown history observability mode: {mode}")
        if width % heads:
            raise ValueError("history observability width must divide heads")
        self.mode = mode
        self.id_buckets = int(id_buckets)
        self.max_history = int(max_history)
        self.semantic_projection = nn.Sequential(
            nn.Linear(input_dim, width, bias=False), nn.LayerNorm(width)
        )
        self.item_id_embedding = nn.Embedding(
            id_buckets, id_dim, sparse=True
        )
        self.item_id_projection = nn.Linear(id_dim, width, bias=False)
        self.score_projection = nn.Linear(1, width, bias=False)
        self.history_weight_projection = nn.Linear(1, width, bias=False)
        self.query_type = nn.Parameter(torch.empty(width))
        self.history_type = nn.Parameter(torch.empty(width))
        self.candidate_type = nn.Parameter(torch.empty(width))
        self.history_position = nn.Parameter(torch.empty(max_history, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.context_encoder = nn.TransformerEncoder(
            layer, num_layers=context_layers, enable_nested_tensor=False
        )
        self.candidate_blocks = nn.ModuleList(
            CandidateCrossBlock(width, heads, ffn_dim, dropout)
            for _ in range(candidate_layers)
        )
        self.output_norm = nn.LayerNorm(width)
        self.output_head = nn.Linear(width, 1, bias=False)
        self.reset_parameters(zero_initial_output=zero_initial_output)

    def reset_parameters(self, *, zero_initial_output: bool) -> None:
        for value in (
            self.query_type,
            self.history_type,
            self.candidate_type,
            self.history_position,
        ):
            nn.init.normal_(value, std=0.02)
        nn.init.normal_(self.item_id_embedding.weight, std=0.02)
        if zero_initial_output:
            nn.init.zeros_(self.output_head.weight)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def dense_parameters(self) -> list[nn.Parameter]:
        return [
            value
            for name, value in self.named_parameters()
            if value.requires_grad and name != "item_id_embedding.weight"
        ]

    def sparse_parameters(self) -> list[nn.Parameter]:
        return [self.item_id_embedding.weight]

    def _id_state(self, indices: torch.Tensor) -> torch.Tensor:
        buckets = torch.remainder(indices.long(), self.id_buckets)
        return self.item_id_projection(self.item_id_embedding(buckets))

    def forward(
        self,
        *,
        query_semantic: torch.Tensor,
        candidate_semantic: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
        candidate_popularity: torch.Tensor,
        history_semantic: torch.Tensor,
        history_indices: torch.Tensor,
        history_mask: torch.Tensor,
        history_weight: torch.Tensor,
    ) -> ObservabilityOutput:
        candidate_mask = candidate_mask.bool()
        history_mask = history_mask.bool()
        if self.mode == "null":
            history_mask = torch.zeros_like(history_mask)
        base = compose_fold_anchor(
            query_semantic, candidate_semantic, candidate_popularity, candidate_mask
        )
        query = self.semantic_projection(query_semantic.float()) + self.query_type
        candidates = self.semantic_projection(candidate_semantic.float())
        candidates = candidates + self._id_state(candidate_indices)
        candidates = candidates + self.candidate_type + self.score_projection(base[..., None])

        history = torch.zeros(
            *history_mask.shape,
            query.shape[-1],
            dtype=query.dtype,
            device=query.device,
        )
        if self.mode in {"full", "text"}:
            history = history + self.semantic_projection(history_semantic.float())
        if self.mode in {"full", "id"}:
            history = history + self._id_state(history_indices)
        history = history + self.history_type
        history = history + self.history_position[: history.shape[1]][None]
        history = history + self.history_weight_projection(history_weight.float()[..., None])
        history = history * history_mask[..., None].to(history.dtype)
        context = torch.cat((query[:, None], history), dim=1)
        context_padding = torch.cat(
            (
                torch.zeros(
                    len(query), 1, dtype=torch.bool, device=query.device
                ),
                ~history_mask,
            ),
            dim=1,
        )
        context = self.context_encoder(
            context, src_key_padding_mask=context_padding
        )
        for block in self.candidate_blocks:
            candidates = block(candidates, context, context_padding)
        raw = self.output_head(self.output_norm(candidates)).squeeze(-1)
        weight = candidate_mask.to(raw.dtype)
        mean = (raw * weight).sum(-1, keepdim=True) / weight.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        correction = (raw - mean) * weight
        scores = (base + correction).masked_fill(~candidate_mask, 0.0)
        return ObservabilityOutput(scores=scores, correction=correction, base_scores=base)


def listwise_loss(
    output: ObservabilityOutput,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    residual_l2_weight: float,
) -> torch.Tensor:
    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(-1) > 0
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    row = -(target * log_probability).sum(-1)
    ranking = row[valid].mean() if bool(valid.any()) else logits.sum() * 0.0
    energy = output.correction[mask].square().mean()
    return ranking + float(residual_l2_weight) * energy


def build_fold_popularity(
    data: PackedObservabilityData,
    labels: CompactFoldLabels,
    train_indices: Sequence[int],
    item_count: int,
) -> np.ndarray:
    popularity = np.zeros(int(item_count), dtype=np.float32)
    for index_value in train_indices:
        index = int(index_value)
        candidates = data.candidates(index)
        row = labels.row(index, len(candidates))
        np.add.at(popularity, candidates, row)
    return popularity


def _sample_candidate_positions(
    labels: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    if len(labels) <= count:
        return np.arange(len(labels), dtype=np.int64)
    positives = np.flatnonzero(labels > 0)
    negatives = np.flatnonzero(labels <= 0)
    if not len(positives):
        raise ValueError("history observability fit request has no positive")
    positive = int(rng.choice(positives))
    needed = count - 1
    if len(negatives) >= needed:
        chosen = rng.choice(negatives, size=needed, replace=False)
    else:
        remaining = np.asarray(
            [value for value in range(len(labels)) if value != positive],
            dtype=np.int64,
        )
        chosen = rng.choice(remaining, size=needed, replace=False)
    positions = np.concatenate((np.asarray([positive]), chosen.astype(np.int64)))
    rng.shuffle(positions)
    return positions


def collate_requests(
    data: PackedObservabilityData,
    features: FrozenSemanticStore,
    request_indices: Sequence[int],
    popularity: np.ndarray,
    *,
    max_history: int,
    label_access: bool,
    fold_labels: CompactFoldLabels | None = None,
    sampled_candidates: int | None = None,
    rng: np.random.Generator | None = None,
    history_sources: Mapping[int, int] | None = None,
    reverse_history: bool = False,
    empty_history: bool = False,
) -> dict[str, Any]:
    if sampled_candidates is not None and (not label_access or rng is None):
        raise PermissionError("candidate sampling requires fit labels and RNG")
    indices = [int(value) for value in request_indices]
    positions: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    for index in indices:
        count = len(data.candidates(index))
        if label_access:
            if fold_labels is None:
                raise PermissionError("HSO fit collate requires compact fold labels")
            labels = fold_labels.row(index, count)
        else:
            labels = np.zeros(count, dtype=np.float32)
        selected = (
            _sample_candidate_positions(labels, int(sampled_candidates), rng)
            if sampled_candidates is not None
            else np.arange(len(labels), dtype=np.int64)
        )
        positions.append(selected)
        label_rows.append(labels[selected])
    max_candidates = max(len(value) for value in positions)
    history_rows: list[tuple[np.ndarray, np.ndarray]] = []
    for index in indices:
        if empty_history:
            row = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))
        else:
            source = int(history_sources[index]) if history_sources is not None else index
            row = data.history(source, max_history=max_history)
            if reverse_history:
                row = (row[0][::-1].copy(), row[1][::-1].copy())
        history_rows.append(row)
    max_history_count = max(1, max(len(row[0]) for row in history_rows))
    batch = len(indices)
    candidate_index = np.zeros((batch, max_candidates), dtype=np.int64)
    candidate_mask = np.zeros((batch, max_candidates), dtype=bool)
    candidate_popularity = np.zeros((batch, max_candidates), dtype=np.float32)
    labels = np.zeros((batch, max_candidates), dtype=np.float32)
    candidate_item_ids = np.empty((batch, max_candidates), dtype=object)
    history_index = np.zeros((batch, max_history_count), dtype=np.int64)
    history_mask = np.zeros((batch, max_history_count), dtype=bool)
    history_weight = np.zeros((batch, max_history_count), dtype=np.float32)
    request_ids: list[str] = []
    for row, (index, selected) in enumerate(zip(indices, positions)):
        all_candidates = data.candidates(index)
        chosen = all_candidates[selected]
        count = len(chosen)
        candidate_index[row, :count] = chosen
        candidate_mask[row, :count] = True
        candidate_popularity[row, :count] = popularity[chosen]
        labels[row, :count] = label_rows[row]
        all_ids = data.candidate_ids(index)
        candidate_item_ids[row, :count] = [str(value) for value in all_ids[selected]]
        history_items, history_weights = history_rows[row]
        length = len(history_items)
        if length:
            history_index[row, :length] = history_items
            history_mask[row, :length] = True
            history_weight[row, :length] = history_weights
        request_ids.append(data.request_ids[index])
    return {
        "request_indices": np.asarray(indices, dtype=np.int64),
        "request_ids": request_ids,
        "candidate_item_ids": candidate_item_ids,
        "query_semantic": features.query(indices),
        "candidate_semantic": features.item(candidate_index),
        "candidate_indices": candidate_index,
        "candidate_mask": candidate_mask,
        "candidate_popularity": candidate_popularity,
        "history_semantic": features.item(history_index),
        "history_indices": history_index,
        "history_mask": history_mask,
        "history_weight": history_weight,
        "labels": labels,
    }


def to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(np.asarray(batch[name])).to(device)
        for name in TENSOR_NAMES
    }


def forward_kwargs(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensors[name] for name in TENSOR_NAMES if name != "labels"}


def iter_training_batches(
    indices: Sequence[int], *, seed: int, batch_size: int
) -> Iterator[np.ndarray]:
    order = np.asarray(indices, dtype=np.int64).copy()
    np.random.default_rng(seed).shuffle(order)
    for start in range(0, len(order), int(batch_size)):
        yield order[start : start + int(batch_size)]


def iter_evaluation_batches(
    data: PackedObservabilityData,
    indices: Sequence[int],
    *,
    max_requests: int,
    max_candidates: int,
) -> Iterator[np.ndarray]:
    batch: list[int] = []
    candidates = 0
    for index_value in indices:
        index = int(index_value)
        count = len(data.candidates(index))
        if batch and (
            len(batch) >= int(max_requests)
            or candidates + count > int(max_candidates)
        ):
            yield np.asarray(batch, dtype=np.int64)
            batch, candidates = [], 0
        batch.append(index)
        candidates += count
    if batch:
        yield np.asarray(batch, dtype=np.int64)
