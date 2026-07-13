"""Full-token candidate cross-encoder for the final history observability test."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


class TokenHistoryData:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.original_indices = np.load(self.root / "request_original_indices.npy", mmap_mode="r")
        self.roles = np.load(self.root / "request_roles.npy", mmap_mode="r")
        self.candidate_offsets = np.load(self.root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_positions = np.load(
            self.root / "candidate_item_positions.npy", mmap_mode="r"
        )
        self.history_offsets = np.load(self.root / "history_offsets.npy", mmap_mode="r")
        self.history_positions = np.load(
            self.root / "history_item_positions.npy", mmap_mode="r"
        )
        self.wrong_offsets = np.load(self.root / "wrong_history_offsets.npy", mmap_mode="r")
        self.wrong_positions = np.load(
            self.root / "wrong_history_item_positions.npy", mmap_mode="r"
        )
        self.query_token_ids = np.load(self.root / "query_token_ids.npy", mmap_mode="r")
        self.query_attention = np.load(self.root / "query_attention_mask.npy", mmap_mode="r")
        self.item_token_ids = np.load(self.root / "item_token_ids.npy", mmap_mode="r")
        self.item_attention = np.load(self.root / "item_attention_mask.npy", mmap_mode="r")
        self.request_ids = self._position_values(self.root / "requests.jsonl", "request_id")
        self.item_ids = self._position_values(self.root / "items.jsonl", "item_id")
        manifest = json.loads((self.root / "token_manifest.json").read_text(encoding="utf-8"))
        self.cls_token_id = int(manifest["special_tokens"]["cls_token_id"])
        self.sep_token_id = int(manifest["special_tokens"]["sep_token_id"])
        self.pad_token_id = int(manifest["special_tokens"]["pad_token_id"])
        count = len(self.request_ids)
        if not (
            len(self.original_indices) == count
            and len(self.roles) == count
            and len(self.candidate_offsets) == count + 1
            and len(self.history_offsets) == count + 1
            and len(self.wrong_offsets) == count + 1
            and len(self.query_token_ids) == count
        ):
            raise ValueError("token HSO request cardinality differs")

    @staticmethod
    def _position_values(path: Path, field: str) -> list[str]:
        output = []
        with path.open("r", encoding="utf-8") as handle:
            for expected, line in enumerate(handle):
                row = json.loads(line)
                if int(row["position"]) != expected:
                    raise ValueError(f"token HSO {path.name} position differs")
                output.append(str(row[field]))
        return output

    @property
    def fit_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.roles) == 0).astype(np.int64)

    @property
    def reserve_indices(self) -> np.ndarray:
        return np.flatnonzero(np.asarray(self.roles) == 1).astype(np.int64)

    def candidates(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_positions[start:stop], dtype=np.int64)

    def candidate_ids(self, index: int) -> list[str]:
        return [self.item_ids[int(value)] for value in self.candidates(index)]

    def history(self, index: int, scenario: str, max_history: int) -> np.ndarray:
        if scenario == "null":
            return np.empty(0, dtype=np.int64)
        if scenario in {"true", "shuffle"}:
            offsets, values = self.history_offsets, self.history_positions
        elif scenario == "wrong":
            offsets, values = self.wrong_offsets, self.wrong_positions
        else:
            raise ValueError(f"unknown token HSO history scenario: {scenario}")
        start, stop = int(offsets[index]), int(offsets[index + 1])
        start = max(start, stop - int(max_history))
        output = np.asarray(values[start:stop], dtype=np.int64)
        return output[::-1].copy() if scenario == "shuffle" else output

    def candidate_hash(self, indices: Sequence[int]) -> str:
        digest = hashlib.sha256()
        for index_value in indices:
            index = int(index_value)
            payload = json.dumps(
                [self.request_ids[index], *self.candidate_ids(index)],
                separators=(",", ":"),
            ).encode()
            digest.update(len(payload).to_bytes(8, "big"))
            digest.update(payload)
        return digest.hexdigest()

    @staticmethod
    def _tokens(ids: np.ndarray, mask: np.ndarray, limit: int) -> list[int]:
        return [int(value) for value in ids[np.asarray(mask, dtype=bool)][: int(limit)]]

    def pack_candidate(
        self,
        request_index: int,
        candidate_position: int,
        *,
        scenario: str,
        query_tokens: int,
        candidate_tokens: int,
        history_item_tokens: int,
        max_history: int,
        max_length: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        query = self._tokens(
            self.query_token_ids[request_index],
            self.query_attention[request_index],
            query_tokens,
        )
        candidate = self._tokens(
            self.item_token_ids[candidate_position],
            self.item_attention[candidate_position],
            candidate_tokens,
        )
        sequence = [self.cls_token_id, *query, self.sep_token_id, *candidate, self.sep_token_id]
        for history_position in self.history(request_index, scenario, max_history):
            history = self._tokens(
                self.item_token_ids[int(history_position)],
                self.item_attention[int(history_position)],
                history_item_tokens,
            )
            remaining = int(max_length) - len(sequence) - 1
            if remaining <= 0:
                break
            sequence.extend(history[:remaining])
            sequence.append(self.sep_token_id)
        sequence = sequence[: int(max_length)]
        ids = np.full(int(max_length), self.pad_token_id, dtype=np.int64)
        attention = np.zeros(int(max_length), dtype=bool)
        ids[: len(sequence)] = sequence
        attention[: len(sequence)] = True
        return ids, attention


class TokenHistoryCrossEncoder(nn.Module):
    def __init__(self, snapshot: str | Path, *, score_head_bias: bool) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(snapshot, local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        self.score_head = nn.Linear(hidden, 1, bias=score_head_bias)
        nn.init.normal_(self.score_head.weight, std=0.02)
        if self.score_head.bias is not None:
            nn.init.zeros_(self.score_head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output: Any = self.backbone(
            input_ids=input_ids.long(), attention_mask=attention_mask.long()
        )
        return self.score_head(output.last_hidden_state[:, 0]).squeeze(-1)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    target = labels.float().clamp_min(0.0)
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    return -(target * F.log_softmax(scores.float(), dim=-1)).sum(-1).mean()


def sample_positions(
    labels: np.ndarray, count: int, rng: np.random.Generator
) -> np.ndarray:
    if len(labels) <= count:
        return np.arange(len(labels), dtype=np.int64)
    positive = np.flatnonzero(labels > 0)
    negative = np.flatnonzero(labels <= 0)
    if len(positive) == 0:
        raise ValueError("full-token ranking request has no positive")
    if len(positive) >= count:
        output = rng.choice(positive, size=count, replace=False).astype(np.int64)
        rng.shuffle(output)
        return output
    chosen = rng.choice(negative, size=count - len(positive), replace=False)
    output = np.concatenate((positive, chosen.astype(np.int64)))
    rng.shuffle(output)
    return output


def pack_training_batch(
    data: TokenHistoryData,
    request_indices: Sequence[int],
    labels_by_request: dict[int, np.ndarray],
    *,
    sampled_candidates: int,
    candidate_rng: np.random.Generator,
    dropout_rng: np.random.Generator,
    history_dropout: float,
    token_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_ids = []
    all_attention = []
    label_rows = []
    batch_candidate_count = min(
        int(sampled_candidates),
        *(len(labels_by_request[int(index)]) for index in request_indices),
    )
    for index_value in request_indices:
        index = int(index_value)
        labels = labels_by_request[index]
        positions = sample_positions(labels, batch_candidate_count, candidate_rng)
        candidates = data.candidates(index)[positions]
        scenario = "null" if dropout_rng.random() < float(history_dropout) else "true"
        for candidate in candidates:
            ids, attention = data.pack_candidate(
                index,
                int(candidate),
                scenario=scenario,
                query_tokens=int(token_config["query_tokens"]),
                candidate_tokens=int(token_config["candidate_tokens"]),
                history_item_tokens=int(token_config["history_item_tokens"]),
                max_history=int(token_config["max_history"]),
                max_length=int(token_config["max_sequence_length"]),
            )
            all_ids.append(ids)
            all_attention.append(attention)
        label_rows.append(labels[positions])
    return (
        np.asarray(all_ids, dtype=np.int64),
        np.asarray(all_attention, dtype=bool),
        np.asarray(label_rows, dtype=np.float32),
    )
