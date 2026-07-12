"""Frozen local-LM feature preparation for the C03 falsifier."""

from __future__ import annotations

import heapq
import time
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from io_utils import (
    assert_manifest,
    assert_safe_input,
    iter_jsonl,
    repo_path,
    sha256_file,
    stable_i63,
    stable_u64,
)


def format_query(query: object) -> str:
    return f"query: {str(query).strip()}"


def format_item(item: dict[str, Any]) -> str:
    categories = " > ".join(str(value) for value in item.get("cat", []) if value)
    fields = [
        str(item.get("title", "")).strip(),
        str(item.get("brand", "")).strip(),
        categories,
    ]
    text = " | ".join(value for value in fields if value)
    return f"item: {text or '[missing text]'}"


def format_coarse(item: dict[str, Any]) -> str:
    categories = [str(value).strip() for value in item.get("cat", []) if str(value).strip()]
    deepest = categories[-1] if categories else "[missing category]"
    return f"category: {deepest}"


class FrozenBGE:
    """Local-only BGE encoder; no network access and no trainable parameters."""

    def __init__(self, model_name: str, device: torch.device) -> None:
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=False,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            local_files_only=True,
            trust_remote_code=False,
        ).to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def encode(self, texts: list[str], batch_size: int) -> Tensor:
        chunks: list[Tensor] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            tokens = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=self.device.type == "cuda",
            ):
                output = self.model(**tokens).last_hidden_state[:, 0]
            chunks.append(F.normalize(output.float(), dim=-1).cpu())
        if not chunks:
            hidden_size = int(self.model.config.hidden_size)
            return torch.empty((0, hidden_size), dtype=torch.float32)
        return torch.cat(chunks, dim=0)


def _select_train_records(path: Path, limit: int, seed: int) -> list[dict[str, Any]]:
    """Select smallest request hashes before inspecting any candidate label."""

    heap: list[tuple[int, str, dict[str, Any]]] = []
    for record in iter_jsonl(path):
        request_id = str(record.get("request_id", ""))
        if not request_id:
            raise ValueError("train record missing request_id")
        priority = stable_u64("c03-train-request", seed, request_id)
        entry = (-priority, request_id, record)
        if len(heap) < limit:
            heapq.heappush(heap, entry)
        elif priority < -heap[0][0]:
            heapq.heapreplace(heap, entry)
    selected = [(-negative, request_id, record) for negative, request_id, record in heap]
    selected.sort(key=lambda value: (value[0], value[1]))
    return [record for _, _, record in selected]


def _select_candidates(
    record: dict[str, Any],
    cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    candidates = list(record.get("candidates", []))
    positives = [candidate for candidate in candidates if int(candidate.get("clicked", 0)) == 1]
    negatives = [candidate for candidate in candidates if int(candidate.get("clicked", 0)) != 1]
    negatives.sort(
        key=lambda candidate: stable_u64(
            "c03-train-negative",
            seed,
            record["request_id"],
            candidate.get("item_id", ""),
        )
    )
    if len(positives) >= cap:
        positives.sort(
            key=lambda candidate: stable_u64(
                "c03-train-positive",
                seed,
                record["request_id"],
                candidate.get("item_id", ""),
            )
        )
        return positives[:cap]
    return positives + negatives[: cap - len(positives)]


def _event_type(event: object) -> int:
    return 2 if str(event) == "purchase" else 1


def _encode_text_universe(
    texts: Iterable[str],
    *,
    encoder: FrozenBGE,
    batch_size: int,
    dtype: torch.dtype,
) -> tuple[list[str], Tensor, dict[str, int]]:
    ordered = sorted(set(texts))
    embeddings = encoder.encode(ordered, batch_size=batch_size).to(dtype=dtype)
    index = {text: position for position, text in enumerate(ordered)}
    return ordered, embeddings, index


def prepare_train_features(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    assert_manifest(config)
    reject = config["integrity"]["reject_path_tokens"]
    train_path = assert_safe_input(config["paths"]["records_train"], reject)
    output_dir = repo_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "train_features.pt"
    seed = int(config["candidate"]["seed"])
    request_limit = int(config["data"]["train_requests"])
    max_history = int(config["data"]["max_history"])
    candidate_cap = int(config["data"]["max_candidates_per_train_request"])
    records = _select_train_records(train_path, request_limit, seed)

    compact: list[dict[str, Any]] = []
    text_universe: list[str] = []
    for record in records:
        history = list(record.get("history", []))[-max_history:]
        candidates = _select_candidates(record, candidate_cap, seed)
        if not candidates:
            continue
        query_text = format_query(record.get("query", ""))
        history_texts = [format_item(item) for item in history]
        coarse_texts = [format_coarse(item) for item in history]
        candidate_texts = [format_item(item) for item in candidates]
        text_universe.extend([query_text, *history_texts, *coarse_texts, *candidate_texts])
        compact.append(
            {
                "request_id": str(record["request_id"]),
                "user_hash": stable_i63("user", record.get("user_id", "")),
                "query_text": query_text,
                "history_texts": history_texts,
                "coarse_texts": coarse_texts,
                "history_item_hashes": [
                    stable_i63("item", item.get("item_id", "")) for item in history
                ],
                "event_types": [_event_type(item.get("event")) for item in history],
                "candidate_texts": candidate_texts,
                "candidate_item_hashes": [
                    stable_i63("item", item.get("item_id", "")) for item in candidates
                ],
                "candidate_item_ids": [str(item.get("item_id", "")) for item in candidates],
                "labels": [float(int(item.get("clicked", 0))) for item in candidates],
            }
        )

    encoder = FrozenBGE(str(config["paths"]["local_model"]), device)
    dtype_name = str(config["data"]["embedding_dtype"])
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    texts, embeddings, index = _encode_text_universe(
        text_universe,
        encoder=encoder,
        batch_size=int(config["data"]["embedding_batch_size"]),
        dtype=dtype,
    )
    requests: list[dict[str, Any]] = []
    validation_fraction = float(config["data"]["internal_validation_fraction"])
    threshold = int(validation_fraction * (2**64 - 1))
    for row in compact:
        history_count = len(row["history_texts"])
        history_embeddings = torch.zeros((max_history, embeddings.shape[1]), dtype=dtype)
        coarse_embeddings = torch.zeros_like(history_embeddings)
        history_mask = torch.zeros(max_history, dtype=torch.bool)
        event_types = torch.zeros(max_history, dtype=torch.long)
        history_hashes = torch.full((max_history,), -1, dtype=torch.long)
        if history_count:
            history_embeddings[:history_count] = torch.stack(
                [embeddings[index[text]] for text in row["history_texts"]]
            )
            coarse_embeddings[:history_count] = torch.stack(
                [embeddings[index[text]] for text in row["coarse_texts"]]
            )
            history_mask[:history_count] = True
            event_types[:history_count] = torch.tensor(row["event_types"], dtype=torch.long)
            history_hashes[:history_count] = torch.tensor(
                row["history_item_hashes"], dtype=torch.long
            )
        candidate_embeddings = torch.stack(
            [embeddings[index[text]] for text in row["candidate_texts"]]
        )
        is_validation = (
            stable_u64("c03-internal-validation", seed, row["request_id"]) <= threshold
        )
        requests.append(
            {
                "request_id": row["request_id"],
                "user_hash": row["user_hash"],
                "is_validation": is_validation,
                "query": embeddings[index[row["query_text"]]],
                "history": history_embeddings,
                "coarse_history": coarse_embeddings,
                "history_mask": history_mask,
                "event_types": event_types,
                "history_item_hashes": history_hashes,
                "candidates": candidate_embeddings,
                "candidate_item_hashes": torch.tensor(
                    row["candidate_item_hashes"], dtype=torch.long
                ),
                "candidate_item_ids": row["candidate_item_ids"],
                "labels": torch.tensor(row["labels"], dtype=torch.float32),
            }
        )
    payload = {
        "metadata": {
            "created_at_unix": time.time(),
            "records_train_sha256": sha256_file(train_path),
            "selection_seed": seed,
            "request_limit": request_limit,
            "selected_requests": len(requests),
            "fit_requests": sum(not row["is_validation"] for row in requests),
            "validation_requests": sum(row["is_validation"] for row in requests),
            "unique_texts": len(texts),
            "local_model": config["paths"]["local_model"],
            "qrels_read": False,
            "test_read": False,
        },
        "requests": requests,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)
    return payload["metadata"] | {"path": str(output_path)}


def prepare_dev_text_store(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    assert_manifest(config)
    reject = config["integrity"]["reject_path_tokens"]
    dev_path = assert_safe_input(config["paths"]["records_dev"], reject)
    output_dir = repo_path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "dev_text_store.pt"
    texts: set[str] = set()
    request_count = 0
    candidate_count = 0
    for record in iter_jsonl(dev_path):
        request_count += 1
        texts.add(format_query(record.get("query", "")))
        for item in record.get("history", []):
            texts.add(format_item(item))
            texts.add(format_coarse(item))
        for item in record.get("candidates", []):
            candidate_count += 1
            texts.add(format_item(item))
    encoder = FrozenBGE(str(config["paths"]["local_model"]), device)
    dtype_name = str(config["data"]["embedding_dtype"])
    dtype = torch.float16 if dtype_name == "float16" else torch.float32
    ordered = sorted(texts)
    embeddings = encoder.encode(
        ordered,
        batch_size=int(config["data"]["embedding_batch_size"]),
    ).to(dtype=dtype)
    payload = {
        "metadata": {
            "created_at_unix": time.time(),
            "records_dev_sha256": sha256_file(dev_path),
            "requests": request_count,
            "candidate_rows": candidate_count,
            "unique_texts": len(ordered),
            "local_model": config["paths"]["local_model"],
            "qrels_read": False,
            "test_read": False,
        },
        "texts": ordered,
        "embeddings": embeddings,
    }
    torch.save(payload, output_path)
    return payload["metadata"] | {"path": str(output_path)}


def load_dev_text_store(config: dict[str, Any]) -> tuple[Tensor, dict[str, int], dict[str, Any]]:
    path = repo_path(config["paths"]["output_dir"]) / "dev_text_store.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    index = {text: position for position, text in enumerate(payload["texts"])}
    return payload["embeddings"], index, payload["metadata"]
