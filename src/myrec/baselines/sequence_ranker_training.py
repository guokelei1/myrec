"""Training and scoring for matched official-core HSTU/SASRec PPS baselines."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from myrec.baselines.frozen_text_features import FrozenTextFeatureStore
from myrec.baselines.hstu_pps_adapter import (
    HSTUPPSRanker,
    SequenceBatch,
    collate_sequence_requests,
)
from myrec.baselines.representative_sequence_adapter import (
    SequenceRequest,
    TrainVocabulary,
    build_sequence_request,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


@dataclass(frozen=True)
class LabeledSequenceRequest:
    request: SequenceRequest
    positive_mask: tuple[bool, ...]


def build_labeled_sequence_requests(
    records_path: str | Path,
    qrels_path: str | Path,
    vocabulary: TrainVocabulary,
    *,
    input_mode: str,
    history_budget: int,
) -> tuple[list[LabeledSequenceRequest], dict[str, int | str]]:
    """Build train-only listwise examples; never use labels embedded in records."""

    if input_mode not in {"qc", "full"}:
        raise ValueError(f"unsupported input_mode={input_mode}")
    qrels: dict[str, set[str]] = {}
    for row in iter_jsonl(qrels_path):
        request_id = str(row["request_id"])
        if request_id in qrels:
            raise ValueError(f"duplicate train qrels request_id={request_id}")
        qrels[request_id] = {
            *(str(item_id) for item_id in row.get("clicked", [])),
            *(str(item_id) for item_id in row.get("purchased", [])),
        }
    results: list[LabeledSequenceRequest] = []
    seen: set[str] = set()
    skipped_no_positive = 0
    skipped_no_negative = 0
    for visible_record in iter_jsonl(records_path):
        request_id = str(visible_record["request_id"])
        if request_id in seen:
            raise ValueError(f"duplicate train record request_id={request_id}")
        seen.add(request_id)
        if request_id not in qrels:
            raise ValueError(f"missing train qrels request_id={request_id}")
        record = visible_record
        if input_mode == "qc":
            record = dict(visible_record, history=[])
        request = build_sequence_request(
            record, vocabulary, history_budget=history_budget
        )
        candidate_ids = {candidate.raw_item_id for candidate in request.candidates}
        unknown = qrels[request_id] - candidate_ids
        if unknown:
            raise ValueError(
                f"positive labels outside candidate slate for {request_id}: "
                f"{sorted(unknown)[:5]}"
            )
        positive_mask = tuple(
            candidate.raw_item_id in qrels[request_id]
            for candidate in request.candidates
        )
        positive_count = sum(positive_mask)
        if positive_count == 0:
            skipped_no_positive += 1
            continue
        if positive_count == len(positive_mask):
            skipped_no_negative += 1
            continue
        results.append(
            LabeledSequenceRequest(request=request, positive_mask=positive_mask)
        )
    if seen != set(qrels):
        raise ValueError("train records and qrels have different request coverage")
    if not results:
        raise ValueError("no labeled sequence requests were constructed")
    return results, {
        "input_mode": input_mode,
        "labeled_requests": len(results),
        "requests": len(seen),
        "skipped_no_negative": skipped_no_negative,
        "skipped_no_positive": skipped_no_positive,
    }


def multilabel_listwise_loss(
    scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative log probability mass assigned to all positive candidates."""

    if scores.shape != candidate_mask.shape or scores.shape != positive_mask.shape:
        raise ValueError("score/candidate/positive mask shapes differ")
    if not bool((positive_mask & ~candidate_mask).sum() == 0):
        raise ValueError("positive mask includes padded candidates")
    if not bool(positive_mask.any(dim=1).all()):
        raise ValueError("every request must contain at least one positive")
    minimum = torch.finfo(scores.dtype).min
    denominator = torch.logsumexp(scores.masked_fill(~candidate_mask, minimum), dim=1)
    numerator = torch.logsumexp(scores.masked_fill(~positive_mask, minimum), dim=1)
    return (denominator - numerator).mean()


def collate_labeled_sequence_requests(
    rows: Sequence[LabeledSequenceRequest],
    feature_store: FrozenTextFeatureStore,
    *,
    max_sequence_length: int,
) -> tuple[SequenceBatch, torch.Tensor]:
    batch = collate_sequence_requests(
        [row.request for row in rows],
        feature_store,
        content_dim=feature_store.dimension,
        max_sequence_length=max_sequence_length,
    )
    positive_mask = torch.zeros_like(batch.candidate_mask)
    for index, row in enumerate(rows):
        positive_mask[index, : len(row.positive_mask)] = torch.tensor(
            row.positive_mask, dtype=torch.bool
        )
    return batch, positive_mask


def train_sequence_ranker(
    standardized_dir: str | Path,
    feature_store_dir: str | Path,
    run_id: str,
    output_model_dir: str | Path,
    *,
    architecture: str,
    input_mode: str,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    device: str = "cuda:0",
    history_budget: int = 8,
    embedding_dim: int = 128,
    num_blocks: int = 2,
    num_heads: int = 4,
    dropout_rate: float = 0.1,
    batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    epochs: int = 2,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    max_grad_norm: float = 1.0,
    seed: int = 20260715,
) -> dict[str, Any]:
    """Train one QC or FULL official-core sequence ranker on train qrels only."""

    if min(batch_size, gradient_accumulation_steps, epochs) <= 0:
        raise ValueError("batch, accumulation, and epoch settings must be positive")
    standardized_dir = Path(standardized_dir)
    output_model_dir = Path(output_model_dir)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    if output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"model directory is not empty: {output_model_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)

    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    vocabulary = TrainVocabulary.fit_file(records_path)
    rows, example_stats = build_labeled_sequence_requests(
        records_path,
        qrels_path,
        vocabulary,
        input_mode=input_mode,
        history_budget=history_budget,
    )
    feature_store = FrozenTextFeatureStore(feature_store_dir)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model_config = {
        "architecture": architecture,
        "num_item_ids": vocabulary.num_item_embeddings,
        "num_event_ids": vocabulary.num_event_embeddings,
        "content_dim": feature_store.dimension,
        "embedding_dim": embedding_dim,
        "max_sequence_length": history_budget + 1,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "dropout_rate": dropout_rate,
    }
    model = HSTUPPSRanker(**model_config).to(device)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        rows,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=lambda values: collate_labeled_sequence_requests(
            values,
            feature_store,
            max_sequence_length=history_budget + 1,
        ),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )
    total_updates = math.ceil(len(loader) / gradient_accumulation_steps) * epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1)
    )
    warnings.filterwarnings(
        "once", message=r"fbgemm::.*autograd kernel was not registered.*"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    micro_steps = 0
    optimizer_steps = 0
    epoch_mean_losses: list[float] = []
    started = time.perf_counter()
    for _epoch in range(epochs):
        epoch_loss_sum = 0.0
        epoch_micro_steps = 0
        for batch_index, (batch, positive_mask) in enumerate(loader, start=1):
            batch = batch.to(device)
            positive_mask = positive_mask.to(device)
            scores = model(batch)
            raw_loss = multilabel_listwise_loss(
                scores, batch.candidate_mask, positive_mask
            )
            (raw_loss / gradient_accumulation_steps).backward()
            loss_sum += float(raw_loss.detach().cpu())
            epoch_loss_sum += float(raw_loss.detach().cpu())
            micro_steps += 1
            epoch_micro_steps += 1
            should_step = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == len(loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
        epoch_mean_losses.append(epoch_loss_sum / epoch_micro_steps)
    elapsed = time.perf_counter() - started
    weights_path = output_model_dir / "model.pt"
    torch.save(model.state_dict(), weights_path)
    vocabulary.write(output_model_dir / "vocabulary.json")
    checkpoint_id = (
        f"{architecture}-{input_mode}@{sha256_file(weights_path)[:20]}"
    )
    with (standardized_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    metadata = {
        "schema_version": 1,
        "architecture": architecture,
        "input_mode": input_mode,
        "checkpoint_id": checkpoint_id,
        "run_id": run_id,
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_version": dataset_manifest["dataset_version"],
        "standardized_dir": str(standardized_dir),
        "feature_store_dir": str(feature_store_dir),
        "feature_store_metadata_sha256": sha256_file(
            Path(feature_store_dir) / "metadata.json"
        ),
        "records_train_sha256": sha256_file(records_path),
        "qrels_train_sha256": sha256_file(qrels_path),
        "training_labels_read": True,
        "dev_labels_read": False,
        "model_config": model_config,
        "example_stats": example_stats,
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "max_grad_norm": max_grad_norm,
            "seed": seed,
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "mean_microbatch_loss": loss_sum / micro_steps,
            "epoch_mean_losses": epoch_mean_losses,
            "elapsed_seconds": elapsed,
            "objective": "multi_positive_listwise_softmax",
        },
        "upstream_commit": "6135bc30398f97e5786674192558d91f2ef2fa90",
        "weights_sha256": sha256_file(weights_path),
    }
    write_json(output_model_dir / "metadata.json", metadata)
    write_json(run_dir / "metadata.json", metadata)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
    return metadata


def write_sequence_ranker_scores(
    standardized_dir: str | Path,
    feature_store_dir: str | Path,
    checkpoint_dir: str | Path,
    history_assignments_path: str | Path,
    run_id: str,
    *,
    history_condition: str,
    split: str = "dev",
    runs_dir: str | Path = "runs",
    device: str = "cuda:0",
    batch_size: int = 16,
    method_id: str | None = None,
) -> dict[str, Any]:
    """Score one label-free true/null/wrong condition with a fixed checkpoint."""

    if history_condition not in {"true", "null", "wrong"}:
        raise ValueError(f"unsupported history_condition={history_condition}")
    standardized_dir = Path(standardized_dir)
    checkpoint_dir = Path(checkpoint_dir)
    assignments_path = Path(history_assignments_path)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    with (checkpoint_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        checkpoint = json.load(handle)
    with (checkpoint_dir / "vocabulary.json").open("r", encoding="utf-8") as handle:
        vocabulary = TrainVocabulary.from_dict(json.load(handle))
    assignments = _load_history_assignments(
        assignments_path, expected_condition=history_condition
    )
    feature_store = FrozenTextFeatureStore(feature_store_dir)
    model = HSTUPPSRanker(**checkpoint["model_config"]).to(device)
    state = torch.load(checkpoint_dir / "model.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    records_path = standardized_dir / f"records_{split}.jsonl"
    requests: list[SequenceRequest] = []
    rows_written = 0
    requests_written = 0
    seen: set[str] = set()
    scores_path = run_dir / "scores.jsonl"
    started = time.perf_counter()

    def flush(handle) -> None:
        nonlocal rows_written, requests_written
        if not requests:
            return
        batch = collate_sequence_requests(
            requests,
            feature_store,
            content_dim=feature_store.dimension,
            max_sequence_length=checkpoint["model_config"]["max_sequence_length"],
        ).to(device)
        with torch.inference_mode():
            scores = model(batch).cpu()
        for row_index, request in enumerate(requests):
            for candidate_index, candidate in enumerate(request.candidates):
                value = float(scores[row_index, candidate_index])
                if not math.isfinite(value):
                    raise ValueError(
                        f"non-finite score for {request.request_id} {candidate.raw_item_id}"
                    )
                handle.write(
                    json.dumps(
                        {
                            "request_id": request.request_id,
                            "candidate_item_id": candidate.raw_item_id,
                            "score": value,
                            "method_id": method_id
                            or f"{checkpoint['architecture']}_{checkpoint['input_mode']}",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows_written += 1
            requests_written += 1
        requests.clear()

    with scores_path.open("w", encoding="utf-8") as handle:
        for visible_record in iter_jsonl(records_path):
            request_id = str(visible_record["request_id"])
            if request_id not in assignments:
                raise ValueError(f"assignment missing request_id={request_id}")
            seen.add(request_id)
            record = dict(visible_record, history=assignments[request_id])
            requests.append(
                build_sequence_request(
                    record,
                    vocabulary,
                    history_budget=checkpoint["model_config"]["max_sequence_length"] - 1,
                )
            )
            if len(requests) >= batch_size:
                flush(handle)
        flush(handle)
    if set(assignments) != seen:
        raise ValueError("assignment and scored request coverage differ")
    elapsed = time.perf_counter() - started
    with (standardized_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    scoring_signature = {
        "serialization_version": "history_events_then_current_query_token_v1",
        "max_sequence_length": checkpoint["model_config"]["max_sequence_length"],
        "history_budget": checkpoint["model_config"]["max_sequence_length"] - 1,
        "candidate_scoring_head": "normalized_dot_product",
        "content_feature_contract": feature_store.metadata["feature_contract"],
        "content_feature_store_sha256": sha256_file(
            Path(feature_store_dir) / "metadata.json"
        ),
        "architecture": checkpoint["architecture"],
    }
    metadata = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id
        or f"{checkpoint['architecture']}_{checkpoint['input_mode']}",
        "checkpoint_id": checkpoint["checkpoint_id"],
        "checkpoint_dir": str(checkpoint_dir),
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_version": dataset_manifest["dataset_version"],
        "split": split,
        "history_condition": history_condition,
        "history_assignments_path": str(assignments_path),
        "history_assignment_sha256": sha256_file(assignments_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            standardized_dir / "request_manifest.json"
        ),
        "scoring_signature": scoring_signature,
        "qrels_read": False,
        "request_count": requests_written,
        "score_rows": rows_written,
        "elapsed_seconds": elapsed,
        "device": device,
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _load_history_assignments(
    path: Path, *, expected_condition: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in result:
            raise ValueError(f"duplicate assignment request_id={request_id}")
        if str(row.get("assignment")) != expected_condition:
            raise ValueError(f"assignment condition mismatch for {request_id}")
        history = row.get("history", [])
        if not isinstance(history, list):
            raise ValueError(f"assignment history is not a list for {request_id}")
        result[request_id] = history
    return result
