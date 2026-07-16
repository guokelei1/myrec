"""Train and score the independent, PPS-adapted LLM-SRec baseline."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.nn import functional as F

from myrec.baselines.llm_srec_adapter import (
    FrozenQwenLLMSRecEncoder,
    LLMSRecRetrievalHead,
)
from myrec.baselines.representative_sequence_adapter import (
    SequenceCandidate,
    SequenceRequest,
    TrainVocabulary,
    build_sequence_request,
)
from myrec.baselines.sequence_ranker_training import (
    LabeledSequenceRequest,
    build_labeled_sequence_requests,
)
from myrec.baselines.sequence_teacher_features import FrozenSequenceTeacherStore
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


def sample_llm_srec_candidates(
    row: LabeledSequenceRequest,
    *,
    negatives: int,
    rng: random.Random,
) -> tuple[list[SequenceCandidate], int]:
    """Sample one deterministic positive plus ordinary in-slate negatives."""

    if negatives <= 0:
        raise ValueError("negative count must be positive")
    positive_indices = [i for i, value in enumerate(row.positive_mask) if value]
    negative_indices = [i for i, value in enumerate(row.positive_mask) if not value]
    if not positive_indices or not negative_indices:
        raise ValueError("LLM-SRec row needs positive and negative candidates")
    positive_index = positive_indices[0]
    if len(negative_indices) > negatives:
        negative_indices = rng.sample(negative_indices, negatives)
    candidates = [row.request.candidates[positive_index]] + [
        row.request.candidates[index] for index in negative_indices
    ]
    return candidates, 0


def collate_teacher_history(
    requests: Sequence[SequenceRequest],
    teacher: FrozenSequenceTeacherStore,
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((request.retained_history_count for request in requests), default=0)
    values = torch.zeros(len(requests), width, teacher.dimension, device=device)
    mask = torch.zeros(len(requests), width, dtype=torch.bool, device=device)
    for row, request in enumerate(requests):
        for column, (raw_id, text) in enumerate(
            zip(request.past_raw_item_ids, request.past_content_texts)
        ):
            values[row, column] = torch.from_numpy(teacher.item(raw_id, text)).to(
                device
            )
            mask[row, column] = True
    return values, mask


def train_llm_srec(
    standardized_dir: str | Path,
    teacher_store_dir: str | Path,
    run_id: str,
    output_model_dir: str | Path,
    *,
    backbone: str = "models/huggingface/Qwen3-Reranker-0.6B",
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    device: str = "cuda:0",
    history_budget: int = 8,
    max_length: int = 1024,
    projection_dim: int = 128,
    hidden_dim: int = 256,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    negatives: int = 3,
    epochs: int = 1,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    retrieval_weight: float = 1.0,
    distillation_weight: float = 1.0,
    uniformity_weight: float = 1.0,
    seed: int = 20260715,
    max_train_requests: int | None = None,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    output_model_dir = Path(output_model_dir)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    if output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"model directory is not empty: {output_model_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    vocabulary = TrainVocabulary.fit_file(records_path)
    rows, example_stats = build_labeled_sequence_requests(
        records_path,
        qrels_path,
        vocabulary,
        input_mode="full",
        history_budget=history_budget,
    )
    if max_train_requests is not None:
        if max_train_requests <= 0:
            raise ValueError("max_train_requests must be positive")
        rows = rows[:max_train_requests]
    teacher = FrozenSequenceTeacherStore(teacher_store_dir)
    encoder = FrozenQwenLLMSRecEncoder(
        model_name_or_path=backbone,
        cf_item_dim=teacher.dimension,
        max_length=max_length,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    head = LLMSRecRetrievalHead(
        llm_dim=encoder.llm_dim,
        cf_dim=teacher.dimension,
        projection_dim=projection_dim,
        hidden_dim=hidden_dim,
        retrieval_weight=retrieval_weight,
        distillation_weight=distillation_weight,
        uniformity_weight=uniformity_weight,
    ).to(device)
    trainable = [
        parameter
        for module in (encoder, head)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=weight_decay
    )
    total_micro_steps = math.ceil(len(rows) / batch_size) * epochs
    total_updates = math.ceil(total_micro_steps / gradient_accumulation_steps)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_updates, 1)
    )
    encoder.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sums = {"total": 0.0, "retrieval": 0.0, "distillation": 0.0, "uniformity": 0.0}
    micro_steps = 0
    optimizer_steps = 0
    started = time.perf_counter()
    order = list(range(len(rows)))
    rng = random.Random(seed)
    for _epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch_rows = [rows[index] for index in order[start : start + batch_size]]
            requests = [row.request for row in batch_rows]
            selected = [
                sample_llm_srec_candidates(row, negatives=negatives, rng=rng)[0]
                for row in batch_rows
            ]
            width = max(len(values) for values in selected)
            flat_candidates: list[SequenceCandidate] = []
            candidate_mask = torch.zeros(
                len(selected), width, dtype=torch.bool, device=device
            )
            flat_cf_items: list[torch.Tensor] = []
            for row_index, candidates in enumerate(selected):
                candidate_mask[row_index, : len(candidates)] = True
                for candidate in candidates:
                    flat_candidates.append(candidate)
                    flat_cf_items.append(
                        torch.from_numpy(
                            teacher.item(candidate.raw_item_id, candidate.content_text)
                        )
                    )
                # Pad with a repeated negative; it is masked after encoding.
                for _ in range(width - len(candidates)):
                    flat_candidates.append(candidates[-1])
                    flat_cf_items.append(
                        torch.from_numpy(
                            teacher.item(
                                candidates[-1].raw_item_id,
                                candidates[-1].content_text,
                            )
                        )
                    )
            cf_history, history_mask = collate_teacher_history(
                requests, teacher, device=device
            )
            llm_user = encoder.encode_users(requests, cf_history, history_mask)
            llm_items = encoder.encode_items(
                flat_candidates, torch.stack(flat_cf_items).to(device)
            ).reshape(len(requests), width, encoder.llm_dim)
            cf_user = torch.stack(
                [torch.from_numpy(teacher.train_user(r.request_id)) for r in requests]
            ).to(device)
            _, losses = head.losses(
                llm_user=llm_user,
                llm_items=llm_items,
                cf_user=cf_user,
                positive_indices=torch.zeros(
                    len(requests), dtype=torch.long, device=device
                ),
                candidate_mask=candidate_mask,
            )
            (losses.total / gradient_accumulation_steps).backward()
            for name in loss_sums:
                loss_sums[name] += float(getattr(losses, name).detach().cpu())
            micro_steps += 1
            if micro_steps % gradient_accumulation_steps == 0 or micro_steps == total_micro_steps:
                torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
    elapsed = time.perf_counter() - started
    weights = {
        "cf_item_projection": encoder.cf_item_projection.state_dict(),
        "user_output_embedding": encoder.user_output_embedding.detach().cpu(),
        "item_output_embedding": encoder.item_output_embedding.detach().cpu(),
        "retrieval_head": head.state_dict(),
    }
    weights_path = output_model_dir / "trainable.pt"
    torch.save(weights, weights_path)
    vocabulary.write(output_model_dir / "vocabulary.json")
    teacher_metadata = Path(teacher_store_dir) / "metadata.json"
    metadata = {
        "schema_version": 1,
        "method_id": "llm_srec_pps",
        "run_id": run_id,
        "checkpoint_id": f"llm-srec@{sha256_file(weights_path)[:20]}",
        "backbone": backbone,
        "backbone_frozen": True,
        "teacher_store": str(teacher_store_dir),
        "teacher_metadata_sha256": sha256_file(teacher_metadata),
        "history_budget": history_budget,
        "max_length": max_length,
        "llm_dim": encoder.llm_dim,
        "cf_dim": teacher.dimension,
        "projection_dim": projection_dim,
        "hidden_dim": hidden_dim,
        "training": {
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "negatives": negatives,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "retrieval_weight": retrieval_weight,
            "distillation_weight": distillation_weight,
            "uniformity_weight": uniformity_weight,
            "seed": seed,
            "max_train_requests": max_train_requests,
            "used_train_requests": len(rows),
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "mean_losses": {key: value / micro_steps for key, value in loss_sums.items()},
            "elapsed_seconds": elapsed,
            "multi_positive_policy": "first_positive_deterministic",
        },
        "standardized_dir": str(standardized_dir),
        "records_train_sha256": sha256_file(records_path),
        "qrels_train_sha256": sha256_file(qrels_path),
        "training_labels_read": True,
        "dev_labels_read": False,
        "confirmation_labels_read": False,
        "weights_sha256": sha256_file(weights_path),
        "example_stats": example_stats,
    }
    write_json(output_model_dir / "metadata.json", metadata)
    write_json(run_dir / "metadata.json", metadata)
    if config_path and Path(config_path).exists():
        config_path = Path(config_path)
        shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
    return metadata


def load_llm_srec(
    checkpoint_dir: str | Path, *, device: str
) -> tuple[FrozenQwenLLMSRecEncoder, LLMSRecRetrievalHead, dict[str, Any]]:
    checkpoint_dir = Path(checkpoint_dir)
    with (checkpoint_dir / "metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    encoder = FrozenQwenLLMSRecEncoder(
        model_name_or_path=metadata["backbone"],
        cf_item_dim=metadata["cf_dim"],
        max_length=metadata["max_length"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(device)
    head = LLMSRecRetrievalHead(
        llm_dim=metadata["llm_dim"],
        cf_dim=metadata["cf_dim"],
        projection_dim=metadata["projection_dim"],
        hidden_dim=metadata["hidden_dim"],
    ).to(device)
    weights = torch.load(
        checkpoint_dir / "trainable.pt", map_location=device, weights_only=True
    )
    encoder.cf_item_projection.load_state_dict(weights["cf_item_projection"])
    with torch.no_grad():
        encoder.user_output_embedding.copy_(weights["user_output_embedding"])
        encoder.item_output_embedding.copy_(weights["item_output_embedding"])
    head.load_state_dict(weights["retrieval_head"])
    encoder.eval()
    head.eval()
    return encoder, head, metadata


def write_llm_srec_scores(
    standardized_dir: str | Path,
    teacher_store_dir: str | Path,
    checkpoint_dir: str | Path,
    assignments_path: str | Path,
    run_id: str,
    *,
    history_condition: str,
    split: str = "dev",
    runs_dir: str | Path = "runs",
    device: str = "cuda:0",
) -> dict[str, Any]:
    if history_condition not in {"true", "null", "wrong"}:
        raise ValueError("history condition must be true/null/wrong")
    if split not in {"dev", "internal", "confirmation"}:
        raise ValueError("LLM-SRec scoring supports dev, internal, or confirmation only")
    standardized_dir = Path(standardized_dir)
    assignments_path = Path(assignments_path)
    checkpoint_dir = Path(checkpoint_dir)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    encoder, head, metadata = load_llm_srec(checkpoint_dir, device=device)
    teacher = FrozenSequenceTeacherStore(teacher_store_dir)
    with (checkpoint_dir / "vocabulary.json").open("r", encoding="utf-8") as handle:
        vocabulary = TrainVocabulary.from_dict(json.load(handle))
    assignments: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(assignments_path):
        if row.get("assignment") != history_condition:
            raise ValueError("assignment condition mismatch")
        request_id = str(row["request_id"])
        if request_id in assignments:
            raise ValueError(f"duplicate history assignment request_id={request_id}")
        assignments[request_id] = row.get("history", [])
    rows_written = 0
    requests_written = 0
    started = time.perf_counter()
    records_path = standardized_dir / f"records_{split}.jsonl"
    if not records_path.exists():
        raise FileNotFoundError(f"missing standardized records for split={split}: {records_path}")
    seen_request_ids: set[str] = set()
    with (run_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for visible in iter_jsonl(records_path):
            request_id = str(visible["request_id"])
            if request_id not in assignments:
                raise ValueError(f"assignment missing request_id={request_id}")
            seen_request_ids.add(request_id)
            request = build_sequence_request(
                dict(visible, history=assignments[request_id]),
                vocabulary,
                history_budget=metadata["history_budget"],
            )
            cf_history, history_mask = collate_teacher_history(
                [request], teacher, device=device
            )
            cf_items = torch.stack(
                [
                    torch.from_numpy(
                        teacher.item(candidate.raw_item_id, candidate.content_text)
                    )
                    for candidate in request.candidates
                ]
            ).to(device)
            with torch.inference_mode():
                user = encoder.encode_users([request], cf_history, history_mask)
                items = encoder.encode_items(request.candidates, cf_items).unsqueeze(0)
                scores, _, _ = head.score(user, items)
            for candidate, score in zip(request.candidates, scores[0].cpu()):
                value = float(score)
                if not math.isfinite(value):
                    raise ValueError("non-finite LLM-SRec score")
                handle.write(
                    json.dumps(
                        {
                            "request_id": request_id,
                            "candidate_item_id": candidate.raw_item_id,
                            "score": value,
                            "method_id": "llm_srec_pps",
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows_written += 1
            requests_written += 1
    if seen_request_ids != set(assignments):
        raise ValueError("assignment and scored request coverage differ")
    with (standardized_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    output = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": "llm_srec_pps",
        "checkpoint_id": metadata["checkpoint_id"],
        "dataset_id": dataset_manifest["dataset_id"],
        "dataset_version": dataset_manifest["dataset_version"],
        "split": split,
        "history_condition": history_condition,
        "history_assignments_path": str(assignments_path),
        "history_assignment_sha256": sha256_file(assignments_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            standardized_dir / "request_manifest.json"
        ),
        "qrels_read": False,
        "request_count": requests_written,
        "score_rows": rows_written,
        "elapsed_seconds": time.perf_counter() - started,
        "scoring_signature": {
            "architecture": "llm_srec_pps",
            "backbone": metadata["backbone"],
            "teacher_metadata_sha256": metadata["teacher_metadata_sha256"],
            "history_budget": metadata["history_budget"],
            "max_length": metadata["max_length"],
            "serialization_version": "llm_srec_pps_query_history_embedding_v1",
        },
    }
    write_json(run_dir / "metadata.json", output)
    return output
