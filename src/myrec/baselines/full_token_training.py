"""Train ordinary query-candidate and full-token cross-encoder controls."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from myrec.baselines.core import document_text
from myrec.baselines.full_token_cross_encoder import serialize_query_history
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


@dataclass(frozen=True)
class PairwiseExample:
    request_id: str
    context: str
    positive_document: str
    negative_document: str


def build_pairwise_examples(
    records_path: str | Path,
    qrels_path: str | Path,
    *,
    input_mode: str,
    history_budget: int,
    negatives_per_positive: int,
    seed: int,
) -> tuple[list[PairwiseExample], dict[str, Any]]:
    """Build deterministic train-only pairs without reading development labels."""

    if input_mode not in {"qc", "full"}:
        raise ValueError(f"unsupported input_mode={input_mode}")
    if negatives_per_positive <= 0:
        raise ValueError("negatives_per_positive must be positive")
    qrels = {}
    for row in iter_jsonl(qrels_path):
        request_id = str(row["request_id"])
        if request_id in qrels:
            raise ValueError(f"duplicate qrels request_id={request_id}")
        qrels[request_id] = {
            *(str(item_id) for item_id in row.get("clicked", [])),
            *(str(item_id) for item_id in row.get("purchased", [])),
        }

    examples = []
    seen_requests = set()
    labeled_requests = 0
    skipped_no_positive = 0
    skipped_no_negative = 0
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        if request_id in seen_requests:
            raise ValueError(f"duplicate record request_id={request_id}")
        seen_requests.add(request_id)
        if request_id not in qrels:
            raise ValueError(f"missing train qrels request_id={request_id}")
        positive_ids = qrels[request_id]
        candidates = list(record["candidates"])
        candidate_ids = {str(candidate["item_id"]) for candidate in candidates}
        unknown_positive = positive_ids - candidate_ids
        if unknown_positive:
            raise ValueError(
                f"positive labels outside candidate slate for {request_id}: "
                f"{sorted(unknown_positive)[:5]}"
            )
        positives = [
            candidate for candidate in candidates if str(candidate["item_id"]) in positive_ids
        ]
        negatives = [
            candidate for candidate in candidates if str(candidate["item_id"]) not in positive_ids
        ]
        if not positives:
            skipped_no_positive += 1
            continue
        if not negatives:
            skipped_no_negative += 1
            continue
        labeled_requests += 1
        history = list(record.get("history", [])) if input_mode == "full" else []
        serialization_version = (
            "query_history_event_text_v1"
            if input_mode == "full"
            else "query_only_text_v1"
        )
        context = serialize_query_history(
            str(record["query"]),
            history,
            history_budget=history_budget,
            serialization_version=serialization_version,
        )
        for positive in positives:
            positive_id = str(positive["item_id"])
            local_seed = _stable_seed(seed, request_id, positive_id)
            sampled_negatives = list(negatives)
            random.Random(local_seed).shuffle(sampled_negatives)
            for negative in sampled_negatives[:negatives_per_positive]:
                examples.append(
                    PairwiseExample(
                        request_id=request_id,
                        context=context,
                        positive_document=document_text(positive),
                        negative_document=document_text(negative),
                    )
                )
    if seen_requests != set(qrels):
        raise ValueError("train records and qrels have different request coverage")
    if not examples:
        raise ValueError("no pairwise training examples were constructed")
    stats = {
        "examples": len(examples),
        "input_mode": input_mode,
        "serialization_version": (
            "query_history_event_text_v1"
            if input_mode == "full"
            else "query_only_text_v1"
        ),
        "labeled_requests": labeled_requests,
        "negatives_per_positive": negatives_per_positive,
        "requests": len(seen_requests),
        "skipped_no_negative": skipped_no_negative,
        "skipped_no_positive": skipped_no_positive,
    }
    return examples, stats


def train_pairwise_cross_encoder(
    standardized_dir: str | Path,
    run_id: str,
    output_model_dir: str | Path,
    *,
    input_mode: str,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    base_model_name: str = "BAAI/bge-reranker-base",
    cache_folder: str | Path = "models/huggingface/cross_encoders",
    local_files_only: bool = True,
    device: str = "cuda:0",
    dtype: str = "float16",
    max_length: int = 512,
    history_budget: int = 10,
    negatives_per_positive: int = 2,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    epochs: int = 2,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = 20260714,
    objective: str = "pairwise_logistic_softplus",
    truncation_strategy: str = "longest_first",
) -> dict[str, Any]:
    """Train one fixed-recipe ordinary ranker with a standard ranking loss."""

    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported dtype={dtype}")
    if batch_size <= 0 or gradient_accumulation_steps <= 0 or epochs <= 0:
        raise ValueError("batch size, accumulation, and epochs must be positive")
    if objective not in {
        "pairwise_logistic_softplus",
        "pointwise_binary_cross_entropy",
    }:
        raise ValueError(f"unsupported objective={objective}")
    if truncation_strategy not in {"longest_first", "only_second"}:
        raise ValueError(f"unsupported truncation_strategy={truncation_strategy}")
    standardized_dir = Path(standardized_dir)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    output_model_dir = Path(output_model_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    if output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"model directory is not empty: {output_model_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    examples, example_stats = build_pairwise_examples(
        records_path,
        qrels_path,
        input_mode=input_mode,
        history_budget=history_budget,
        negatives_per_positive=negatives_per_positive,
        seed=seed,
    )

    import sentence_transformers
    import torch
    import transformers
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=str(cache_folder),
        local_files_only=local_files_only,
        trust_remote_code=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        cache_dir=str(cache_folder),
        local_files_only=local_files_only,
        trust_remote_code=True,
        # Mixed-precision training keeps master parameters in fp32 and uses
        # autocast for the forward/backward kernels. Loading parameters as
        # fp16 makes GradScaler unable to unscale their gradients.
        dtype=torch.float32,
    )
    model.to(device)
    model.train()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=lambda batch: batch,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    updates_per_epoch = math.ceil(len(loader) / gradient_accumulation_steps)
    total_updates = updates_per_epoch * epochs
    warmup_steps = int(total_updates * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_updates
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == "float16")
    autocast_enabled = device.startswith("cuda") and dtype != "float32"
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    micro_steps = 0
    optimizer_steps = 0
    started = time.perf_counter()
    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader, start=1):
            contexts = [example.context for example in batch]
            documents = [example.positive_document for example in batch] + [
                example.negative_document for example in batch
            ]
            repeated_contexts = contexts + contexts
            encoded = tokenizer(
                repeated_contexts,
                documents,
                padding=True,
                truncation=truncation_strategy,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.autocast(
                device_type="cuda",
                dtype=torch_dtype,
                enabled=autocast_enabled,
            ):
                logits = model(**encoded).logits.reshape(-1)
                positive_logits = logits[: len(batch)]
                negative_logits = logits[len(batch) :]
                if objective == "pairwise_logistic_softplus":
                    raw_loss = F.softplus(
                        -(positive_logits - negative_logits)
                    ).mean()
                else:
                    raw_loss = F.binary_cross_entropy_with_logits(
                        torch.cat((positive_logits, negative_logits)),
                        torch.cat(
                            (
                                torch.ones_like(positive_logits),
                                torch.zeros_like(negative_logits),
                            )
                        ),
                    )
                loss = raw_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            micro_steps += 1
            loss_sum += float(raw_loss.detach().cpu())
            should_step = (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == len(loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
    elapsed = time.perf_counter() - started
    model.save_pretrained(output_model_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_model_dir)
    weights_path = output_model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"saved checkpoint missing: {weights_path}")
    objective_short = (
        "pairwise" if objective == "pairwise_logistic_softplus" else "pointwise"
    )
    checkpoint_id = (
        f"{input_mode}-{objective_short}@{sha256_file(weights_path)[:20]}"
    )
    metadata = {
        "base_model_name": base_model_name,
        "checkpoint_id": checkpoint_id,
        "config_path": str(config_path) if config_path else None,
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "dev_labels_read": False,
        "device": device,
        "dtype": dtype,
        "parameter_dtype": "float32",
        "elapsed_seconds": elapsed,
        "example_stats": example_stats,
        "input_mode": input_mode,
        "objective": objective,
        "truncation_strategy": truncation_strategy,
        "output_model_dir": str(output_model_dir),
        "package_versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "records_train_sha256": sha256_file(records_path),
        "run_id": run_id,
        "seed": seed,
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "history_budget": history_budget,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "max_length": max_length,
            "micro_steps": micro_steps,
            "mean_microbatch_loss": loss_sum / micro_steps,
            "optimizer_steps": optimizer_steps,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
        },
        "training_labels_path": str(qrels_path),
        "training_labels_read": True,
        "training_labels_sha256": sha256_file(qrels_path),
        "weights_sha256": sha256_file(weights_path),
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(output_model_dir / "myrec_training_metadata.json", metadata)
    if config_path:
        config_path = Path(config_path)
        if config_path.exists():
            shutil.copyfile(config_path, run_dir / f"config_snapshot{config_path.suffix}")
    return metadata


def _stable_seed(seed: int, *parts: str) -> int:
    value = ":".join((str(seed), *parts)).encode("utf-8")
    return int(hashlib.sha256(value).hexdigest()[:16], 16)
