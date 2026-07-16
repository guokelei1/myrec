"""InstructRec-style seq2seq candidate-likelihood ranker for PPS.

The paper's T3 setting treats personalized search as instruction following and
reranks a fixed candidate list by the likelihood of each candidate response.
This adapter keeps that boundary: training reads only train records/qrels and
scoring reads label-free records plus a pre-materialized history assignment.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.baselines.full_token_cross_encoder import _load_assignments
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


@dataclass(frozen=True)
class InstructRecExample:
    request_id: str
    prompt: str
    target: str


def candidate_text(candidate: dict[str, Any]) -> str:
    categories = "/".join(str(value) for value in candidate.get("cat", []) if value)
    values = [
        str(candidate.get("title") or "").strip(),
        str(candidate.get("brand") or "").strip(),
        categories,
    ]
    return " | ".join(value for value in values if value) or "[EMPTY PRODUCT]"


def history_text(history: list[dict[str, Any]], history_budget: int) -> str:
    selected = history[-history_budget:] if history_budget else []
    if not selected:
        return "(empty)"
    rows = []
    for index, event in enumerate(selected, start=1):
        categories = "/".join(str(value) for value in event.get("cat", []) if value)
        values = [
            str(event.get("event") or "interaction"),
            str(event.get("title") or ""),
            str(event.get("brand") or ""),
            categories,
        ]
        rows.append(f"{index}. " + " | ".join(value for value in values if value))
    return "\n".join(rows)


def serialize_instructrec_prompt(
    record: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    history_budget: int,
) -> str:
    candidates = "\n".join(
        f"{index}. {candidate_text(candidate)}"
        for index, candidate in enumerate(record["candidates"], start=1)
    )
    return (
        "You are a personalized product-search engine. The user has a current "
        "search query and a history of prior product interactions. Select the "
        "single candidate product that best satisfies this user's query and "
        "preferences. Output only the exact selected product text.\n\n"
        "User history:\n"
        f"{history_text(history, history_budget)}\n\n"
        "Current query:\n"
        f"{str(record['query'])}\n\n"
        "Candidate products:\n"
        f"{candidates}\n\n"
        "Selected product:"
    )


def build_instructrec_examples(
    records_path: str | Path,
    qrels_path: str | Path,
    *,
    input_mode: str,
    history_budget: int,
    seed: int,
) -> tuple[list[InstructRecExample], dict[str, Any]]:
    """Build one candidate-selection target per train request/positive."""

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

    examples: list[InstructRecExample] = []
    seen: set[str] = set()
    positive_requests = 0
    skipped_requests = 0
    for record in iter_jsonl(records_path):
        request_id = str(record["request_id"])
        if request_id in seen:
            raise ValueError(f"duplicate train record request_id={request_id}")
        seen.add(request_id)
        positives = qrels.get(request_id)
        if positives is None:
            raise ValueError(f"missing train qrels request_id={request_id}")
        slate = list(record["candidates"])
        positive_candidates = [
            candidate
            for candidate in slate
            if str(candidate["item_id"]) in positives
        ]
        if not positive_candidates:
            skipped_requests += 1
            continue
        positive_requests += 1
        history = list(record.get("history", [])) if input_mode == "full" else []
        prompt = serialize_instructrec_prompt(
            record, history, history_budget=history_budget
        )
        for positive in positive_candidates:
            examples.append(
                InstructRecExample(
                    request_id=request_id,
                    prompt=prompt,
                    target=candidate_text(positive),
                )
            )
    if seen != set(qrels):
        raise ValueError("train records and qrels have different request coverage")
    if not examples:
        raise ValueError("no InstructRec training examples were constructed")
    random.Random(seed).shuffle(examples)
    return examples, {
        "examples": len(examples),
        "input_mode": input_mode,
        "history_budget": history_budget,
        "labeled_requests": positive_requests,
        "requests": len(seen),
        "skipped_no_positive": skipped_requests,
        "target_definition": "candidate text for each train clicked/purchased positive",
    }


def train_instructrec(
    standardized_dir: str | Path,
    run_id: str,
    output_model_dir: str | Path,
    *,
    input_mode: str,
    base_model_name: str = "google/flan-t5-xl",
    cache_folder: str | Path = "models/huggingface/llm",
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    local_files_only: bool = True,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_source_length: int = 2048,
    max_target_length: int = 64,
    history_budget: int = 6,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = 20260716,
    gradient_checkpointing: bool = True,
    max_train_examples: int | None = None,
) -> dict[str, Any]:
    """Train an InstructRec T3 candidate-likelihood model on train labels."""

    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported dtype={dtype}")
    if batch_size <= 0 or gradient_accumulation_steps <= 0 or epochs <= 0:
        raise ValueError("batch size, accumulation, and epochs must be positive")
    if max_train_examples is not None and max_train_examples <= 0:
        raise ValueError("max_train_examples must be positive when provided")
    standardized_dir = Path(standardized_dir)
    run_dir = Path(runs_dir) / run_id
    output_model_dir = Path(output_model_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    if output_model_dir.exists() and any(output_model_dir.iterdir()):
        raise FileExistsError(f"model directory is not empty: {output_model_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    output_model_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    examples, example_stats = build_instructrec_examples(
        records_path,
        qrels_path,
        input_mode=input_mode,
        history_budget=history_budget,
        seed=seed,
    )
    examples_before_cap = len(examples)
    if max_train_examples is not None:
        examples = examples[:max_train_examples]
    example_stats = {
        **example_stats,
        "examples_before_cap": examples_before_cap,
        "examples_after_cap": len(examples),
    }

    import torch
    import transformers
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.optimization import get_linear_schedule_with_warmup

    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        cache_dir=str(cache_folder),
        local_files_only=local_files_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name,
        cache_dir=str(cache_folder),
        local_files_only=local_files_only,
        dtype=torch_dtype if dtype != "float32" else torch.float32,
    ).to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    model.train()

    def collate(batch: list[InstructRecExample]) -> dict[str, Any]:
        inputs = tokenizer(
            [example.prompt for example in batch],
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        targets = tokenizer(
            text_target=[example.target for example in batch],
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        labels = targets["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return inputs

    loader = DataLoader(
        examples,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=collate,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    updates_per_epoch = math.ceil(len(loader) / gradient_accumulation_steps)
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * warmup_ratio),
        num_training_steps=total_updates,
    )
    autocast_enabled = device.startswith("cuda") and dtype != "float32"
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == "float16")
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    micro_steps = 0
    optimizer_steps = 0
    started = time.perf_counter()
    for _epoch in range(epochs):
        for batch_index, batch in enumerate(loader, start=1):
            batch = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type="cuda",
                dtype=torch_dtype,
                enabled=autocast_enabled,
            ):
                raw_loss = model(**batch).loss
                loss = raw_loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
            loss_sum += float(raw_loss.detach().cpu())
            micro_steps += 1
            if (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == len(loader)
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1

    elapsed = time.perf_counter() - started
    model.save_pretrained(str(output_model_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_model_dir))
    weight_files = sorted(
        path for path in output_model_dir.iterdir() if path.name.endswith((".safetensors", ".bin"))
    )
    if not weight_files:
        raise FileNotFoundError(f"saved InstructRec weights missing: {output_model_dir}")
    metadata = {
        "base_model_name": base_model_name,
        "checkpoint_id": f"instructrec-{input_mode}@{sha256_file(weight_files[0])[:20]}",
        "config_path": str(config_path) if config_path else None,
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "dev_labels_read": False,
        "confirmation_labels_read": False,
        "device": device,
        "dtype": dtype,
        "elapsed_seconds": elapsed,
        "example_stats": example_stats,
        "input_mode": input_mode,
        "objective": "instructrec_t3_candidate_text_negative_log_likelihood",
        "output_model_dir": str(output_model_dir),
        "package_versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "records_train_sha256": sha256_file(records_path),
        "run_id": run_id,
        "seed": seed,
        "training": {
            "batch_size": batch_size,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "history_budget": history_budget,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "max_train_examples": max_train_examples,
            "mean_microbatch_loss": loss_sum / micro_steps,
            "micro_steps": micro_steps,
            "optimizer_steps": optimizer_steps,
            "warmup_ratio": warmup_ratio,
            "weight_decay": weight_decay,
        },
        "training_labels_path": str(qrels_path),
        "training_labels_read": True,
        "training_labels_sha256": sha256_file(qrels_path),
        "weights_sha256": sha256_file(weight_files[0]),
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(output_model_dir / "myrec_training_metadata.json", metadata)
    if config_path and Path(config_path).exists():
        shutil.copyfile(config_path, run_dir / f"config_snapshot{Path(config_path).suffix}")
    return metadata


def write_instructrec_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    history_condition: str,
    history_assignments_path: str | Path,
    model_dir: str | Path,
    *,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    cache_folder: str | Path = "models/huggingface/llm",
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_source_length: int = 2048,
    max_target_length: int = 64,
    history_budget: int = 6,
    batch_size: int = 8,
    local_files_only: bool = True,
    method_id: str = "instructrec_t3",
    max_requests: int | None = None,
) -> dict[str, Any]:
    """Score every candidate by normalized target-text log likelihood."""

    if split == "test":
        raise ValueError("test scoring is locked")
    if history_condition not in {"true", "null", "wrong"}:
        raise ValueError(f"unsupported history condition: {history_condition}")
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError(f"unsupported dtype={dtype}")
    if max_requests is not None and max_requests <= 0:
        raise ValueError("max_requests must be positive when provided")
    standardized_dir = Path(standardized_dir)
    run_dir = Path(runs_dir) / run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = Path(history_assignments_path)
    assignments = _load_assignments(assignment_path, expected_condition=history_condition)
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    dataset_manifest_path = standardized_dir / "manifest.json"

    import torch
    import transformers
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.modeling_outputs import BaseModelOutput

    torch_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype]
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), cache_dir=str(cache_folder), local_files_only=local_files_only
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(model_dir),
        cache_dir=str(cache_folder),
        local_files_only=local_files_only,
        dtype=torch_dtype if dtype != "float32" else torch.float32,
    ).to(device)
    model.eval()

    def score_targets_with_cached_encoder(
        prompt: str, targets: list[str]
    ) -> list[float]:
        """Score targets while encoding the unchanged request prompt once."""

        prompt_inputs = tokenizer(
            [prompt],
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        input_ids = prompt_inputs["input_ids"].to(device)
        attention_mask = prompt_inputs["attention_mask"].to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch_dtype,
            enabled=device.startswith("cuda") and dtype != "float32",
        ):
            encoded = model.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )

        target_values: list[float] = []
        for start in range(0, len(targets), batch_size):
            target_batch = targets[start : start + batch_size]
            target_tokens = tokenizer(
                text_target=target_batch,
                padding=True,
                truncation=True,
                max_length=max_target_length,
                return_tensors="pt",
            )
            labels = target_tokens["input_ids"]
            labels[labels == tokenizer.pad_token_id] = -100
            labels = labels.to(device)
            count = len(target_batch)
            hidden = encoded.last_hidden_state.expand(
                count, -1, -1
            ).contiguous()
            repeated_attention_mask = attention_mask.expand(count, -1)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch_dtype,
                enabled=device.startswith("cuda") and dtype != "float32",
            ):
                logits = model(
                    encoder_outputs=BaseModelOutput(last_hidden_state=hidden),
                    attention_mask=repeated_attention_mask,
                    labels=labels,
                ).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)
                safe_labels = labels.masked_fill(labels.eq(-100), 0)
                token_log_probs = log_probs.gather(
                    2, safe_labels.unsqueeze(-1)
                ).squeeze(-1)
                mask = labels.ne(-100)
                values = (token_log_probs * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp_min(1)
            target_values.extend(float(value.detach().cpu()) for value in values)
        return target_values

    def score_batch(prompts: list[str], targets: list[str]) -> list[float]:
        """Compatibility helper for the mechanical path."""

        if len(set(prompts)) == 1:
            return score_targets_with_cached_encoder(prompts[0], targets)
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_source_length,
            return_tensors="pt",
        )
        target_tokens = tokenizer(
            text_target=targets,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )
        labels = target_tokens["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        inputs = {key: value.to(device) for key, value in inputs.items()}
        labels = labels.to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch_dtype,
            enabled=device.startswith("cuda") and dtype != "float32",
        ):
            logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            safe_labels = labels.masked_fill(labels.eq(-100), 0)
            token_log_probs = log_probs.gather(2, safe_labels.unsqueeze(-1)).squeeze(-1)
            mask = labels.ne(-100)
            values = (token_log_probs * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return [float(value.detach().cpu()) for value in values]

    scores_path = run_dir / "scores.jsonl"
    rows = 0
    requests = 0
    started = time.perf_counter()
    with scores_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(standardized_dir / f"records_{split}.jsonl"):
            if max_requests is not None and requests >= max_requests:
                break
            request_id = str(record["request_id"])
            if request_id not in assignments:
                raise ValueError(f"history assignment missing request_id={request_id}")
            history = assignments[request_id]["history"]
            prompt = serialize_instructrec_prompt(
                record, history, history_budget=history_budget
            )
            candidates = list(record["candidates"])
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start : start + batch_size]
                values = score_batch(
                    [prompt] * len(batch), [candidate_text(candidate) for candidate in batch]
                )
                for candidate, score in zip(batch, values):
                    if not math.isfinite(score):
                        raise ValueError(f"non-finite score for {request_id}")
                    handle.write(
                        json.dumps(
                            {
                                "candidate_item_id": str(candidate["item_id"]),
                                "method_id": method_id,
                                "request_id": request_id,
                                "score": score,
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    rows += 1
            requests += 1

    with dataset_manifest_path.open("r", encoding="utf-8") as handle:
        dataset_manifest = json.load(handle)
    elapsed = time.perf_counter() - started
    metadata = {
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "checkpoint_id": f"{Path(model_dir).name}@{sha256_file(Path(model_dir) / 'config.json')[:20]}",
        "config_path": str(config_path) if config_path else None,
        "dataset_id": str(dataset_manifest["dataset_id"]),
        "dataset_version": str(dataset_manifest["dataset_version"]),
        "device": device,
        "dtype": dtype,
        "elapsed_seconds": elapsed,
        "history_assignment_sha256": sha256_file(assignment_path),
        "history_assignments_path": str(assignment_path),
        "history_condition": history_condition,
        "input_fields_used": [
            "assigned_history.event",
            "assigned_history.title",
            "assigned_history.brand",
            "assigned_history.cat",
            "query",
            "candidates.title",
            "candidates.brand",
            "candidates.cat",
            "full_candidate_slate_in_prompt",
        ],
        "local_files_only": local_files_only,
        "max_requests": max_requests,
        "method_id": method_id,
        "model_dir": str(model_dir),
        "package_versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        "qrels_read": False,
        "request_count": requests,
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "run_id": run_id,
        "score_definition": "normalized seq2seq log likelihood of each candidate text under the InstructRec T3 prompt",
        "score_rows": rows,
        "scoring_signature": {
            "batch_size": batch_size,
            "candidate_list": "all_frozen_candidates_in_prompt",
            "history_budget": history_budget,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "model_family": "encoder_decoder_seq2seq",
            "target_score": "length_normalized_candidate_text_log_likelihood",
        },
        "split": split,
        "standardized_dir": str(standardized_dir),
        "tuning": {"class": "bounded_train_only_recipe", "dev_labels_used_during_scoring": False},
    }
    if config_path and Path(config_path).exists():
        shutil.copyfile(config_path, run_dir / f"config_snapshot{Path(config_path).suffix}")
    write_json(run_dir / "metadata.json", metadata)
    return metadata
