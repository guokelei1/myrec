"""Ordinary supervised adaptation for decoder-only CrossEncoder rerankers."""

from __future__ import annotations

import math
import random
import shutil
import time
from pathlib import Path
from typing import Any

from myrec.baselines.full_token_training import build_pairwise_examples
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


def train_decoder_cross_encoder(
    standardized_dir: str | Path,
    run_id: str,
    output_model_dir: str | Path,
    *,
    input_mode: str,
    runs_dir: str | Path = "runs",
    config_path: str | Path | None = None,
    base_model_name: str = "models/huggingface/Qwen3-Reranker-0.6B",
    cache_folder: str | Path = "models/huggingface/cross_encoders",
    local_files_only: bool = True,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    max_length: int = 512,
    history_budget: int = 5,
    negatives_per_positive: int = 2,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    epochs: int = 1,
    learning_rate: float = 1e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    seed: int = 20260714,
    gradient_checkpointing: bool = False,
) -> dict[str, Any]:
    """Adapt one decoder reranker with ordinary pointwise BCE on train labels.

    The QC and FULL variants use the same deterministic positive/negative pairs,
    order, optimizer recipe, and number of updates. Their only intended input
    difference is whether the serialized context contains prior history.
    """

    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported dtype={dtype}")
    if batch_size <= 0 or gradient_accumulation_steps <= 0 or epochs <= 0:
        raise ValueError("batch size, accumulation, and epochs must be positive")
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
    from sentence_transformers import CrossEncoder
    from torch.nn import functional as F
    from torch.utils.data import DataLoader
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
    model = CrossEncoder(
        base_model_name,
        cache_folder=str(cache_folder),
        device=device,
        max_length=max_length,
        trust_remote_code=True,
        local_files_only=local_files_only,
        # Keep fp32 master weights. Autocast controls the training kernels.
        model_kwargs={"dtype": torch.float32},
    )
    transformer = getattr(model[0], "model", None)
    if transformer is None:
        raise TypeError("CrossEncoder first module does not expose a transformer model")
    if hasattr(transformer.config, "use_cache"):
        transformer.config.use_cache = False
    if gradient_checkpointing:
        transformer.gradient_checkpointing_enable()
    model.train()
    prompt = model._resolve_prompt(None, None)

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
            pairs = [
                (example.context, example.positive_document) for example in batch
            ] + [
                (example.context, example.negative_document) for example in batch
            ]
            features = model.preprocess(pairs, prompt=prompt)
            features = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in features.items()
            }
            with torch.autocast(
                device_type="cuda",
                dtype=torch_dtype,
                enabled=autocast_enabled,
            ):
                scores = model(features)["scores"].reshape(-1)
                labels = torch.cat(
                    (
                        torch.ones(len(batch), device=device, dtype=scores.dtype),
                        torch.zeros(len(batch), device=device, dtype=scores.dtype),
                    )
                )
                raw_loss = F.binary_cross_entropy_with_logits(scores, labels)
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
    model.save_pretrained(
        str(output_model_dir), create_model_card=False, safe_serialization=True
    )
    weights_path = output_model_dir / "model.safetensors"
    if not weights_path.exists():
        raise FileNotFoundError(f"saved checkpoint missing: {weights_path}")
    checkpoint_id = (
        f"{input_mode}-qwen3-pointwise@{sha256_file(weights_path)[:20]}"
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
        "objective": "pointwise_binary_cross_entropy",
        "output_model_dir": str(output_model_dir),
        "package_versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "preprocess_prompt": prompt,
        "records_train_sha256": sha256_file(records_path),
        "run_id": run_id,
        "seed": seed,
        "training": {
            "batch_size": batch_size,
            "candidate_presentations": 2 * len(examples) * epochs,
            "epochs": epochs,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "history_budget": history_budget,
            "learning_rate": learning_rate,
            "max_grad_norm": max_grad_norm,
            "max_length": max_length,
            "mean_microbatch_loss": loss_sum / micro_steps,
            "micro_steps": micro_steps,
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
