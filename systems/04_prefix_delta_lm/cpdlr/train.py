"""Minimal train/internal execution for C04 and its preregistered controls."""

from __future__ import annotations

import json
import os
import random
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from transformers import AutoTokenizer

from .data import ProbeDataset, collate_requests, move_inputs, reshape_logits
from .io import (
    assert_candidate_manifest,
    assert_train_only_path,
    sha256_file,
    write_json,
)
from .losses import compute_probe_loss
from .model import PrefixDeltaRanker
from .tokenization import PrefixTokenizer


CORRUPTIONS = ("wrong", "shuffled", "query_masked", "coarse")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def _request_batches(
    dataset: ProbeDataset, batch_size: int, seed: int, shuffle: bool
) -> Iterator[list[dict[str, Any]]]:
    indices = np.arange(len(dataset), dtype=np.int64)
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [dataset[int(index)] for index in indices[start : start + batch_size]]


def _tokenizer(config: dict[str, Any]) -> PrefixTokenizer:
    model_cfg = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["backbone"],
        local_files_only=bool(model_cfg["local_files_only"]),
        use_fast=False,
    )
    return PrefixTokenizer(
        tokenizer,
        max_length=int(model_cfg["max_length"]),
        query_tokens=int(model_cfg["query_tokens"]),
        candidate_tokens=int(model_cfg["candidate_tokens"]),
        max_history_events=int(model_cfg["max_history_events"]),
        event_tokens=int(model_cfg["event_tokens"]),
    )


def _forward(
    model: PrefixDeltaRanker,
    batch: dict[str, Any],
    device: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    shape = batch["shape"]
    joint_inputs = {
        key: torch.cat([batch["factual_inputs"][key], batch["null_inputs"][key]], dim=0)
        for key in batch["factual_inputs"]
    }
    joint_scores = model.score(move_inputs(joint_inputs, device))
    branch_size = shape[0] * shape[1]
    factual = reshape_logits(joint_scores[:branch_size], shape)
    null = reshape_logits(joint_scores[branch_size:], shape)
    candidate_mask = batch["candidate_mask"].to(device)
    history_present = batch["history_present"].to(device)
    exact_repeat = batch["exact_repeat"].to(device)
    outputs = model.combine(
        factual,
        null,
        candidate_mask,
        history_present,
        exact_repeat=exact_repeat,
    )
    corruption_delta = None
    if "corrupt_inputs" in batch:
        corrupt = reshape_logits(
            model.score(move_inputs(batch["corrupt_inputs"], device)), shape
        )
        corrupt_null = reshape_logits(
            model.score(move_inputs(batch["corrupt_null_inputs"], device)), shape
        )
        _, corruption_delta = model.tangent_delta(
            corrupt,
            corrupt_null,
            candidate_mask,
            history_present,
        )
    return outputs, corruption_delta


def _control_loss_config(config: dict[str, Any], mode: str) -> dict[str, Any]:
    if mode == "paired_delta":
        return config
    copied = dict(config)
    copied["loss"] = dict(config["loss"])
    copied["loss"].update(
        {
            "consistency_weight": 0.0,
            "repeat_weight": 0.0,
            "transfer_weight": 0.0,
        }
    )
    return copied


def _diagnostic_rows(
    outputs: dict[str, torch.Tensor], batch: dict[str, Any]
) -> dict[str, float]:
    mask = batch["candidate_mask"].to(outputs["final"].device)
    labels = batch["labels"].to(outputs["final"].device)
    repeat = batch["exact_repeat"].to(outputs["final"].device)
    delta = outputs["tangent_delta"]
    positive = (labels > 0) & mask
    negative = (labels <= 0) & mask
    positive_margin = []
    concordant = 0
    comparable = 0
    teacher = batch["teacher_scores"].to(outputs["final"].device)
    null = outputs["null"]
    for row in range(mask.shape[0]):
        if positive[row].any() and negative[row].any():
            positive_margin.append(
                float(
                    outputs["final"][row][positive[row]].mean()
                    - outputs["final"][row][negative[row]].mean()
                )
            )
        valid = torch.nonzero(mask[row], as_tuple=False).flatten()
        for left_offset in range(len(valid)):
            for right_offset in range(left_offset + 1, len(valid)):
                left = valid[left_offset]
                right = valid[right_offset]
                teacher_diff = teacher[row, left] - teacher[row, right]
                if teacher_diff == 0:
                    continue
                null_diff = null[row, left] - null[row, right]
                comparable += 1
                concordant += int(torch.sign(teacher_diff) == torch.sign(null_diff))
    repeat_values = delta.masked_select(repeat & mask)
    nonrepeat_values = delta.masked_select((~repeat) & mask)
    return {
        "anchor_pair_concordance": concordant / comparable if comparable else 0.0,
        "mean_abs_delta": float(delta.masked_select(mask).abs().mean()),
        "mean_delta_nonrepeat": float(nonrepeat_values.mean()) if nonrepeat_values.numel() else 0.0,
        "mean_delta_repeat": float(repeat_values.mean()) if repeat_values.numel() else 0.0,
        "positive_negative_logit_margin": (
            sum(positive_margin) / len(positive_margin) if positive_margin else 0.0
        ),
    }


def _evaluate_internal(
    model: PrefixDeltaRanker,
    dataset: ProbeDataset,
    tokenizer: PrefixTokenizer,
    config: dict[str, Any],
    device: str,
    mode: str,
) -> dict[str, float]:
    model.eval()
    loss_totals: dict[str, float] = {}
    diagnostic_totals: dict[str, float] = {}
    batches = 0
    structured = mode != "concat_head"
    loss_config = _control_loss_config(config, mode)
    with torch.inference_mode():
        for rows in _request_batches(
            dataset,
            int(config["training"]["batch_requests"]),
            int(config["seed"]),
            False,
        ):
            batch = collate_requests(
                rows,
                tokenizer,
                int(config["seed"]),
                None,
                structured=structured,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda"),
            ):
                outputs, _ = _forward(model, batch, device)
                _, loss_rows = compute_probe_loss(
                    outputs,
                    {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    },
                    loss_config,
                )
            diagnostics = _diagnostic_rows(outputs, batch)
            for key, value in loss_rows.items():
                loss_totals[key] = loss_totals.get(key, 0.0) + value
            for key, value in diagnostics.items():
                diagnostic_totals[key] = diagnostic_totals.get(key, 0.0) + value
            batches += 1
    return {
        **{f"loss_{key}": value / batches for key, value in loss_totals.items()},
        **{key: value / batches for key, value in diagnostic_totals.items()},
        "batches": float(batches),
        "requests": float(len(dataset)),
    }


def train_probe(
    config: dict[str, Any],
    config_path: str | Path,
    mode: str,
    run_id: str,
    device: str,
) -> dict[str, Any]:
    if not run_id.startswith("20260710_kuaisearch_c04_"):
        raise ValueError(f"invalid C04 run prefix: {run_id}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise ValueError("C04 GPU commands must set CUDA_VISIBLE_DEVICES=3")
    paths = config["paths"]
    assert_train_only_path(paths["probe_train"])
    assert_train_only_path(paths["probe_internal"])
    candidate_hash = assert_candidate_manifest(
        paths["candidate_manifest"], config["candidate_manifest_sha256"]
    )
    set_seed(int(config["seed"]))
    started = time.time()
    mode_cfg = config["controls"].get(mode, {})
    train_limit = mode_cfg.get("train_request_limit")
    internal_limit = mode_cfg.get("internal_request_limit")
    epochs = int(mode_cfg.get("epochs", config["training"]["epochs"]))
    train_data = ProbeDataset(paths["probe_train"], limit=train_limit)
    internal_data = ProbeDataset(paths["probe_internal"], limit=internal_limit)
    tokenizer = _tokenizer(config)
    model = PrefixDeltaRanker(config, mode=mode).to(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    accumulation = int(config["training"]["gradient_accumulation"])
    batch_requests = int(config["training"]["batch_requests"])
    structured = mode != "concat_head"
    loss_config = _control_loss_config(config, mode)
    epoch_rows = []
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        totals: dict[str, float] = {}
        batches = 0
        for batch_index, rows in enumerate(
            _request_batches(
                train_data,
                batch_requests,
                int(config["seed"]) + epoch,
                True,
            )
        ):
            corruption = (
                CORRUPTIONS[(global_step + epoch - 1) % len(CORRUPTIONS)]
                if mode == "paired_delta"
                else None
            )
            batch = collate_requests(
                rows,
                tokenizer,
                int(config["seed"]),
                corruption,
                structured=structured,
            )
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda"),
            ):
                outputs, corruption_delta = _forward(model, batch, device)
                device_batch = {
                    key: value.to(device) if isinstance(value, torch.Tensor) else value
                    for key, value in batch.items()
                }
                loss, loss_rows = compute_probe_loss(
                    outputs,
                    device_batch,
                    loss_config,
                    corruption_delta=corruption_delta,
                )
                scaled_loss = loss / accumulation
            scaled_loss.backward()
            if (batch_index + 1) % accumulation == 0:
                torch.nn.utils.clip_grad_norm_(
                    parameters, float(config["training"]["gradient_clip_norm"])
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for key, value in loss_rows.items():
                totals[key] = totals.get(key, 0.0) + value
            batches += 1
            global_step += 1
        if batches % accumulation:
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["training"]["gradient_clip_norm"])
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        internal = _evaluate_internal(
            model, internal_data, tokenizer, config, device, mode
        )
        epoch_rows.append(
            {
                "epoch": epoch,
                "internal": internal,
                "train_losses": {key: value / batches for key, value in totals.items()},
            }
        )

    checkpoint_path = Path(config["paths"]["model_dir"]) / f"{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "candidate_id": config["candidate_id"],
            "config_sha256": sha256_file(config_path),
            "mode": mode,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "seed": int(config["seed"]),
        },
        checkpoint_path,
    )
    elapsed = time.time() - started
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(config_path, run_dir / "config_snapshot.yaml")
    summary = {
        "candidate_id": config["candidate_id"],
        "candidate_manifest_sha256": candidate_hash,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
        "elapsed_seconds": elapsed,
        "epochs": epoch_rows,
        "gpu_hours": elapsed / 3600.0,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "hostname": socket.gethostname(),
        "lora_modules": model.lora_modules,
        "mode": mode,
        "qrels_read": False,
        "run_id": run_id,
        "seed": int(config["seed"]),
        "test_read": False,
        "total_parameters": model.total_parameter_count(),
        "train_requests": len(train_data),
        "trainable_parameters": model.trainable_parameter_count(),
    }
    write_json(run_dir / "train_summary.json", summary)
    write_json(
        run_dir / "metadata.json",
        {
            "candidate_manifest_sha256": candidate_hash,
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "cuda_visible_devices": "3",
            "dataset_id": "kuaisearch",
            "dataset_version": "v0_lite",
            "env_group": "system-04",
            "env_name": "myrec-c04",
            "git_commit": config["lock_context"]["git_commit"],
            "git_dirty": True,
            "method_id": f"c04_{mode}",
            "qrels_read": False,
            "run_id": run_id,
            "seed": int(config["seed"]),
            "split": "train_internal",
            "test_read": False,
        },
    )
    return summary
