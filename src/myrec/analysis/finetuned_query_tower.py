"""Fine-tuned non-personalized query-tower controls for PPS motivation."""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer

from myrec.analysis.supervised_diagnostics import (
    PackedRequestData,
    collate_request_batch,
    multi_positive_listwise_loss,
)
from myrec.eval.metrics import ScoredCandidate, request_metrics
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


class FineTunedQueryTower(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        item_embeddings: torch.Tensor,
        logit_scale_initial: float,
        logit_scale_bounds: tuple[float, float],
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.item_embeddings = item_embeddings
        dimension = int(item_embeddings.shape[1])
        hidden_size = int(encoder.config.hidden_size)
        if hidden_size != dimension:
            raise ValueError(f"encoder/item dimensions differ: {hidden_size} != {dimension}")
        self.item_adapter = nn.Linear(dimension, dimension, bias=False)
        with torch.no_grad():
            self.item_adapter.weight.copy_(torch.eye(dimension))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(logit_scale_initial)))
        self.logit_scale_bounds = logit_scale_bounds

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        candidate_indices: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> torch.Tensor:
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        query = F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6)
        candidate = self.item_embeddings[candidate_indices].float()
        candidate = F.normalize(self.item_adapter(candidate), dim=-1, eps=1e-6)
        lower, upper = self.logit_scale_bounds
        scale = self.logit_scale.exp().clamp(min=lower, max=upper)
        scores = scale * torch.einsum("bd,bcd->bc", query, candidate)
        return scores.masked_fill(~candidate_mask, -torch.finfo(scores.dtype).max)


def materialize_query_tokens(
    config: dict[str, Any], config_path: str | Path
) -> dict[str, Any]:
    packed_root = Path(config["packed_data_dir"])
    output_dir = Path(config["tokenized_queries"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    standardized_dir = Path(config["standardized_dir"])
    tokenizer = AutoTokenizer.from_pretrained(
        config["encoder"]["model_name"],
        local_files_only=bool(config["encoder"]["local_files_only"]),
    )
    split_rows = {}
    output_files = {}
    for split in ("train", "dev"):
        packed = PackedRequestData.load(packed_root, split)
        expected = set(packed.request_ids)
        request_ids = []
        queries = []
        records_path = standardized_dir / f"records_{split}.jsonl"
        for record in iter_jsonl(records_path):
            request_id = str(record["request_id"])
            if request_id not in expected:
                continue
            request_ids.append(request_id)
            queries.append(str(record["query"]))
        if request_ids != packed.request_ids:
            raise ValueError(f"tokenized query order mismatch for {split}")
        encoded = tokenizer(
            queries,
            padding="max_length",
            truncation=True,
            max_length=int(config["encoder"]["max_length"]),
            return_tensors="np",
        )
        input_ids = np.asarray(encoded["input_ids"], dtype=np.int32)
        attention_mask = np.asarray(encoded["attention_mask"], dtype=np.uint8)
        input_path = output_dir / f"{split}_input_ids.npy"
        mask_path = output_dir / f"{split}_attention_mask.npy"
        np.save(input_path, input_ids)
        np.save(mask_path, attention_mask)
        split_rows[split] = len(request_ids)
        output_files[split] = {
            "attention_mask": {"path": str(mask_path), "sha256": sha256_file(mask_path)},
            "input_ids": {"path": str(input_path), "sha256": sha256_file(input_path)},
            "records_path": str(records_path),
            "records_sha256": sha256_file(records_path),
        }
    report = {
        "analysis_id": config["analysis_id"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "max_length": int(config["encoder"]["max_length"]),
        "model_name": config["encoder"]["model_name"],
        "output_files": output_files,
        "qrels_read": False,
        "request_counts": split_rows,
        "test_read": False,
    }
    write_json(output_dir / "manifest.json", report)
    return report


def load_tokens(config: dict[str, Any], split: str) -> tuple[np.ndarray, np.ndarray]:
    root = Path(config["tokenized_queries"]["output_dir"])
    return (
        np.load(root / f"{split}_input_ids.npy", mmap_mode="r"),
        np.load(root / f"{split}_attention_mask.npy", mmap_mode="r"),
    )


def build_model(config: dict[str, Any], device: str) -> FineTunedQueryTower:
    encoder = AutoModel.from_pretrained(
        config["encoder"]["model_name"],
        local_files_only=bool(config["encoder"]["local_files_only"]),
    )
    item_embeddings = torch.from_numpy(
        np.array(
            np.load(config["encoder"]["frozen_item_embeddings"], mmap_mode="r"),
            copy=True,
        )
    ).to(device)
    bounds = tuple(float(value) for value in config["encoder"]["logit_scale_bounds"])
    return FineTunedQueryTower(
        encoder=encoder,
        item_embeddings=item_embeddings,
        logit_scale_initial=float(config["encoder"]["logit_scale_initial"]),
        logit_scale_bounds=bounds,
    ).to(device)


def iter_query_batches(
    data: PackedRequestData,
    request_indices: np.ndarray,
    max_requests: int,
    max_padded_candidates: int,
    seed: int,
    shuffle: bool,
) -> Iterator[dict[str, np.ndarray]]:
    indices = np.asarray(request_indices, dtype=np.int64).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(indices)
    batch = []
    max_candidates = 0
    for request_index in indices:
        request_index = int(request_index)
        count = int(
            data.candidate_offsets[request_index + 1]
            - data.candidate_offsets[request_index]
        )
        next_size = len(batch) + 1
        next_max = max(max_candidates, count)
        if batch and (
            next_size > max_requests
            or next_size * next_max > max_padded_candidates
        ):
            yield collate_request_batch(data, np.asarray(batch, dtype=np.int64))
            batch = []
            max_candidates = 0
        batch.append(request_index)
        max_candidates = max(max_candidates, count)
    if batch:
        yield collate_request_batch(data, np.asarray(batch, dtype=np.int64))


def count_query_batches(
    data: PackedRequestData,
    request_indices: np.ndarray,
    max_requests: int,
    max_padded_candidates: int,
    seed: int,
) -> int:
    return sum(
        1
        for _ in iter_query_batches(
            data,
            request_indices,
            max_requests,
            max_padded_candidates,
            seed,
            True,
        )
    )


def initialization_reproduction_check(
    model: FineTunedQueryTower,
    data: PackedRequestData,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    frozen_query_embeddings_path: str | Path,
    device: str,
    sample_size: int = 128,
) -> dict[str, Any]:
    count = min(sample_size, len(data))
    indices = np.arange(count, dtype=np.int64)
    model.eval()
    with torch.inference_mode():
        output = model.encoder(
            input_ids=torch.from_numpy(np.asarray(input_ids[indices], dtype=np.int64)).to(device),
            attention_mask=torch.from_numpy(
                np.asarray(attention_mask[indices], dtype=np.int64)
            ).to(device),
        )
        current = F.normalize(output.last_hidden_state[:, 0, :].float(), dim=-1)
    frozen = np.load(frozen_query_embeddings_path, mmap_mode="r")
    expected = torch.from_numpy(
        np.asarray(frozen[np.asarray(data.query_indices[indices], dtype=np.int64)], dtype=np.float32).copy()
    ).to(device)
    expected = F.normalize(expected, dim=-1)
    cosine = (current * expected).sum(dim=-1).cpu().numpy()
    result = {
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
        "sample_size": count,
        "status": "passed" if cosine.mean() >= 0.999 and cosine.min() >= 0.995 else "failed",
    }
    if result["status"] != "passed":
        raise AssertionError(f"query encoder initialization mismatch: {result}")
    return result


def train_query_tower(
    config: dict[str, Any],
    config_path: str | Path,
    mode: str,
    seed: int,
    device: str,
    epochs: int | None = None,
) -> dict[str, Any]:
    if mode not in {"calibrate", "final"}:
        raise ValueError(f"unknown mode: {mode}")
    _set_seed(seed)
    data = PackedRequestData.load(config["packed_data_dir"], "train")
    input_ids, attention_mask = load_tokens(config, "train")
    if len(input_ids) != len(data):
        raise ValueError("train token count mismatch")
    manifest = _read_json(Path(config["packed_data_dir"]) / "manifest.json")
    cut = int(manifest["internal_calibration"]["cut_request_index"])
    if mode == "calibrate":
        train_indices = np.arange(cut, dtype=np.int64)
        validation_indices = np.arange(cut, len(data), dtype=np.int64)
        max_epochs = int(config["internal_calibration"]["max_epochs"])
    else:
        if epochs is None or epochs < 1:
            raise ValueError("final mode requires positive epochs")
        train_indices = np.arange(len(data), dtype=np.int64)
        validation_indices = None
        max_epochs = int(epochs)

    model = build_model(config, device)
    embedding_root = Path("artifacts/batch2b/b5o_stageb_standardized/data")
    reproduction = initialization_reproduction_check(
        model,
        data,
        input_ids,
        attention_mask,
        embedding_root / "query_emb.npy",
        device,
    )
    encoder_parameters = list(model.encoder.parameters())
    adapter_parameters = list(model.item_adapter.parameters()) + [model.logit_scale]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": float(config["training"]["encoder_learning_rate"]),
            },
            {
                "params": adapter_parameters,
                "lr": float(config["training"]["adapter_learning_rate"]),
            },
        ],
        weight_decay=float(config["training"]["weight_decay"]),
    )
    max_requests = int(config["training"]["max_requests_per_batch"])
    max_candidates = int(config["training"]["max_padded_candidates_per_batch"])
    total_steps = sum(
        count_query_batches(
            data,
            train_indices,
            max_requests,
            max_candidates,
            seed + epoch,
        )
        for epoch in range(1, max_epochs + 1)
    )
    warmup_steps = int(round(total_steps * float(config["training"]["warmup_ratio"])))

    def learning_rate_factor(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-6)
        remaining = max(total_steps - warmup_steps, 1)
        return max((total_steps - step) / remaining, 0.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
    run_id = f"20260710_kuaisearch_d2t_{mode}_s{seed}"
    run_dir = Path("runs") / run_id
    checkpoint_dir = Path("models/diagnostics") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "model.pt"
    started = time.time()
    best_metric = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0
    epoch_rows = []
    global_step = 0
    patience = int(config["internal_calibration"]["patience"])
    min_delta = float(config["internal_calibration"]["min_delta"])
    amp_enabled = str(device).startswith("cuda")

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_total = 0.0
        batches = 0
        for batch in iter_query_batches(
            data,
            train_indices,
            max_requests,
            max_candidates,
            seed + epoch,
            True,
        ):
            request_indices = batch["request_indices"]
            token_ids = torch.from_numpy(
                np.asarray(input_ids[request_indices], dtype=np.int64)
            ).to(device)
            token_mask = torch.from_numpy(
                np.asarray(attention_mask[request_indices], dtype=np.int64)
            ).to(device)
            candidate_indices = torch.from_numpy(batch["candidate_indices"]).to(device)
            candidate_mask = torch.from_numpy(batch["candidate_mask"]).to(device)
            labels = torch.from_numpy(batch["candidate_labels"]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=amp_enabled,
            ):
                scores = model(
                    token_ids, token_mask, candidate_indices, candidate_mask
                )
                loss = multi_positive_listwise_loss(scores, labels, candidate_mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite D2 loss at epoch {epoch}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["training"]["gradient_clip_norm"])
            )
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite D2 gradient at epoch {epoch}")
            optimizer.step()
            scheduler.step()
            global_step += 1
            loss_total += float(loss.detach().cpu())
            batches += 1
        row = {
            "batches": batches,
            "epoch": epoch,
            "train_loss": loss_total / batches,
        }
        if validation_indices is not None:
            metric = evaluate_query_tower(
                model,
                data,
                input_ids,
                attention_mask,
                validation_indices,
                max_requests,
                max_candidates,
                device,
            )
            row["internal_validation_ndcg@10"] = metric
            if metric > best_metric + min_delta:
                best_metric = metric
                best_epoch = epoch
                _save_checkpoint(model, checkpoint_path, config, seed, epoch)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
        else:
            best_epoch = epoch
            _save_checkpoint(model, checkpoint_path, config, seed, epoch)
        epoch_rows.append(row)
        if validation_indices is not None and epochs_without_improvement >= patience:
            break

    alpha_result = None
    if validation_indices is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        alpha_result = calibrate_popularity_alpha(
            model,
            data,
            input_ids,
            attention_mask,
            validation_indices,
            np.load(
                Path(config["packed_data_dir"])
                / "item_log_click_internal_train.npy",
                mmap_mode="r",
            ),
            [float(value) for value in config["internal_calibration"]["d2p_alpha_grid"]],
            max_requests,
            max_candidates,
            device,
        )

    summary = {
        "alpha_calibration": alpha_result,
        "analysis_id": config["analysis_id"],
        "best_epoch": best_epoch,
        "best_internal_validation_ndcg@10": (
            best_metric if validation_indices is not None else None
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "elapsed_seconds": time.time() - started,
        "epochs": epoch_rows,
        "final_epochs_requested": epochs,
        "initialization_reproduction": reproduction,
        "mode": mode,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "qrels_read": False,
        "run_id": run_id,
        "seed": seed,
        "test_read": False,
        "total_scheduler_steps": total_steps,
        "train_requests": len(train_indices),
        "variant": "d2t",
    }
    _copy_config(config_path, run_dir)
    write_json(run_dir / "train_summary.json", summary)
    return summary


def evaluate_query_tower(
    model: FineTunedQueryTower,
    data: PackedRequestData,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    request_indices: np.ndarray,
    max_requests: int,
    max_candidates: int,
    device: str,
) -> float:
    model.eval()
    total = 0.0
    requests = 0
    with torch.inference_mode():
        for batch in iter_query_batches(
            data,
            request_indices,
            max_requests,
            max_candidates,
            0,
            False,
        ):
            indices = batch["request_indices"]
            scores = model(
                torch.from_numpy(np.asarray(input_ids[indices], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(attention_mask[indices], dtype=np.int64)).to(device),
                torch.from_numpy(batch["candidate_indices"]).to(device),
                torch.from_numpy(batch["candidate_mask"]).to(device),
            ).float().cpu().numpy()
            for row, request_index in enumerate(indices):
                request_index = int(request_index)
                count = int(batch["candidate_mask"][row].sum())
                labels = batch["candidate_labels"][row, :count]
                item_ids = batch["candidate_item_ids"][row, :count]
                total += _request_ndcg(
                    data.request_ids[request_index], item_ids, scores[row, :count], labels
                )
                requests += 1
    return total / requests


def calibrate_popularity_alpha(
    model: FineTunedQueryTower,
    data: PackedRequestData,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    request_indices: np.ndarray,
    item_log_click: np.ndarray,
    alpha_grid: list[float],
    max_requests: int,
    max_candidates: int,
    device: str,
) -> dict[str, Any]:
    totals = {str(alpha): 0.0 for alpha in alpha_grid}
    requests = 0
    model.eval()
    with torch.inference_mode():
        for batch in iter_query_batches(
            data, request_indices, max_requests, max_candidates, 0, False
        ):
            indices = batch["request_indices"]
            scores = model(
                torch.from_numpy(np.asarray(input_ids[indices], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(attention_mask[indices], dtype=np.int64)).to(device),
                torch.from_numpy(batch["candidate_indices"]).to(device),
                torch.from_numpy(batch["candidate_mask"]).to(device),
            ).float().cpu().numpy()
            for row, request_index in enumerate(indices):
                request_index = int(request_index)
                count = int(batch["candidate_mask"][row].sum())
                query_score = _zscore(scores[row, :count])
                popularity = _zscore(
                    np.asarray(item_log_click[batch["candidate_indices"][row, :count]])
                )
                labels = batch["candidate_labels"][row, :count]
                item_ids = batch["candidate_item_ids"][row, :count]
                for alpha in alpha_grid:
                    mixed = alpha * query_score + (1.0 - alpha) * popularity
                    totals[str(alpha)] += _request_ndcg(
                        data.request_ids[request_index], item_ids, mixed, labels
                    )
                requests += 1
    metrics = {key: value / requests for key, value in totals.items()}
    selected = max(alpha_grid, key=lambda alpha: (metrics[str(alpha)], alpha))
    return {
        "alpha_metrics": metrics,
        "request_count": requests,
        "selected_alpha": selected,
        "selected_ndcg@10": metrics[str(selected)],
        "tie_break": "largest alpha (more text) wins",
    }


def calibrate_history_alpha(
    model: FineTunedQueryTower,
    data: PackedRequestData,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    request_indices: np.ndarray,
    alpha_grid: list[float],
    max_requests: int,
    max_candidates: int,
    device: str,
) -> dict[str, Any]:
    totals = {str(alpha): 0.0 for alpha in alpha_grid}
    requests = 0
    model.eval()
    with torch.inference_mode():
        for batch in iter_query_batches(
            data, request_indices, max_requests, max_candidates, 0, False
        ):
            indices = batch["request_indices"]
            scores = model(
                torch.from_numpy(np.asarray(input_ids[indices], dtype=np.int64)).to(device),
                torch.from_numpy(np.asarray(attention_mask[indices], dtype=np.int64)).to(device),
                torch.from_numpy(batch["candidate_indices"]).to(device),
                torch.from_numpy(batch["candidate_mask"]).to(device),
            ).float().cpu().numpy()
            for row, request_index in enumerate(indices):
                request_index = int(request_index)
                count = int(batch["candidate_mask"][row].sum())
                query_score = _zscore(scores[row, :count])
                history_score = _zscore(batch["candidate_b0b"][row, :count])
                labels = batch["candidate_labels"][row, :count]
                item_ids = batch["candidate_item_ids"][row, :count]
                for alpha in alpha_grid:
                    mixed = alpha * query_score + (1.0 - alpha) * history_score
                    totals[str(alpha)] += _request_ndcg(
                        data.request_ids[request_index], item_ids, mixed, labels
                    )
                requests += 1
    metrics = {key: value / requests for key, value in totals.items()}
    selected = max(alpha_grid, key=lambda alpha: (metrics[str(alpha)], alpha))
    return {
        "alpha_metrics": metrics,
        "request_count": requests,
        "selected_alpha": selected,
        "selected_ndcg@10": metrics[str(selected)],
        "tie_break": "largest alpha (more text) wins",
    }


def compose_d2p_history_scores(
    text_scores: np.ndarray,
    popularity_scores: np.ndarray,
    history_scores: np.ndarray,
    d2p_alpha: float,
    beta: float,
) -> np.ndarray:
    """Compose the frozen D2p score with a static history score."""
    d2p = d2p_alpha * _zscore(text_scores) + (1.0 - d2p_alpha) * _zscore(
        popularity_scores
    )
    return beta * _zscore(d2p) + (1.0 - beta) * _zscore(history_scores)


def calibrate_d2p_history_beta(
    model: FineTunedQueryTower,
    data: PackedRequestData,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    request_indices: np.ndarray,
    item_log_click: np.ndarray,
    d2p_alpha: float,
    beta_grid: list[float],
    max_requests: int,
    max_candidates: int,
    device: str,
) -> dict[str, Any]:
    totals = {str(beta): 0.0 for beta in beta_grid}
    requests = 0
    model.eval()
    with torch.inference_mode():
        for batch in iter_query_batches(
            data, request_indices, max_requests, max_candidates, 0, False
        ):
            indices = batch["request_indices"]
            scores = model(
                torch.from_numpy(np.asarray(input_ids[indices], dtype=np.int64)).to(
                    device
                ),
                torch.from_numpy(
                    np.asarray(attention_mask[indices], dtype=np.int64)
                ).to(device),
                torch.from_numpy(batch["candidate_indices"]).to(device),
                torch.from_numpy(batch["candidate_mask"]).to(device),
            ).float().cpu().numpy()
            for row, request_index in enumerate(indices):
                request_index = int(request_index)
                count = int(batch["candidate_mask"][row].sum())
                text_scores = scores[row, :count]
                popularity_scores = np.asarray(
                    item_log_click[batch["candidate_indices"][row, :count]]
                )
                history_scores = batch["candidate_b0b"][row, :count]
                labels = batch["candidate_labels"][row, :count]
                item_ids = batch["candidate_item_ids"][row, :count]
                for beta in beta_grid:
                    mixed = compose_d2p_history_scores(
                        text_scores,
                        popularity_scores,
                        history_scores,
                        d2p_alpha,
                        beta,
                    )
                    totals[str(beta)] += _request_ndcg(
                        data.request_ids[request_index], item_ids, mixed, labels
                    )
                requests += 1
    metrics = {key: value / requests for key, value in totals.items()}
    selected = max(beta_grid, key=lambda beta: (metrics[str(beta)], beta))
    return {
        "beta_metrics": metrics,
        "d2p_alpha": d2p_alpha,
        "request_count": requests,
        "selected_beta": selected,
        "selected_ndcg@10": metrics[str(selected)],
        "tie_break": "largest beta (more D2p) wins",
    }


def score_query_tower(
    config: dict[str, Any],
    config_path: str | Path,
    final_config: dict[str, Any],
    final_config_path: str | Path,
    seed: int,
    checkpoint_path: str | Path,
    device: str,
) -> dict[str, Any]:
    data = PackedRequestData.load(config["packed_data_dir"], "dev")
    input_ids, attention_mask = load_tokens(config, "dev")
    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    alpha = float(final_config["final_training"]["d2p_alpha"])
    item_log_click = np.load(
        Path(config["packed_data_dir"]) / "item_log_click_full_train.npy",
        mmap_mode="r",
    )
    run_ids = {
        "d2t": f"20260710_kuaisearch_d2t_finetuned_text_dev_s{seed}",
        "d2p": f"20260710_kuaisearch_d2p_text_pop_dev_s{seed}",
    }
    handles = {}
    started = time.time()
    try:
        for variant, identifier in run_ids.items():
            run_dir = Path("runs") / identifier
            run_dir.mkdir(parents=True, exist_ok=True)
            handles[variant] = (run_dir / "scores.jsonl").open("w", encoding="utf-8")
        rows = 0
        with torch.inference_mode():
            for batch in iter_query_batches(
                data,
                np.arange(len(data), dtype=np.int64),
                int(config["training"]["max_requests_per_batch"]),
                int(config["training"]["max_padded_candidates_per_batch"]),
                0,
                False,
            ):
                indices = batch["request_indices"]
                raw_scores = model(
                    torch.from_numpy(np.asarray(input_ids[indices], dtype=np.int64)).to(device),
                    torch.from_numpy(np.asarray(attention_mask[indices], dtype=np.int64)).to(device),
                    torch.from_numpy(batch["candidate_indices"]).to(device),
                    torch.from_numpy(batch["candidate_mask"]).to(device),
                ).float().cpu().numpy()
                for row, request_index in enumerate(indices):
                    request_index = int(request_index)
                    count = int(batch["candidate_mask"][row].sum())
                    text_score = raw_scores[row, :count]
                    text_z = _zscore(text_score)
                    popularity_z = _zscore(
                        np.asarray(
                            item_log_click[batch["candidate_indices"][row, :count]]
                        )
                    )
                    mixed = alpha * text_z + (1.0 - alpha) * popularity_z
                    request_id = data.request_ids[request_index]
                    for offset in range(count):
                        item_id = str(batch["candidate_item_ids"][row, offset])
                        for variant, value in (
                            ("d2t", text_score[offset]),
                            ("d2p", mixed[offset]),
                        ):
                            handles[variant].write(
                                json.dumps(
                                    {
                                        "candidate_item_id": item_id,
                                        "method_id": (
                                            "d2t_finetuned_text"
                                            if variant == "d2t"
                                            else "d2p_text_popularity"
                                        ),
                                        "request_id": request_id,
                                        "score": float(value),
                                    },
                                    ensure_ascii=False,
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        rows += 1
    finally:
        for handle in handles.values():
            handle.close()

    manifest_path = Path(config["standardized_dir"]) / "candidate_manifest.json"
    results = {}
    for variant, identifier in run_ids.items():
        run_dir = Path("runs") / identifier
        metadata = {
            "alpha": alpha if variant == "d2p" else None,
            "analysis_id": config["analysis_id"],
            "candidate_manifest_path": str(manifest_path),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "dataset_id": config["dataset_id"],
            "dataset_version": config["dataset_version"],
            "elapsed_seconds_shared_scoring": time.time() - started,
            "final_config_path": str(final_config_path),
            "final_config_sha256": sha256_file(final_config_path),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "input_fields_used": (
                ["query", "candidate title"]
                if variant == "d2t"
                else ["query", "candidate title", "train-only item click count"]
            ),
            "method_id": (
                "d2t_finetuned_text" if variant == "d2t" else "d2p_text_popularity"
            ),
            "qrels_read": False,
            "request_count": len(data),
            "run_id": identifier,
            "score_rows": rows,
            "seed": seed,
            "split": "dev",
            "test_read": False,
            "variant": variant,
        }
        _copy_config(config_path, run_dir)
        write_json(run_dir / "metadata.json", metadata)
        results[variant] = metadata
    return results


def _request_ndcg(
    request_id: str,
    item_ids: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
) -> float:
    row = request_metrics(
        request_id=request_id,
        scored_candidates=[
            ScoredCandidate(str(item_id), float(score))
            for item_id, score in zip(item_ids, scores)
        ],
        clicked_item_ids={
            str(item_id) for item_id, label in zip(item_ids, labels) if label > 0
        },
        purchased_item_ids=set(),
    )
    return float(row["ndcg@10"])


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / math.sqrt(float(values.var()) + 1e-6)


def _save_checkpoint(
    model: FineTunedQueryTower,
    path: Path,
    config: dict[str, Any],
    seed: int,
    epoch: int,
) -> None:
    torch.save(
        {
            "analysis_id": config["analysis_id"],
            "epoch": epoch,
            "model_state": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "seed": seed,
        },
        path,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _copy_config(config_path: str | Path, run_dir: Path) -> None:
    shutil.copyfile(config_path, run_dir / "config_snapshot.yaml")


def _read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
