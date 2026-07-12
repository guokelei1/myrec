#!/usr/bin/env python
"""Train one full-token HSO seed and score the unopened Amazon reserve."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_history_observability import (  # noqa: E402
    TokenHistoryCrossEncoder,
    TokenHistoryData,
    listwise_loss,
    pack_training_batch,
)
from prepare_amazon_token_history_observability import load_config, verify_lock  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def assert_device(config: dict[str, Any], seed: int, device_name: str) -> torch.device:
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("token HSO physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("token HSO requires exactly one visible GPU")
    return torch.device(device_name)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def load_fit_labels(path: Path) -> dict[int, np.ndarray]:
    with np.load(path, allow_pickle=False) as value:
        indices = np.asarray(value["request_indices"], dtype=np.int64)
        offsets = np.asarray(value["offsets"], dtype=np.int64)
        labels = np.asarray(value["labels"], dtype=np.float32)
    return {
        int(index): labels[int(offsets[row]) : int(offsets[row + 1])].copy()
        for row, index in enumerate(indices)
    }


def train(
    model: TokenHistoryCrossEncoder,
    data: TokenHistoryData,
    labels: dict[int, np.ndarray],
    config: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    row = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.backbone.parameters(),
                "lr": float(row["backbone_learning_rate"]),
            },
            {
                "params": model.score_head.parameters(),
                "lr": float(row["head_learning_rate"]),
            },
        ],
        weight_decay=float(row["weight_decay"]),
    )
    losses = []
    active_backbone = False
    active_head = False
    steps = 0
    fit = np.asarray(data.fit_indices, dtype=np.int64)
    for epoch in range(int(row["epochs"])):
        order = fit.copy()
        np.random.default_rng(seed + epoch * 1009).shuffle(order)
        candidate_rng = np.random.default_rng(seed + epoch * 1009 + 17)
        dropout_rng = np.random.default_rng(seed + epoch * 1009 + 29)
        model.train()
        for start in range(0, len(order), int(row["requests_per_batch"])):
            requests = order[start : start + int(row["requests_per_batch"])]
            input_ids, attention, label_rows = pack_training_batch(
                data,
                requests,
                labels,
                sampled_candidates=int(row["sampled_candidates"]),
                candidate_rng=candidate_rng,
                dropout_rng=dropout_rng,
                history_dropout=float(row["history_dropout"]),
                token_config=config["tokens"],
            )
            ids = torch.from_numpy(input_ids).to(device)
            mask = torch.from_numpy(attention).to(device)
            target = torch.from_numpy(label_rows).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model(ids, mask).reshape(len(requests), -1)
                loss = listwise_loss(scores, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("token HSO nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"token HSO nonfinite gradient: {name}")
                if bool(parameter.grad.ne(0).any()):
                    active_backbone |= name.startswith("backbone.")
                    active_head |= name.startswith("score_head.")
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(row["gradient_clip_norm"])
            )
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            steps += 1
    window = min(100, max(1, len(losses) // 4))
    return {
        "steps": steps,
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:]))
        < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "backbone_gradient": active_backbone,
        "head_gradient": active_head,
    }


def score_candidates(
    model: TokenHistoryCrossEncoder,
    data: TokenHistoryData,
    request_index: int,
    candidates: np.ndarray,
    scenario: str,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    output = []
    batch_size = int(config["evaluation"]["encoded_candidates_per_batch"])
    token = config["tokens"]
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            rows = [
                data.pack_candidate(
                    request_index,
                    int(candidate),
                    scenario=scenario,
                    query_tokens=int(token["query_tokens"]),
                    candidate_tokens=int(token["candidate_tokens"]),
                    history_item_tokens=int(token["history_item_tokens"]),
                    max_history=int(token["max_history"]),
                    max_length=int(token["max_sequence_length"]),
                )
                for candidate in candidates[start : start + batch_size]
            ]
            ids = torch.from_numpy(np.asarray([row[0] for row in rows], dtype=np.int64)).to(device)
            mask = torch.from_numpy(np.asarray([row[1] for row in rows], dtype=bool)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model(ids, mask)
            output.append(values.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def score_reserve(
    model: TokenHistoryCrossEncoder,
    data: TokenHistoryData,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    flat = {name: [] for name in ("true", "null", "wrong", "shuffle")}
    offsets = [0]
    request_indices = []
    deterministic = 0.0
    permutation = 0.0
    checked = False
    for index_value in data.reserve_indices:
        index = int(index_value)
        candidates = data.candidates(index)
        for scenario in flat:
            flat[scenario].append(
                score_candidates(model, data, index, candidates, scenario, config, device)
            )
        if not checked:
            repeated = score_candidates(model, data, index, candidates, "true", config, device)
            deterministic = float(np.max(np.abs(flat["true"][-1] - repeated)))
            reversed_scores = score_candidates(
                model, data, index, candidates[::-1].copy(), "true", config, device
            )[::-1]
            permutation = float(np.max(np.abs(flat["true"][-1] - reversed_scores)))
            checked = True
        offsets.append(offsets[-1] + len(candidates))
        request_indices.append(index)
    values = {
        name: np.concatenate(rows).astype(np.float32, copy=False)
        for name, rows in flat.items()
    }
    values["request_indices"] = np.asarray(request_indices, dtype=np.int64)
    values["offsets"] = np.asarray(offsets, dtype=np.int64)
    return values, {
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": permutation,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.seed not in [int(value) for value in config["training"]["seeds"]]:
        raise ValueError("token HSO seed differs")
    _, lock_hash = verify_lock(config, config_path)
    device = assert_device(config, args.seed, args.device)
    seed_all(args.seed)
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    report_path = root / f"seed_{args.seed}_report.json"
    score_path = root / f"seed_{args.seed}_scores.npz"
    checkpoint_path = checkpoint_root / f"seed_{args.seed}.pt"
    for path in (report_path, score_path, checkpoint_path):
        if path.exists():
            raise FileExistsError(path)
    data = TokenHistoryData(root)
    labels = load_fit_labels(root / "fit_labels.npz")
    if set(labels) != set(data.fit_indices.tolist()):
        raise RuntimeError("token HSO compact fit labels differ")
    model = TokenHistoryCrossEncoder(
        ROOT / paths["bge_snapshot"],
        score_head_bias=bool(config["model"]["score_head_bias"]),
    ).to(device)
    training = train(model, data, labels, config, seed=args.seed, device=device)
    torch.save(
        {
            "analysis_id": config["analysis_id"],
            "seed": args.seed,
            "model_state": model.state_dict(),
        },
        checkpoint_path,
    )
    scores, numeric = score_reserve(model, data, config, device)
    temporary = score_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **scores)
    temporary.replace(score_path)
    checks = {
        "loss_decreased": bool(training["loss_decreased"]),
        "finite": bool(training["finite"]),
        "backbone_gradient": bool(training["backbone_gradient"]),
        "head_gradient": bool(training["head_gradient"]),
        "deterministic": numeric["deterministic_max_abs"]
        <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": numeric["candidate_permutation_max_abs"]
        <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "candidate_hash": data.candidate_hash(scores["request_indices"])
        == json.loads((root / "token_manifest.json").read_text(encoding="utf-8"))[
            "candidate_hash_reserve"
        ],
        "reserve_labels_opened": False,
        "dev_test_qrels_closed": True,
    }
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "full_token_fit_and_label_free_reserve_scoring",
        "seed": args.seed,
        "execution_lock_sha256": lock_hash,
        "fit_requests": len(data.fit_indices),
        "reserve_requests": len(data.reserve_indices),
        "parameters": model.parameter_count(),
        "training": training,
        "scoring": {
            "path": str(score_path.relative_to(ROOT)),
            "sha256": sha256_file(score_path),
            "candidate_rows": int(scores["offsets"][-1]),
            **numeric,
        },
        "checkpoint": {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint_path),
        },
        "checks": checks,
        "passed_mechanics": all(checks.values()),
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "loss_first": training["loss_first"],
                "loss_last": training["loss_last"],
                "mechanics": report["passed_mechanics"],
                "score_sha256": report["scoring"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
