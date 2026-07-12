#!/usr/bin/env python
"""Train one C80 seed across all frozen modes and score fresh data label-free."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from freeze_lock import verify_lock  # noqa: E402
from model.pitt import (  # noqa: E402
    MODES,
    AuthenticatedHistoryRanker,
    GraphBatch,
    combine_candidate_scores,
    pack_candidate_graph,
    stack_graph_batches,
    tensor_sha256,
)
from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_history_observability import (  # noqa: E402
    TokenHistoryCrossEncoder,
    TokenHistoryData,
    listwise_loss,
    sample_positions,
)
from prepare_real_gate import load_config  # noqa: E402


SCENARIOS = ("true", "null", "wrong", "shuffle")


@dataclass(frozen=True)
class TrainingBatch:
    requests: np.ndarray
    candidates: np.ndarray
    labels: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def assert_device(config: dict[str, Any], seed: int, name: str) -> torch.device:
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C80 physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C80 requires exactly one visible GPU")
    return torch.device(name)


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


def checkpoint_state(config: dict[str, Any], seed: int) -> tuple[dict[str, torch.Tensor], Path]:
    base_seed = int(config["training"]["base_seed_map"][str(seed)])
    path = ROOT / config["paths"]["base_checkpoint_root"] / f"seed_{base_seed}.pt"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if int(checkpoint["seed"]) != base_seed:
        raise RuntimeError("C80 upstream checkpoint seed differs")
    return checkpoint["model_state"], path


def token_config(config: dict[str, Any], data: TokenHistoryData) -> dict[str, Any]:
    return {
        **config["tokens"],
        "cls_token_id": data.cls_token_id,
        "sep_token_id": data.sep_token_id,
        "pad_token_id": data.pad_token_id,
    }


def training_schedule(
    data: TokenHistoryData,
    labels: dict[int, np.ndarray],
    config: dict[str, Any],
    seed: int,
) -> list[TrainingBatch]:
    row = config["training"]
    output = []
    fit = np.asarray(data.fit_indices, dtype=np.int64)
    for epoch in range(int(row["epochs"])):
        order = fit.copy()
        np.random.default_rng(seed + epoch * 1009).shuffle(order)
        candidate_rng = np.random.default_rng(seed + epoch * 1009 + 17)
        for start in range(0, len(order), int(row["requests_per_batch"])):
            requests = order[start : start + int(row["requests_per_batch"])]
            candidates = []
            label_rows = []
            for index_value in requests:
                index = int(index_value)
                positions = sample_positions(
                    labels[index], int(row["sampled_candidates"]), candidate_rng
                )
                candidates.append(data.candidates(index)[positions])
                label_rows.append(labels[index][positions])
            output.append(
                TrainingBatch(
                    requests=requests.copy(),
                    candidates=np.asarray(candidates, dtype=np.int64),
                    labels=np.asarray(label_rows, dtype=np.float32),
                )
            )
    return output


def disable_dropout(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0


def standard_scores(
    model: TokenHistoryCrossEncoder,
    data: TokenHistoryData,
    request_index: int,
    candidates: np.ndarray,
    scenario: str,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    token = config["tokens"]
    batch_size = int(config["evaluation"]["encoded_candidates_per_batch"])
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            packed = [
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
            ids = torch.from_numpy(np.asarray([value[0] for value in packed], dtype=np.int64)).to(device)
            mask = torch.from_numpy(np.asarray([value[1] for value in packed], dtype=bool)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model(ids, mask)
            output.append(scores.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def score_external_surface(
    model: TokenHistoryCrossEncoder,
    data: TokenHistoryData,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    flat = {scenario: [] for scenario in SCENARIOS}
    offsets = [0]
    for index_value in data.reserve_indices:
        index = int(index_value)
        candidates = data.candidates(index)
        for scenario in SCENARIOS:
            flat[scenario].append(
                standard_scores(model, data, index, candidates, scenario, config, device)
            )
        offsets.append(offsets[-1] + len(candidates))
    arrays = {
        scenario: np.concatenate(values).astype(np.float32, copy=False)
        for scenario, values in flat.items()
    }
    return (
        arrays,
        np.asarray(data.reserve_indices, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
    )


def base_schedule_scores(
    model: TokenHistoryCrossEncoder,
    data: TokenHistoryData,
    schedule: Sequence[TrainingBatch],
    config: dict[str, Any],
    device: torch.device,
) -> list[np.ndarray]:
    output = []
    token = config["tokens"]
    batch_size = int(config["evaluation"]["encoded_candidates_per_batch"])
    model.eval()
    for batch in schedule:
        packed = []
        for request, candidates in zip(batch.requests, batch.candidates, strict=True):
            packed.extend(
                data.pack_candidate(
                    int(request),
                    int(candidate),
                    scenario="null",
                    query_tokens=int(token["query_tokens"]),
                    candidate_tokens=int(token["candidate_tokens"]),
                    history_item_tokens=int(token["history_item_tokens"]),
                    max_history=int(token["max_history"]),
                    max_length=int(token["max_sequence_length"]),
                )
                for candidate in candidates
            )
        values = []
        with torch.inference_mode():
            for start in range(0, len(packed), batch_size):
                rows = packed[start : start + batch_size]
                ids = torch.from_numpy(
                    np.asarray([value[0] for value in rows], dtype=np.int64)
                ).to(device)
                mask = torch.from_numpy(
                    np.asarray([value[1] for value in rows], dtype=bool)
                ).to(device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    scores = model(ids, mask)
                values.append(scores.float().cpu().numpy())
        output.append(np.concatenate(values).astype(np.float32, copy=False))
    return output


def pack_training_graph(
    data: TokenHistoryData,
    batch: TrainingBatch,
    token: dict[str, Any],
) -> GraphBatch:
    return stack_graph_batches(
        [
            pack_candidate_graph(
                data,
                int(request),
                int(candidate),
                scenario="true",
                token_config=token,
            )
            for request, candidates in zip(batch.requests, batch.candidates, strict=True)
            for candidate in candidates
        ]
    )


def train_mode(
    model: AuthenticatedHistoryRanker,
    data: TokenHistoryData,
    schedule: Sequence[TrainingBatch],
    base_scores: Sequence[np.ndarray],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    row = config["training"]
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(row["backbone_learning_rate"])},
            {"params": model.score_head.parameters(), "lr": float(row["head_learning_rate"])},
        ],
        weight_decay=float(row["weight_decay"]),
    )
    token = token_config(config, data)
    losses = []
    backbone_gradient = False
    head_gradient = False
    anchor_before = tensor_sha256(model.semantic_anchors)
    active_values = []
    model.train()
    for batch, base_values in zip(schedule, base_scores, strict=True):
        packed = pack_training_graph(data, batch, token)
        base = torch.from_numpy(base_values).to(device)
        labels = torch.from_numpy(batch.labels).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            raw, diagnostics = model.raw_correction(packed)
            scores = combine_candidate_scores(
                base,
                raw,
                [batch.candidates.shape[1]] * len(batch.requests),
            ).reshape(len(batch.requests), -1)
            loss = listwise_loss(scores, labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"C80 {model.mode} nonfinite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None:
                continue
            if not bool(torch.isfinite(parameter.grad).all()):
                raise RuntimeError(f"C80 {model.mode} nonfinite gradient: {name}")
            if bool(parameter.grad.ne(0).any()):
                backbone_gradient |= name.startswith("backbone.")
                head_gradient |= name.startswith("score_head.")
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(row["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        active_values.append(float(diagnostics["active_fraction"].cpu()))
    window = min(100, max(1, len(losses) // 4))
    return {
        "steps": len(losses),
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:])) < float(np.mean(losses[:window])),
        "finite": bool(np.isfinite(losses).all()),
        "backbone_gradient": backbone_gradient,
        "head_gradient": head_gradient,
        "anchor_has_no_gradient": model.semantic_anchors.grad is None,
        "anchor_unchanged": tensor_sha256(model.semantic_anchors) == anchor_before,
        "anchor_sha256": anchor_before,
        "mean_active_fraction": float(np.mean(active_values)),
    }


def mode_request_scores(
    model: AuthenticatedHistoryRanker,
    data: TokenHistoryData,
    request_index: int,
    candidates: np.ndarray,
    base_scores: np.ndarray,
    scenario: str,
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    if len(candidates) != len(base_scores):
        raise ValueError("C80 request base surface differs")
    token = token_config(config, data)
    batch_size = int(config["evaluation"]["encoded_candidates_per_batch"])
    raw_values = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            packed = stack_graph_batches(
                [
                    pack_candidate_graph(
                        data,
                        request_index,
                        int(candidate),
                        scenario=scenario,
                        token_config=token,
                    )
                    for candidate in candidates[start : start + batch_size]
                ]
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                raw, _ = model.raw_correction(packed)
            raw_values.append(raw.float().cpu())
    raw = torch.cat(raw_values)
    base = torch.from_numpy(np.asarray(base_scores, dtype=np.float32))
    scores = combine_candidate_scores(base, raw, [len(candidates)])
    return scores.numpy().astype(np.float32, copy=False)


def score_mode_surface(
    model: AuthenticatedHistoryRanker,
    data: TokenHistoryData,
    external_null: np.ndarray,
    offsets: np.ndarray,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    flat = {scenario: [] for scenario in SCENARIOS}
    deterministic = 0.0
    candidate_permutation = 0.0
    event_permutation = 0.0
    for row, index_value in enumerate(data.reserve_indices):
        index = int(index_value)
        candidates = data.candidates(index)
        start, stop = int(offsets[row]), int(offsets[row + 1])
        base = external_null[start:stop]
        values = {
            scenario: mode_request_scores(
                model, data, index, candidates, base, scenario, config, device
            )
            for scenario in SCENARIOS
        }
        for scenario in SCENARIOS:
            flat[scenario].append(values[scenario])
        if row == 0:
            repeated = mode_request_scores(
                model, data, index, candidates, base, "true", config, device
            )
            reversed_scores = mode_request_scores(
                model,
                data,
                index,
                candidates[::-1].copy(),
                base[::-1].copy(),
                "true",
                config,
                device,
            )[::-1]
            deterministic = float(np.max(np.abs(values["true"] - repeated)))
            candidate_permutation = float(np.max(np.abs(values["true"] - reversed_scores)))
            event_permutation = float(
                np.max(np.abs(values["true"] - values["shuffle"]))
            )
    arrays = {
        scenario: np.concatenate(values).astype(np.float32, copy=False)
        for scenario, values in flat.items()
    }
    return arrays, {
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": candidate_permutation,
        "event_permutation_max_abs": event_permutation,
        "nohistory_base_max_abs": float(np.max(np.abs(arrays["null"] - external_null))),
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    seeds = [int(value) for value in config["training"]["seeds"]]
    if args.seed not in seeds:
        raise ValueError("C80 seed differs")
    _, lock_hash = verify_lock(config, config_path)
    device = assert_device(config, args.seed, args.device)
    seed_all(args.seed)
    paths = config["paths"]
    fresh_root = ROOT / paths["fresh_root"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    report_path = fresh_root / f"seed_{args.seed}_report.json"
    score_path = fresh_root / f"seed_{args.seed}_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(report_path)

    fit_data = TokenHistoryData(ROOT / paths["fit_token_root"])
    fresh_data = TokenHistoryData(fresh_root)
    labels = load_fit_labels(ROOT / paths["fit_labels"])
    if set(labels) != set(int(value) for value in fit_data.fit_indices):
        raise RuntimeError("C80 fit labels differ from fit role")
    schedule = training_schedule(fit_data, labels, config, args.seed)
    state, upstream_checkpoint = checkpoint_state(config, args.seed)

    base_model = TokenHistoryCrossEncoder(
        ROOT / paths["bge_snapshot"],
        score_head_bias=bool(config["model"]["score_head_bias"]),
    ).to(device)
    base_model.load_state_dict(state, strict=True)
    disable_dropout(base_model)
    external, request_indices, offsets = score_external_surface(
        base_model, fresh_data, config, device
    )
    schedule_base = base_schedule_scores(
        base_model, fit_data, schedule, config, device
    )
    del base_model
    torch.cuda.empty_cache()

    all_scores: dict[str, np.ndarray] = {
        f"external_{scenario}": values for scenario, values in external.items()
    }
    all_scores["base"] = external["null"]
    all_scores["request_indices"] = request_indices
    all_scores["offsets"] = offsets
    training_reports: dict[str, Any] = {}
    mechanics: dict[str, Any] = {}
    parameter_counts = {}
    anchor_hashes = set()
    checkpoint_rows = {}
    tolerance = float(config["evaluation"]["deterministic_tolerance"])

    for mode in MODES:
        # Restore the identical registered initialization for every reduction.
        seed_all(args.seed)
        model = AuthenticatedHistoryRanker(
            ROOT / paths["bge_snapshot"],
            state,
            mode=mode,
            token_config=token_config(config, fit_data),
            correction_bound=float(config["model"]["correction_bound"]),
            score_head_bias=bool(config["model"]["score_head_bias"]),
        ).to(device)
        parameter_counts[mode] = model.trainable_parameter_count()
        training_reports[mode] = train_mode(
            model, fit_data, schedule, schedule_base, config, device
        )
        anchor_hashes.add(training_reports[mode]["anchor_sha256"])
        mode_scores, numeric = score_mode_surface(
            model,
            fresh_data,
            external["null"],
            offsets,
            config,
            device,
        )
        for scenario, values in mode_scores.items():
            all_scores[f"{mode}_{scenario}"] = values
        mechanics[mode] = {
            **numeric,
            "deterministic": numeric["deterministic_max_abs"] <= tolerance,
            "candidate_permutation": numeric["candidate_permutation_max_abs"]
            <= float(config["evaluation"]["candidate_permutation_tolerance"]),
            # The registered invariance is binding for the C80 primary.  The
            # controls retain their measured error as a diagnostic, but are
            # not themselves candidate acceptance conditions.
            "event_permutation": (
                numeric["event_permutation_max_abs"]
                <= float(config["evaluation"]["event_permutation_tolerance"])
                if mode == "triadic_set"
                else True
            ),
            "nohistory_exact": numeric["nohistory_base_max_abs"] == 0.0,
        }
        checkpoint_path = checkpoint_root / f"seed_{args.seed}_{mode}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        torch.save(
            {
                "candidate_id": "c80",
                "seed": args.seed,
                "mode": mode,
                "model_state": model.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint_rows[mode] = {
            "path": str(checkpoint_path.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint_path),
        }
        del model
        torch.cuda.empty_cache()

    temporary = score_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **all_scores)
    temporary.replace(score_path)
    manifest = json.loads((fresh_root / "token_manifest.json").read_text(encoding="utf-8"))
    upstream_report = json.loads(
        (
            ROOT
            / paths["fit_token_root"]
            / f"seed_{int(config['training']['base_seed_map'][str(args.seed)])}_report.json"
        ).read_text(encoding="utf-8")
    )
    checks = {
        "all_training_finite": all(value["finite"] for value in training_reports.values()),
        "all_loss_decreased": all(value["loss_decreased"] for value in training_reports.values()),
        "all_backbone_gradients": all(value["backbone_gradient"] for value in training_reports.values()),
        "all_head_gradients": all(value["head_gradient"] for value in training_reports.values()),
        "anchors_frozen": all(
            value["anchor_has_no_gradient"] and value["anchor_unchanged"]
            for value in training_reports.values()
        ),
        "same_anchor_initialization": len(anchor_hashes) == 1,
        "equal_trainable_parameters": len(set(parameter_counts.values())) == 1,
        "all_numeric_mechanics": all(
            value["deterministic"]
            and value["candidate_permutation"]
            and value["event_permutation"]
            and value["nohistory_exact"]
            for value in mechanics.values()
        ),
        "candidate_hash": fresh_data.candidate_hash(fresh_data.reserve_indices)
        == manifest["candidate_hash_reserve"],
        "upstream_checkpoint_hash": sha256_file(upstream_checkpoint)
        == upstream_report["checkpoint"]["sha256"],
        "fresh_labels_opened": False,
        "dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c80",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "five_mode_fit_and_label_free_fresh_scoring",
        "seed": args.seed,
        "execution_lock_sha256": lock_hash,
        "fit_requests": len(fit_data.fit_indices),
        "fresh_requests": len(fresh_data.reserve_indices),
        "training": training_reports,
        "mechanics": mechanics,
        "trainable_parameters": parameter_counts,
        "checkpoints": checkpoint_rows,
        "scores": {"path": str(score_path.relative_to(ROOT)), "sha256": sha256_file(score_path)},
        "candidate_rows": int(offsets[-1]),
        "checks": checks,
        "passed_mechanics": all(checks.values()),
    }
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "seed": args.seed,
                "passed_mechanics": report["passed_mechanics"],
                "loss": {
                    mode: [value["loss_first"], value["loss_last"]]
                    for mode, value in training_reports.items()
                },
                "mechanics": mechanics,
                "score_sha256": report["scores"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
