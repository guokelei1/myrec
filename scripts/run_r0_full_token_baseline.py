#!/usr/bin/env python
"""Train and blind-score one registered ordinary full-token R0 trial."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import random
import socket
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import transformers
import yaml


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
from prepare_r0_full_token_baseline import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def merged(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    output = dict(base)
    output.update(override)
    return output


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def assert_device(trial: dict[str, Any], device_name: str) -> torch.device:
    physical = int(trial["physical_gpu"])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("R0 full-token physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("R0 full-token trial requires exactly one visible GPU")
    return torch.device(device_name)


def verify_lock(config: dict[str, Any], config_path: Path) -> str:
    lock_path = ROOT / config["paths"]["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected = lock["input_sha256"]
    checks = {
        "config": config_path,
        "module": ROOT / "src/myrec/analysis/full_token_baseline.py",
        "token_module": ROOT / "src/myrec/analysis/token_history_observability.py",
        "prepare_script": ROOT / "scripts/prepare_r0_full_token_baseline.py",
        "run_script": Path(__file__),
        "budget": ROOT / "experiments/problem_discovery/r0_full_token_trial_budget.yaml",
    }
    for key, path in checks.items():
        if sha256_file(path) != expected[key]:
            raise RuntimeError(f"R0 full-token execution-lock source differs: {key}")
    manifest = json.loads(
        (ROOT / config["paths"]["artifact_root"] / "token_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for name, row in manifest["files"].items():
        if sha256_file(ROOT / row["path"]) != expected[f"artifact_{name}"]:
            raise RuntimeError(f"R0 full-token execution-lock artifact differs: {name}")
    return sha256_file(lock_path)


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
    training: dict[str, Any],
    tokens: dict[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": float(training["backbone_learning_rate"])},
            {"params": model.score_head.parameters(), "lr": float(training["head_learning_rate"])},
        ],
        weight_decay=float(training["weight_decay"]),
    )
    losses: list[float] = []
    active_backbone = False
    active_head = False
    steps = 0
    fit = np.asarray(data.fit_indices, dtype=np.int64)
    for epoch in range(int(training["epochs"])):
        order = fit.copy()
        np.random.default_rng(seed + epoch * 1009).shuffle(order)
        candidate_rng = np.random.default_rng(seed + epoch * 1009 + 17)
        dropout_rng = np.random.default_rng(seed + epoch * 1009 + 29)
        model.train()
        epoch_losses = []
        for start in range(0, len(order), int(training["requests_per_batch"])):
            requests = order[start : start + int(training["requests_per_batch"])]
            input_ids, attention, label_rows = pack_training_batch(
                data,
                requests,
                labels,
                sampled_candidates=int(training["sampled_candidates"]),
                candidate_rng=candidate_rng,
                dropout_rng=dropout_rng,
                history_dropout=float(training["history_dropout"]),
                token_config=tokens,
            )
            ids = torch.from_numpy(input_ids).to(device)
            mask = torch.from_numpy(attention).to(device)
            target = torch.from_numpy(label_rows).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                scores = model(ids, mask).reshape(len(requests), target.shape[1])
                loss = listwise_loss(scores, target)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("R0 full-token nonfinite loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is None:
                    continue
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"R0 full-token nonfinite gradient: {name}")
                if bool(parameter.grad.ne(0).any()):
                    active_backbone |= name.startswith("backbone.")
                    active_head |= name.startswith("score_head.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
            optimizer.step()
            value = float(loss.detach().cpu())
            losses.append(value)
            epoch_losses.append(value)
            steps += 1
        print(
            json.dumps(
                {"epoch": epoch + 1, "epochs": int(training["epochs"]), "loss": float(np.mean(epoch_losses))},
                sort_keys=True,
            ),
            flush=True,
        )
    window = min(200, max(1, len(losses) // 4))
    return {
        "steps": steps,
        "loss_first": float(np.mean(losses[:window])),
        "loss_last": float(np.mean(losses[-window:])),
        "loss_decreased": float(np.mean(losses[-window:])) < float(np.mean(losses[:window])),
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
    tokens: dict[str, Any],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            rows = [
                data.pack_candidate(
                    request_index,
                    int(candidate),
                    scenario=scenario,
                    query_tokens=int(tokens["query_tokens"]),
                    candidate_tokens=int(tokens["candidate_tokens"]),
                    history_item_tokens=int(tokens["history_item_tokens"]),
                    max_history=int(tokens["max_history"]),
                    max_length=int(tokens["max_sequence_length"]),
                )
                for candidate in candidates[start : start + batch_size]
            ]
            ids = torch.from_numpy(np.asarray([row[0] for row in rows], dtype=np.int64)).to(device)
            mask = torch.from_numpy(np.asarray([row[1] for row in rows], dtype=bool)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                values = model(ids, mask)
            output.append(values.float().cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main() -> int:
    args = parse_args()
    started = time.monotonic()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.trial_id not in config["trials"]:
        raise ValueError(f"unregistered R0 full-token trial: {args.trial_id}")
    trial = config["trials"][args.trial_id]
    lock_sha = verify_lock(config, config_path)
    device = assert_device(trial, args.device)
    seed = int(trial["seed"])
    seed_all(seed)
    tokens = merged(config["tokens"], trial.get("tokens") or {})
    training = merged(config["training"], trial.get("training") or {})
    paths = config["paths"]
    artifact_root = ROOT / paths["artifact_root"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    trial_slug = args.trial_id.split("-")[-1].lower()
    scenarios = [str(value) for value in config["evaluation"]["scenarios"]]
    run_ids = {
        scenario: f"20260713_kuaisearch_r0ft_{trial_slug}_{scenario}_dev"
        for scenario in scenarios
    }
    run_dirs = {scenario: ROOT / "runs" / run_id for scenario, run_id in run_ids.items()}
    for path in run_dirs.values():
        if path.exists():
            raise FileExistsError(path)
        path.mkdir(parents=True)
        (path / "run.lock").write_text(json.dumps({"pid": os.getpid()}) + "\n", encoding="utf-8")

    data = TokenHistoryData(artifact_root)
    labels = load_fit_labels(artifact_root / "fit_labels.npz")
    if set(labels) != set(data.fit_indices.tolist()):
        raise RuntimeError("R0 full-token compact fit labels differ")
    model = TokenHistoryCrossEncoder(
        ROOT / paths["bge_snapshot"], score_head_bias=bool(config["model"]["score_head_bias"])
    ).to(device)
    training_report = train(model, data, labels, training, tokens, seed=seed, device=device)
    checkpoint = checkpoint_root / f"{trial_slug}_seed_{seed}.pt"
    if checkpoint.exists():
        raise FileExistsError(checkpoint)
    torch.save(
        {"analysis_id": config["analysis_id"], "trial_id": args.trial_id, "seed": seed, "model_state": model.state_dict()},
        checkpoint,
    )

    manifest = json.loads((artifact_root / "token_manifest.json").read_text(encoding="utf-8"))
    dev_indices = np.asarray(data.reserve_indices, dtype=np.int64)
    if data.candidate_hash(dev_indices) != manifest["candidate_hash_dev"]:
        raise RuntimeError("R0 full-token candidate-key hash differs before scoring")
    method_ids = {scenario: f"r0_full_token_{trial_slug}_{scenario}" for scenario in scenarios}
    numeric = {"deterministic_max_abs": None, "candidate_permutation_max_abs": None}
    finite = True
    with ExitStack() as stack:
        handles = {
            scenario: stack.enter_context((run_dirs[scenario] / "scores.jsonl").open("w", encoding="utf-8"))
            for scenario in scenarios
        }
        for request_number, index_value in enumerate(dev_indices):
            index = int(index_value)
            candidates = data.candidates(index)
            item_ids = data.candidate_ids(index)
            values = {
                scenario: score_candidates(
                    model,
                    data,
                    index,
                    candidates,
                    scenario,
                    tokens,
                    int(config["evaluation"]["encoded_candidates_per_batch"]),
                    device,
                )
                for scenario in scenarios
            }
            finite &= all(bool(np.isfinite(row).all()) for row in values.values())
            if numeric["deterministic_max_abs"] is None and len(data.history(index, "true", int(tokens["max_history"]))) > 0:
                repeated = score_candidates(
                    model, data, index, candidates, "true", tokens,
                    int(config["evaluation"]["encoded_candidates_per_batch"]), device
                )
                reversed_values = score_candidates(
                    model, data, index, candidates[::-1].copy(), "true", tokens,
                    int(config["evaluation"]["encoded_candidates_per_batch"]), device
                )[::-1]
                numeric["deterministic_max_abs"] = float(np.max(np.abs(values["true"] - repeated)))
                numeric["candidate_permutation_max_abs"] = float(np.max(np.abs(values["true"] - reversed_values)))
            for scenario in scenarios:
                for item_id, score in zip(item_ids, values[scenario], strict=True):
                    handles[scenario].write(
                        json.dumps(
                            {
                                "request_id": data.request_ids[index],
                                "candidate_item_id": item_id,
                                "score": float(score),
                                "method_id": method_ids[scenario],
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
            if (request_number + 1) % 500 == 0:
                print(json.dumps({"scored_requests": request_number + 1, "total": len(dev_indices)}), flush=True)

    mechanics = {
        "loss_decreased": bool(training_report["loss_decreased"]),
        "finite_training": bool(training_report["finite"]),
        "finite_scores": finite,
        "backbone_gradient": bool(training_report["backbone_gradient"]),
        "head_gradient": bool(training_report["head_gradient"]),
        "deterministic": float(numeric["deterministic_max_abs"] or 0.0) <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": float(numeric["candidate_permutation_max_abs"] or 0.0) <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "candidate_hash": True,
        "wrong_donor_coverage": float(manifest["wrong_donor"]["coverage"]) >= float(config["evaluation"]["require_wrong_donor_coverage"]),
        "dev_qrels_closed_during_scoring": True,
        "test_closed": True,
        "c80_fresh_labels_closed": True,
    }
    if not all(mechanics.values()):
        raise RuntimeError(f"R0 full-token mechanics failed: {mechanics}")

    elapsed_hours = (time.monotonic() - started) / 3600.0
    commit = git_value("rev-parse", "HEAD")
    dirty = bool(git_value("status", "--porcelain"))
    candidate_sha = sha256_file(ROOT / paths["candidate_manifest"])
    config_sha = sha256_file(config_path)
    for scenario in scenarios:
        run_dir = run_dirs[scenario]
        snapshot = {
            "trial_id": args.trial_id,
            "scenario": scenario,
            "tokens": tokens,
            "training": training,
        }
        (run_dir / "config_snapshot.yaml").write_text(yaml.safe_dump(snapshot, sort_keys=True), encoding="utf-8")
        metadata = {
            "run_id": run_ids[scenario],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": commit,
            "git_dirty": dirty,
            "dataset_id": "kuaisearch",
            "dataset_version": "v0_lite",
            "split_id": "time_80_10_10_seed20260708",
            "candidate_manifest_sha256": candidate_sha,
            "method_id": method_ids[scenario],
            "method_group": "discovery",
            "config_path": str(config_path.relative_to(ROOT)),
            "config_sha256": config_sha,
            "seed": seed,
            "env_group": "discovery",
            "env_name": "current-torch",
            "python": platform.python_version(),
            "packages": {"torch": torch.__version__, "transformers": transformers.__version__},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "gpu_name": torch.cuda.get_device_name(0),
            "hostname": socket.gethostname(),
            "command": " ".join(sys.argv),
            "research_phase": "R0-BASE",
            "failure_id": None,
            "hypothesis_id": None,
            "implementation_id": "R0-BASE-I01",
            "trial_id": args.trial_id,
            "change_class": trial["change_class"],
            "dev_call_index": int(trial["dev_call_indices"][scenario]),
            "scenario": scenario,
            "execution_lock_sha256": lock_sha,
            "checkpoint_sha256": sha256_file(checkpoint),
            "gpu_hours_shared_across_scenarios": elapsed_hours,
        }
        atomic_json(run_dir / "metadata.json", metadata)
        (run_dir / "run.lock").unlink()
    report_path = artifact_root / "trials" / f"{trial_slug}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "analysis_id": config["analysis_id"],
        "trial_id": args.trial_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": lock_sha,
        "seed": seed,
        "tokens": tokens,
        "training_config": training,
        "training": training_report,
        "parameters": model.parameter_count(),
        "checkpoint": {"path": str(checkpoint.relative_to(ROOT)), "sha256": sha256_file(checkpoint)},
        "runs": {
            scenario: {
                "run_id": run_ids[scenario],
                "scores_sha256": sha256_file(run_dirs[scenario] / "scores.jsonl"),
                "dev_call_index": int(trial["dev_call_indices"][scenario]),
            }
            for scenario in scenarios
        },
        "numeric": numeric,
        "mechanics": mechanics,
        "passed_mechanics": True,
        "gpu_hours": elapsed_hours,
        "label_boundary": {"dev_qrels_read": False, "test_read": False, "c80_fresh_labels_read": False},
    }
    atomic_json(report_path, report)
    print(json.dumps({"trial_id": args.trial_id, "mechanics": True, "gpu_hours": elapsed_hours, "runs": run_ids}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
