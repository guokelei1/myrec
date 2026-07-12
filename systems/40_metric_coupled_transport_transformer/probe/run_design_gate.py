"""Freeze and run the data-free C40 structural/recovery gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.metric_coupled import (  # noqa: E402
    MODES,
    MULTIHEAD_COUPLED,
    SELECTION_ONLY,
    SHIFTED_LOOP,
    SINGLE_WIDE_COUPLED,
    MetricCoupledTransportTransformer,
)


LOCK_PATH = SYSTEM_ROOT / "notes" / "design_lock.json"
REPORT_PATH = REPO_ROOT / "reports" / "pps_c40_design_gate.json"
LOCKED_FILES = (
    "README.md",
    "environment.txt",
    "configs/design_gate.yaml",
    "model/__init__.py",
    "model/metric_coupled.py",
    "notes/proposal.md",
    "notes/mechanism_fingerprint.md",
    "notes/nearest_neighbors.md",
    "notes/design_gate_protocol.md",
    "probe/run_design_gate.py",
    "tests/test_model.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    with (SYSTEM_ROOT / "configs" / "design_gate.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C40 config must be an object")
    return value


def freeze() -> dict[str, Any]:
    if REPORT_PATH.exists():
        raise RuntimeError("C40 outcome already exists")
    files = {name: sha256_file(SYSTEM_ROOT / name) for name in LOCKED_FILES}
    payload = {
        "candidate_id": "c40",
        "gate_id": "c40_metric_coupled_design_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_files": files,
        "repository_data_authorized": False,
        "dev_test_authorized": False,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    write_json(LOCK_PATH, payload)
    return payload


def verify_lock() -> tuple[dict[str, Any], str]:
    with LOCK_PATH.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    actual = {name: sha256_file(SYSTEM_ROOT / name) for name in LOCKED_FILES}
    if actual != lock["locked_files"]:
        raise RuntimeError("C40 design lock differs")
    return lock, sha256_file(LOCK_PATH)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def make_model(
    config: Mapping[str, Any], seed: int, mode: str, *, teacher: bool = False
) -> MetricCoupledTransportTransformer:
    row = config["model"]
    init_std = (
        float(config["data"]["teacher_init_std"])
        if teacher
        else float(row["init_std"])
    )
    return MetricCoupledTransportTransformer(
        dim=int(row["dim"]),
        heads=int(row["heads"]),
        rank=int(row["rank"]),
        temperature=float(row["temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        seed=seed,
        mode=mode,
        init_std=init_std,
    )


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def generate(
    config: Mapping[str, Any], device: torch.device
) -> dict[str, torch.Tensor]:
    data = config["data"]
    count = int(data["train_requests"]) + int(data["test_requests"])
    dim = int(config["model"]["dim"])
    history_length = int(data["history_length"])
    candidate_count = int(data["candidates"])
    generator = torch.Generator().manual_seed(int(data["teacher_seed"]) + 17)
    query = F.normalize(torch.randn(count, dim, generator=generator), dim=-1)

    # Histories share a query-relevant component but retain independent events.
    history_noise = torch.randn(
        count, history_length, dim, generator=generator
    )
    history = F.normalize(
        0.35 * query[:, None, :] + history_noise, dim=-1
    )
    candidates = F.normalize(
        torch.randn(count, candidate_count, dim, generator=generator), dim=-1
    )
    base = 0.25 * torch.einsum("bd,bcd->bc", query, candidates)

    teacher = make_model(
        config,
        int(data["teacher_seed"]),
        MULTIHEAD_COUPLED,
        teacher=True,
    ).eval()
    teacher_scores = []
    with torch.inference_mode():
        for index in range(count):
            teacher_scores.append(
                base[index]
                + teacher(query[index], history[index], candidates[index])
            )
    teacher_score = torch.stack(teacher_scores)
    target = teacher_score.argmax(dim=-1)

    # A full deterministic derangement; wrong histories are never self-donors.
    wrong_permutation = torch.roll(torch.arange(count), shifts=1)
    return {
        "query": query.to(device),
        "history": history.to(device),
        "wrong_history": history[wrong_permutation].to(device),
        "candidates": candidates.to(device),
        "base": base.to(device),
        "target": target.to(device),
    }


def ndcg10(scores: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(scores, dim=-1, descending=True, stable=True)
    rank = (order == target[:, None]).nonzero(as_tuple=False)[:, 1]
    gain = 1.0 / torch.log2(rank.float() + 2.0)
    return torch.where(rank < 10, gain, torch.zeros_like(gain))


def score_dataset(
    model: MetricCoupledTransportTransformer,
    dataset: Mapping[str, torch.Tensor],
    indices: torch.Tensor,
    history_key: str,
) -> torch.Tensor:
    rows = []
    model.eval()
    with torch.inference_mode():
        for raw in indices:
            index = int(raw)
            rows.append(
                dataset["base"][index]
                + model(
                    dataset["query"][index],
                    dataset[history_key][index],
                    dataset["candidates"][index],
                )
            )
    return torch.stack(rows)


def train_mode(
    config: Mapping[str, Any],
    dataset: Mapping[str, torch.Tensor],
    seed: int,
    mode: str,
    device: torch.device,
) -> tuple[MetricCoupledTransportTransformer, dict[str, Any]]:
    seed_all(seed)
    model = make_model(config, seed, mode).to(device)
    initial = state_sha256(model)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    count = int(config["data"]["train_requests"])
    losses = []
    gradient_names: set[str] = set()
    model.train()
    for epoch in range(int(training["epochs"])):
        order = torch.randperm(
            count,
            generator=torch.Generator().manual_seed(seed + epoch * 1009),
        )
        batch = int(training["batch_size"])
        for start in range(0, count, batch):
            request_losses = []
            for raw in order[start : start + batch]:
                index = int(raw)
                score = dataset["base"][index] + model(
                    dataset["query"][index],
                    dataset["history"][index],
                    dataset["candidates"][index],
                )
                request_losses.append(
                    F.cross_entropy(
                        score.unsqueeze(0), dataset["target"][index].view(1)
                    )
                )
            loss = torch.stack(request_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite C40 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and bool(parameter.grad.ne(0).any()):
                    gradient_names.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return model, {
        "steps": len(losses),
        "finite": bool(np.isfinite(losses).all()),
        "loss_first_20": float(np.mean(losses[:20])),
        "loss_last_20": float(np.mean(losses[-20:])),
        "gradient_parameter_names": sorted(gradient_names),
        "initial_state_sha256": initial,
        "final_state_sha256": state_sha256(model),
    }


def structural_checks(
    config: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    seed = int(config["training"]["seeds"][0])
    seed_all(seed)
    generator = torch.Generator().manual_seed(seed + 71)
    dim = int(config["model"]["dim"])
    query = torch.randn(dim, generator=generator).to(device)
    history = torch.randn(7, dim, generator=generator).to(device)
    candidates = torch.randn(9, dim, generator=generator).to(device)
    permutation = torch.tensor([4, 0, 8, 1, 6, 2, 7, 3, 5], device=device)
    reports = {}
    initial_hashes = []
    parameter_counts = []
    corrections = {}
    for mode in MODES:
        model = make_model(config, seed, mode).to(device)
        parameter_counts.append(model.trainable_parameter_count())
        initial_hashes.append(state_sha256(model))
        first = model(query, history, candidates)
        second = model(query, history, candidates)
        permuted = model(query, history, candidates[permutation])
        loss = (first * torch.linspace(-1, 1, len(first), device=device)).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        state = model.components(query, history, candidates)
        reports[mode] = {
            "parameters": model.trainable_parameter_count(),
            "initial_state_sha256": initial_hashes[-1],
            "finite": bool(torch.isfinite(first).all()),
            "deterministic_max_abs": float((first - second).abs().max().cpu()),
            "permutation_max_abs": float(
                (first[permutation] - permuted).abs().max().cpu()
            ),
            "nohistory_max_abs": float(
                model(query, history[:0], candidates).abs().max().cpu()
            ),
            "query_absent_max_abs": float(
                model(query, history, candidates, query_present=False)
                .abs()
                .max()
                .cpu()
            ),
            "repeat_max_abs": float(
                model(query, history, candidates, repeat_present=True)
                .abs()
                .max()
                .cpu()
            ),
            "down_gradient_nonzero": bool(model.down.grad.ne(0).any()),
            "up_gradient_nonzero": bool(model.up.grad.ne(0).any()),
            "loop_assignment": state["loop_assignment"].cpu().tolist(),
        }
        corrections[mode] = first.detach()
    distinct = {
        mode: float(
            (corrections[MULTIHEAD_COUPLED] - corrections[mode]).abs().max().cpu()
        )
        for mode in MODES
        if mode != MULTIHEAD_COUPLED
    }
    limit = float(config["evaluation"]["permutation_error_max"])
    checks = {
        "cuda": device.type == "cuda",
        "equal_parameters": len(set(parameter_counts)) == 1,
        "paired_initialization": len(set(initial_hashes)) == 1,
        "finite": all(row["finite"] for row in reports.values()),
        "deterministic": all(
            row["deterministic_max_abs"] == 0.0 for row in reports.values()
        ),
        "candidate_permutation": all(
            row["permutation_max_abs"] <= limit for row in reports.values()
        ),
        "exact_fallbacks": all(
            row["nohistory_max_abs"] == 0.0
            and row["query_absent_max_abs"] == 0.0
            and row["repeat_max_abs"] == 0.0
            for row in reports.values()
        ),
        "both_factors_receive_gradient": all(
            row["down_gradient_nonzero"] and row["up_gradient_nonzero"]
            for row in reports.values()
        ),
        "primary_identity_loop": reports[MULTIHEAD_COUPLED]["loop_assignment"]
        == list(range(int(config["model"]["heads"]))),
        "shifted_fixed_point_free": all(
            index != value
            for index, value in enumerate(reports[SHIFTED_LOOP]["loop_assignment"])
        ),
        "reductions_functionally_distinct": all(value > 0 for value in distinct.values()),
    }
    return {"checks": checks, "mode_reports": reports, "primary_differences": distinct}


def run() -> dict[str, Any]:
    lock, lock_sha = verify_lock()
    if REPORT_PATH.exists():
        raise FileExistsError(REPORT_PATH)
    config = load_config()
    if not torch.cuda.is_available():
        raise RuntimeError("C40 design gate requires CUDA")
    device = torch.device(str(config["device"]))
    d0 = structural_checks(config, device)
    report: dict[str, Any] = {
        "candidate_id": "c40",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_lock_sha256": lock_sha,
        "design_lock_content_sha256": lock["content_sha256"],
        "D0": d0,
        "repository_data_read": False,
        "dev_test_read": False,
    }
    if not all(d0["checks"].values()):
        report["status"] = "failed_D0_terminal"
        write_json(REPORT_PATH, report)
        return report

    dataset = generate(config, device)
    start = int(config["data"]["train_requests"])
    stop = start + int(config["data"]["test_requests"])
    test_indices = torch.arange(start, stop)
    target = dataset["target"][test_indices]
    base_ndcg = ndcg10(dataset["base"][test_indices], target)
    seeds = [int(value) for value in config["training"]["seeds"]]
    seed_reports = {}
    all_checks = []
    for seed in seeds:
        mode_rows = {}
        for mode in MODES:
            model, training = train_mode(config, dataset, seed, mode, device)
            clean_scores = score_dataset(model, dataset, test_indices, "history")
            wrong_scores = score_dataset(model, dataset, test_indices, "wrong_history")
            clean = ndcg10(clean_scores, target)
            wrong = ndcg10(wrong_scores, target)
            correction = clean_scores - dataset["base"][test_indices]
            nohistory = max(
                float(
                    model(
                        dataset["query"][start],
                        dataset["history"][start, :0],
                        dataset["candidates"][start],
                    )
                    .abs()
                    .max()
                    .cpu()
                ),
                float(
                    model(
                        dataset["query"][start],
                        dataset["history"][start],
                        dataset["candidates"][start],
                        repeat_present=True,
                    )
                    .abs()
                    .max()
                    .cpu()
                ),
            )
            mode_rows[mode] = {
                "training": training,
                "clean_ndcg10": float(clean.mean().cpu()),
                "wrong_ndcg10": float(wrong.mean().cpu()),
                "clean_minus_wrong": float((clean - wrong).mean().cpu()),
                "correction_rms": float(correction.square().mean().sqrt().cpu()),
                "exact_fallback_max_abs": nohistory,
                "parameters": model.trainable_parameter_count(),
            }
        primary = mode_rows[MULTIHEAD_COUPLED]
        clean_gain = primary["clean_ndcg10"] - float(base_ndcg.mean().cpu())
        retention = (
            (primary["wrong_ndcg10"] - float(base_ndcg.mean().cpu())) / clean_gain
            if clean_gain > 0
            else math.inf
        )
        checks = {
            "primary_over_base": clean_gain
            >= float(config["evaluation"]["primary_minus_base_min"]),
            "primary_over_single_wide": primary["clean_ndcg10"]
            - mode_rows[SINGLE_WIDE_COUPLED]["clean_ndcg10"]
            >= float(config["evaluation"]["primary_minus_single_wide_min"]),
            "primary_over_shifted": primary["clean_ndcg10"]
            - mode_rows[SHIFTED_LOOP]["clean_ndcg10"]
            >= float(config["evaluation"]["primary_minus_shifted_min"]),
            "clean_over_wrong": primary["clean_minus_wrong"]
            >= float(config["evaluation"]["clean_minus_wrong_min"]),
            "wrong_gain_retention": retention
            <= float(config["evaluation"]["wrong_gain_retention_max"]),
            "exact_fallbacks": all(
                row["exact_fallback_max_abs"] == 0.0 for row in mode_rows.values()
            ),
            "all_modes_finite": all(
                row["training"]["finite"] for row in mode_rows.values()
            ),
            "all_modes_active": all(
                row["correction_rms"] > 1e-6 for row in mode_rows.values()
            ),
        }
        all_checks.extend(checks.values())
        seed_reports[str(seed)] = {
            "base_ndcg10": float(base_ndcg.mean().cpu()),
            "modes": mode_rows,
            "primary_clean_gain": clean_gain,
            "primary_wrong_gain_retention": retention,
            "checks": checks,
        }
    report["D1"] = {"seed_reports": seed_reports, "all_checks_pass": all(all_checks)}
    report["status"] = (
        "passed_design_gate" if all(all_checks) else "failed_D1_terminal"
    )
    write_json(REPORT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    value = freeze() if args.stage == "lock" else run()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
