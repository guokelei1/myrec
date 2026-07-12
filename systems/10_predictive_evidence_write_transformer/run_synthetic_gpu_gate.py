#!/usr/bin/env python3
"""Execute the locked C10 synthetic learned gate on one physical GPU."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from model import PredictiveEvidenceWriteTransformer
from synthetic import SyntheticBatch, SyntheticSpec, corrupt_history, generate_batch


LOCK_PATH = HERE / "frozen_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_lock() -> dict:
    manifest = json.loads(LOCK_PATH.read_text())
    failures = {}
    for relative, expected in manifest["sha256"].items():
        actual = sha256(HERE / relative)
        if actual != expected:
            failures[relative] = {"expected": expected, "actual": actual}
    if failures:
        raise RuntimeError(f"C10 pre-outcome lock mismatch: {json.dumps(failures, sort_keys=True)}")
    return manifest


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset_metric(scores: Tensor, targets: Tensor, subset: Tensor) -> dict[str, float]:
    if not bool(subset.any()):
        raise RuntimeError("empty evaluation subset")
    chosen = scores[subset]
    target = targets[subset]
    target_score = chosen.gather(1, target[:, None])
    rank = 1 + (chosen > target_score).sum(dim=1)
    reciprocal_rank = rank.float().reciprocal()
    ndcg = torch.log(torch.tensor(2.0, device=scores.device)) / torch.log(rank.float() + 1.0)
    return {
        "count": int(subset.sum()),
        "ndcg": float(ndcg.mean()),
        "mrr": float(reciprocal_rank.mean()),
        "top1": float(rank.eq(1).float().mean()),
    }


@torch.no_grad()
def evaluate(
    model: PredictiveEvidenceWriteTransformer,
    batch: SyntheticBatch,
    corruption_seed: int,
) -> dict:
    model.eval()
    output = model(
        query_tokens=batch.query_tokens,
        candidate_tokens=batch.candidate_tokens,
        history_tokens=batch.history_tokens,
        history_mask=batch.history_mask,
    )
    all_rows = torch.ones_like(batch.exact_repeat)
    transfer = ~batch.exact_repeat
    exact = batch.exact_repeat
    item_only_scores = output.base_scores + output.identity_contribution
    score_delta = output.scores - output.base_scores
    result = {
        "all": subset_metric(output.scores, batch.targets, all_rows),
        "transfer": subset_metric(output.scores, batch.targets, transfer),
        "transfer_base": subset_metric(output.base_scores, batch.targets, transfer),
        "exact": subset_metric(output.scores, batch.targets, exact),
        "exact_item_only": subset_metric(item_only_scores, batch.targets, exact),
        "delta_std": float(score_delta[transfer].std()),
        "delta_abs_mean": float(score_delta[transfer].abs().mean()),
        "order_change_fraction": float(
            output.scores[transfer].argsort(dim=1).ne(output.base_scores[transfer].argsort(dim=1)).any(dim=1).float().mean()
        ),
        "write_norm_max": float(output.hidden_write.norm(dim=-1).amax()),
        "write_sum_abs_max": float(output.hidden_write.sum(dim=1).abs().amax()),
    }

    generator = torch.Generator().manual_seed(corruption_seed)
    for kind in ("wrong_user", "shuffle_events", "query_mask"):
        corrupt_cpu = SyntheticBatch(**{name: value.detach().cpu() for name, value in batch.__dict__.items()})
        history, evidence_query = corrupt_history(corrupt_cpu, kind, generator)
        history = history.to(batch.history_tokens.device)
        if evidence_query is not None:
            evidence_query = evidence_query.to(batch.query_tokens.device)
        corrupt_output = model(
            query_tokens=batch.query_tokens,
            candidate_tokens=batch.candidate_tokens,
            history_tokens=history,
            history_mask=batch.history_mask,
            evidence_query_tokens=evidence_query,
        )
        result[kind] = subset_metric(corrupt_output.scores, batch.targets, transfer)
    return result


def train_one(
    mode: str,
    template_state: dict,
    spec: SyntheticSpec,
    config: dict,
    train_data: SyntheticBatch,
    eval_data: SyntheticBatch,
    device: torch.device,
    seed: int,
) -> tuple[dict, dict]:
    model_config = config["model"]
    model = PredictiveEvidenceWriteTransformer(
        vocab_size=spec.vocab_size,
        candidate_token_count=spec.item_tokens,
        d_model=model_config["d_model"],
        nhead=model_config["nhead"],
        num_layers=model_config["num_layers"],
        dim_feedforward=model_config["dim_feedforward"],
        max_sequence_length=model_config["max_sequence_length"],
        dropout=model_config["dropout"],
        max_write_norm=model_config["max_write_norm"],
        initial_repeat_bonus=model_config["initial_repeat_bonus"],
        mode=mode,
    ).to(device)
    model.load_state_dict(template_state)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    generator = torch.Generator().manual_seed(seed + 9000)
    losses = []
    model.train()
    examples = train_data.targets.shape[0]
    for epoch in range(training["epochs"]):
        order = torch.randperm(examples, generator=generator)
        for start in range(0, examples, training["batch_size"]):
            index = order[start : start + training["batch_size"]].to(device)
            mini = SyntheticBatch(**{name: value[index] for name, value in train_data.__dict__.items()})
            optimizer.zero_grad(set_to_none=True)
            output = model(
                query_tokens=mini.query_tokens,
                candidate_tokens=mini.candidate_tokens,
                history_tokens=mini.history_tokens,
                history_mask=mini.history_mask,
            )
            rank_loss = F.cross_entropy(output.scores, mini.targets)
            row = torch.arange(mini.targets.shape[0], device=device)
            positive_log_probs = output.log_probs_history[row, mini.targets]
            auxiliary = -positive_log_probs.mean()
            loss = rank_loss + training["auxiliary_token_nll_weight"] * auxiliary
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss for {mode}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip_norm"])
            optimizer.step()
            losses.append(float(loss.detach()))
    metrics = evaluate(model, eval_data, seed + 18000)
    training_record = {
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "minimum_loss": min(losses),
        "steps": len(losses),
        "trainable_parameters": model.trainable_parameter_count,
    }
    return metrics, training_record


def decide_gate(results: dict, config: dict) -> dict:
    gate = config["gate"]
    primary = gate["primary_mode"]
    seeds = [str(seed) for seed in config["training"]["seeds"]]
    checks: dict[str, dict] = {}

    primary_gain = {
        seed: results[seed][primary]["metrics"]["transfer"]["ndcg"]
        - results[seed][primary]["metrics"]["transfer_base"]["ndcg"]
        for seed in seeds
    }
    checks["transfer_gain_each_seed"] = {
        "values": primary_gain,
        "threshold": gate["minimum_transfer_gain_each_seed"],
        "pass": all(value >= gate["minimum_transfer_gain_each_seed"] for value in primary_gain.values()),
    }

    for control in gate["exact_capacity_controls"] + [gate["nearest_neighbor_control"]]:
        advantages = {
            seed: results[seed][primary]["metrics"]["transfer"]["ndcg"]
            - results[seed][control]["metrics"]["transfer"]["ndcg"]
            for seed in seeds
        }
        checks[f"advantage_over_{control}"] = {
            "values": advantages,
            "mean": sum(advantages.values()) / len(advantages),
            "minimum_mean": gate["minimum_mean_advantage_over_each_control"],
            "nonnegative_required": gate["minimum_nonnegative_control_wins"],
            "pass": (
                sum(advantages.values()) / len(advantages)
                >= gate["minimum_mean_advantage_over_each_control"]
                and sum(value >= 0 for value in advantages.values())
                >= gate["minimum_nonnegative_control_wins"]
            ),
        }

    repeat_delta = {
        seed: results[seed][primary]["metrics"]["exact"]["ndcg"]
        - results[seed][primary]["metrics"]["exact_item_only"]["ndcg"]
        for seed in seeds
    }
    checks["repeat_noninferiority"] = {
        "values": repeat_delta,
        "margin": gate["repeat_ndcg_noninferiority_margin"],
        "pass": all(value >= -gate["repeat_ndcg_noninferiority_margin"] for value in repeat_delta.values()),
    }

    clean_gain_mean = sum(primary_gain.values()) / len(primary_gain)
    for corruption, limit_key in (
        ("wrong_user", "maximum_wrong_user_retention"),
        ("shuffle_events", "maximum_shuffle_retention"),
        ("query_mask", "maximum_query_mask_retention"),
    ):
        corrupted_gain = []
        for seed in seeds:
            metric = results[seed][primary]["metrics"]
            corrupted_gain.append(metric[corruption]["ndcg"] - metric["transfer_base"]["ndcg"])
        mean_gain = sum(corrupted_gain) / len(corrupted_gain)
        retention = mean_gain / clean_gain_mean if clean_gain_mean > 0 else math.inf
        checks[f"{corruption}_retention"] = {
            "gains": corrupted_gain,
            "retention": retention,
            "maximum": gate[limit_key],
            "pass": retention <= gate[limit_key],
        }

    delta_std = [results[seed][primary]["metrics"]["delta_std"] for seed in seeds]
    order_change = [results[seed][primary]["metrics"]["order_change_fraction"] for seed in seeds]
    checks["conditional_noncollapse"] = {
        "delta_std": delta_std,
        "order_change_fraction": order_change,
        "pass": all(value >= gate["minimum_delta_std"] for value in delta_std)
        and all(value >= gate["minimum_order_change_fraction"] for value in order_change),
    }
    checks["zero_sum_and_bound"] = {
        "pass": all(
            results[seed][primary]["metrics"]["write_sum_abs_max"] <= 2e-5
            and results[seed][primary]["metrics"]["write_norm_max"] < config["model"]["max_write_norm"] + 1e-6
            for seed in seeds
        )
    }
    return {"pass": all(check["pass"] for check in checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "configs/synthetic_gpu_gate.yaml")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_lock()
    config = yaml.safe_load(args.config.read_text())
    if config["data"]["kind"] != "synthetic_only":
        raise RuntimeError("C10 locked runner accepts synthetic data only")
    if any(word in str(args.config).lower() for word in ("qrels", "dev", "test")):
        raise RuntimeError("label-bearing/dev/test paths are forbidden")
    if config["device"]["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen gate")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)

    data_config = config["data"]
    spec = SyntheticSpec(
        categories=data_config["categories"],
        attributes=data_config["attributes"],
        item_variants=data_config["item_variants"],
        history_events=data_config["history_events"],
        candidates=data_config["candidates"],
        exact_repeat_probability=data_config["exact_repeat_probability"],
    )
    started = time.time()
    results = {}
    for seed in config["training"]["seeds"]:
        seed_everything(seed)
        train_cpu = generate_batch(spec, data_config["train_examples"], torch.Generator().manual_seed(seed + 1))
        eval_cpu = generate_batch(spec, data_config["evaluation_examples"], torch.Generator().manual_seed(seed + 2))
        train_data = train_cpu.to(device)
        eval_data = eval_cpu.to(device)
        template = PredictiveEvidenceWriteTransformer(
            vocab_size=spec.vocab_size,
            candidate_token_count=spec.item_tokens,
            d_model=config["model"]["d_model"],
            nhead=config["model"]["nhead"],
            num_layers=config["model"]["num_layers"],
            dim_feedforward=config["model"]["dim_feedforward"],
            max_sequence_length=config["model"]["max_sequence_length"],
            dropout=config["model"]["dropout"],
            max_write_norm=config["model"]["max_write_norm"],
            initial_repeat_bonus=config["model"]["initial_repeat_bonus"],
            mode="predictive_gain",
        )
        template_state = copy.deepcopy(template.state_dict())
        seed_results = {}
        for mode in config["training"]["modes"]:
            seed_everything(seed)
            metrics, training_record = train_one(
                mode, template_state, spec, config, train_data, eval_data, device, seed
            )
            seed_results[mode] = {"metrics": metrics, "training": training_record}
        results[str(seed)] = seed_results

    # Independent no-history executable contract on the final initialization.
    nohistory_cpu = generate_batch(
        spec, data_config["no_history_examples"], torch.Generator().manual_seed(991001)
    )
    nohistory = nohistory_cpu.to(device)
    nohistory.history_mask.zero_()
    nohistory_model = PredictiveEvidenceWriteTransformer(
        vocab_size=spec.vocab_size,
        candidate_token_count=spec.item_tokens,
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_layers=config["model"]["num_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        max_sequence_length=config["model"]["max_sequence_length"],
        dropout=0.0,
        mode="predictive_gain",
    ).to(device).eval()
    with torch.no_grad():
        nohistory_out = nohistory_model(
            query_tokens=nohistory.query_tokens,
            candidate_tokens=nohistory.candidate_tokens,
            history_tokens=nohistory.history_tokens,
            history_mask=nohistory.history_mask,
        )
    nohistory_identity = torch.equal(nohistory_out.scores, nohistory_out.base_scores)
    decision = decide_gate(results, config)
    decision["checks"]["no_history_bitwise_identity"] = {
        "pass": nohistory_identity,
        "examples": data_config["no_history_examples"],
    }
    decision["pass"] = decision["pass"] and nohistory_identity

    report = {
        "protocol_id": config["protocol_id"],
        "run_prefix": config["run_prefix"],
        "status": "PASS" if decision["pass"] else "FAIL",
        "scientific_scope": "synthetic learned-mechanism falsifier only",
        "manifest_sha256": sha256(LOCK_PATH),
        "locked_files": manifest["sha256"],
        "device": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "decision": decision,
        "results": results,
        "forbidden_data_access": {"dev": 0, "test": 0, "qrels": 0, "real_records": 0},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output), "elapsed_seconds": report["elapsed_seconds"]}))


if __name__ == "__main__":
    main()
