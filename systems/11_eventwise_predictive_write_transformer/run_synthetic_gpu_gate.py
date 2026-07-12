#!/usr/bin/env python3
"""Locked C11 synthetic runner.  It refuses to run without approved review."""

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

from model import EventwisePredictiveWriteTransformer
from synthetic import (
    SyntheticBatch,
    SyntheticSpec,
    construct_audit,
    corrupt_history,
    generate_batch,
)


LOCK_PATH = HERE / "frozen_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def verify_approved_lock() -> dict:
    if not LOCK_PATH.exists():
        raise RuntimeError("C11 has no approved frozen_manifest.json; GPU execution is forbidden")
    manifest = json.loads(LOCK_PATH.read_text())
    if manifest.get("review_status") != "approved_for_single_gpu_run":
        raise RuntimeError("C11 independent architecture review is not approved")
    mismatch = {}
    for relative, expected in manifest["sha256"].items():
        observed = sha256(HERE / relative)
        if observed != expected:
            mismatch[relative] = {"expected": expected, "observed": observed}
    if mismatch:
        raise RuntimeError(f"C11 lock mismatch: {json.dumps(mismatch, sort_keys=True)}")
    return manifest


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def subset_metric(scores: Tensor, targets: Tensor, subset: Tensor) -> dict[str, float]:
    if not bool(subset.any()):
        raise RuntimeError("empty metric subset")
    selected = scores[subset]
    target = targets[subset]
    target_score = selected.gather(1, target[:, None])
    rank = 1 + selected.gt(target_score).sum(dim=1)
    ndcg = torch.log(torch.tensor(2.0, device=scores.device)) / torch.log(rank.float() + 1)
    return {
        "count": int(subset.sum()),
        "ndcg": float(ndcg.mean()),
        "mrr": float(rank.float().reciprocal().mean()),
        "top1": float(rank.eq(1).float().mean()),
    }


@torch.no_grad()
def evaluate(
    model: EventwisePredictiveWriteTransformer,
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
    transfer = ~batch.exact_repeat
    exact = batch.exact_repeat
    all_rows = torch.ones_like(exact)
    item_only = output.base_scores + output.identity_contribution
    delta = output.scores - output.base_scores
    metrics = {
        "all": subset_metric(output.scores, batch.targets, all_rows),
        "transfer": subset_metric(output.scores, batch.targets, transfer),
        "transfer_base": subset_metric(output.base_scores, batch.targets, transfer),
        "exact": subset_metric(output.scores, batch.targets, exact),
        "exact_item_only": subset_metric(item_only, batch.targets, exact),
        "delta_std": float(delta[transfer].std()),
        "order_change_fraction": float(
            output.scores[transfer]
            .argsort(dim=1)
            .ne(output.base_scores[transfer].argsort(dim=1))
            .any(dim=1)
            .float()
            .mean()
        ),
        "write_norm_max": float(output.hidden_write.norm(dim=-1).amax()),
        "write_sum_abs_max": float(output.hidden_write.sum(dim=1).abs().amax()),
        "event_gain_std": float(output.gain_matrix[transfer].std()),
    }

    cpu_batch = SyntheticBatch(
        **{name: value.detach().cpu() for name, value in batch.__dict__.items()}
    )
    generator = torch.Generator().manual_seed(corruption_seed)
    for kind in ("wrong_user", "shuffle_events", "query_mask"):
        history, evidence_query = corrupt_history(cpu_batch, kind, generator)
        history = history.to(batch.history_tokens.device)
        if evidence_query is not None:
            evidence_query = evidence_query.to(batch.query_tokens.device)
        corrupted = model(
            query_tokens=batch.query_tokens,
            candidate_tokens=batch.candidate_tokens,
            history_tokens=history,
            history_mask=batch.history_mask,
            evidence_query_tokens=evidence_query,
        )
        metrics[kind] = subset_metric(corrupted.scores, batch.targets, transfer)
    return metrics


def make_model(mode: str, spec: SyntheticSpec, config: dict, device: torch.device):
    model = config["model"]
    return EventwisePredictiveWriteTransformer(
        vocab_size=spec.vocab_size,
        candidate_token_count=spec.item_tokens,
        max_history_events=spec.history_events,
        d_model=model["d_model"],
        nhead=model["nhead"],
        lm_layers=model["lm_layers"],
        integrator_layers=model["integrator_layers"],
        dim_feedforward=model["dim_feedforward"],
        max_lm_sequence_length=model["max_lm_sequence_length"],
        dropout=model["dropout"],
        max_write_norm=model["max_write_norm"],
        initial_identity_weight=model["initial_identity_weight"],
        mode=mode,
    ).to(device)


def train_one(
    *,
    mode: str,
    initial_state: dict,
    spec: SyntheticSpec,
    config: dict,
    train: SyntheticBatch,
    evaluation: SyntheticBatch,
    device: torch.device,
    seed: int,
) -> dict:
    model = make_model(mode, spec, config, device)
    model.load_state_dict(initial_state)
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["learning_rate"],
        weight_decay=training["weight_decay"],
    )
    generator = torch.Generator().manual_seed(seed + 7000)
    losses = []
    examples = train.targets.shape[0]
    model.train()
    for _ in range(training["epochs"]):
        order = torch.randperm(examples, generator=generator)
        for start in range(0, examples, training["batch_size"]):
            index = order[start : start + training["batch_size"]].to(device)
            mini = SyntheticBatch(
                **{name: value[index] for name, value in train.__dict__.items()}
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(
                query_tokens=mini.query_tokens,
                candidate_tokens=mini.candidate_tokens,
                history_tokens=mini.history_tokens,
                history_mask=mini.history_mask,
            )
            rank_loss = F.cross_entropy(output.scores, mini.targets)
            # Only the common listwise ranking objective is optimized.  No
            # generator-role or "last event is reliable" supervision is exposed.
            loss = rank_loss
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"non-finite loss in {mode}, seed {seed}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), training["gradient_clip_norm"]
            )
            optimizer.step()
            losses.append(float(loss.detach()))
    return {
        "metrics": evaluate(model, evaluation, seed + 17000),
        "training": {
            "first_loss": losses[0],
            "last_loss": losses[-1],
            "minimum_loss": min(losses),
            "steps": len(losses),
            "trainable_parameters": model.trainable_parameter_count,
        },
    }


def construct_decision(audit: dict, config: dict) -> dict:
    threshold = config["construct_gate"]
    checks = {
        "target_position": audit["target_position_max_deviation"]
        <= threshold["maximum_target_position_deviation"],
        "variant_marginal": audit["variant_total_variation"]
        <= threshold["maximum_variant_total_variation"],
        "attribute_variant_marginal": audit["attribute_variant_total_variation"]
        <= threshold["maximum_attribute_variant_total_variation"],
        "variant_zero_rate": abs(
            audit["positive_variant_zero_rate"] - audit["negative_variant_zero_rate"]
        )
        <= threshold["maximum_variant_zero_rate_difference"],
        "exact_membership": bool(audit["exact_membership_ok"]),
        "hard_negative_count": bool(audit["hard_negative_count_ok"]),
    }
    return {"pass": all(checks.values()), "checks": checks, "values": audit}


def gate_decision(results: dict, config: dict) -> dict:
    gate = config["gate"]
    primary = gate["primary_mode"]
    seeds = [str(seed) for seed in config["training"]["seeds"]]
    checks = {}

    base_values = {
        seed: results[seed]["base"]["metrics"]["transfer"]["ndcg"] for seed in seeds
    }
    checks["base_ceiling"] = {
        "values": base_values,
        "maximum": gate["maximum_base_transfer_ndcg"],
        "pass": all(value <= gate["maximum_base_transfer_ndcg"] for value in base_values.values()),
    }
    gain = {
        seed: results[seed][primary]["metrics"]["transfer"]["ndcg"]
        - results[seed][primary]["metrics"]["transfer_base"]["ndcg"]
        for seed in seeds
    }
    checks["transfer_gain_each_seed"] = {
        "values": gain,
        "minimum": gate["minimum_transfer_gain_each_seed"],
        "pass": all(value >= gate["minimum_transfer_gain_each_seed"] for value in gain.values()),
    }
    for control in gate["exact_capacity_controls"] + [gate["nearest_neighbor_control"]]:
        advantage = {
            seed: results[seed][primary]["metrics"]["transfer"]["ndcg"]
            - results[seed][control]["metrics"]["transfer"]["ndcg"]
            for seed in seeds
        }
        mean = sum(advantage.values()) / len(advantage)
        checks[f"advantage_over_{control}"] = {
            "values": advantage,
            "mean": mean,
            "pass": mean >= gate["minimum_mean_advantage_over_each_control"]
            and sum(value >= 0 for value in advantage.values())
            >= gate["minimum_nonnegative_control_wins"],
        }

    repeat_delta = {
        seed: results[seed][primary]["metrics"]["exact"]["ndcg"]
        - results[seed][primary]["metrics"]["exact_item_only"]["ndcg"]
        for seed in seeds
    }
    checks["repeat_noninferiority"] = {
        "values": repeat_delta,
        "pass": all(
            value >= -gate["repeat_ndcg_noninferiority_margin"]
            for value in repeat_delta.values()
        ),
    }
    clean_mean = sum(gain.values()) / len(gain)
    for corruption, key in (
        ("wrong_user", "maximum_wrong_user_retention"),
        ("shuffle_events", "maximum_shuffle_retention"),
        ("query_mask", "maximum_query_mask_retention"),
    ):
        corrupted_gain = [
            results[seed][primary]["metrics"][corruption]["ndcg"]
            - results[seed][primary]["metrics"]["transfer_base"]["ndcg"]
            for seed in seeds
        ]
        retention = (
            sum(corrupted_gain) / len(corrupted_gain) / clean_mean
            if clean_mean > 0
            else math.inf
        )
        checks[f"{corruption}_retention"] = {
            "values": corrupted_gain,
            "retention": retention,
            "pass": retention <= gate[key],
        }
    checks["conditional_noncollapse"] = {
        "delta_std": [results[seed][primary]["metrics"]["delta_std"] for seed in seeds],
        "order_change": [
            results[seed][primary]["metrics"]["order_change_fraction"] for seed in seeds
        ],
        "pass": all(
            results[seed][primary]["metrics"]["delta_std"] >= gate["minimum_delta_std"]
            and results[seed][primary]["metrics"]["order_change_fraction"]
            >= gate["minimum_order_change_fraction"]
            for seed in seeds
        ),
    }
    checks["zero_sum_and_bound"] = {
        "pass": all(
            results[seed][primary]["metrics"]["write_sum_abs_max"] <= 2e-5
            and results[seed][primary]["metrics"]["write_norm_max"]
            < config["model"]["max_write_norm"] + 1e-6
            for seed in seeds
        )
    }
    return {"pass": all(value["pass"] for value in checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=HERE / "configs/synthetic_gpu_gate.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = verify_approved_lock()
    config = yaml.safe_load(args.config.read_text())
    if config["data"]["kind"] != "synthetic_only":
        raise RuntimeError("C11 runner accepts synthetic data only")
    if any(word in str(args.config).lower() for word in ("qrels", "dev", "test")):
        raise RuntimeError("dev/test/qrels paths are forbidden")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(config["device"]["physical_gpu"]):
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not match the approved physical GPU")
    if config["device"]["require_cuda"] and not torch.cuda.is_available():
        raise RuntimeError("approved gate requires CUDA")
    device = torch.device("cuda:0")
    torch.use_deterministic_algorithms(True)

    data = config["data"]
    spec = SyntheticSpec(
        categories=data["categories"],
        attributes=data["attributes"],
        item_variants=data["item_variants"],
        history_events=data["history_events"],
        candidates=data["candidates"],
        exact_repeat_probability=data["exact_repeat_probability"],
    )
    audit_batch = generate_batch(
        spec,
        data["construct_audit_examples"],
        torch.Generator().manual_seed(data["construct_audit_seed"]),
    )
    audit = construct_decision(construct_audit(audit_batch, spec), config)
    if not audit["pass"]:
        raise RuntimeError(f"pre-training construct audit failed: {json.dumps(audit, sort_keys=True)}")

    started = time.time()
    results = {}
    for seed in config["training"]["seeds"]:
        seed_all(seed)
        train = generate_batch(
            spec, data["train_examples"], torch.Generator().manual_seed(seed + 1)
        ).to(device)
        evaluation = generate_batch(
            spec, data["evaluation_examples"], torch.Generator().manual_seed(seed + 2)
        ).to(device)
        template = make_model("eventwise_predictive", spec, config, torch.device("cpu"))
        initial_state = copy.deepcopy(template.state_dict())
        per_mode = {}
        for mode in config["training"]["modes"]:
            seed_all(seed)
            per_mode[mode] = train_one(
                mode=mode,
                initial_state=initial_state,
                spec=spec,
                config=config,
                train=train,
                evaluation=evaluation,
                device=device,
                seed=seed,
            )
            if time.time() - started > config["device"]["maximum_elapsed_seconds"]:
                raise RuntimeError("C11 exceeded its frozen single-A40 wall-time budget")
        results[str(seed)] = per_mode

    nohistory = generate_batch(
        spec, data["no_history_examples"], torch.Generator().manual_seed(991101)
    ).to(device)
    nohistory.history_mask.zero_()
    nohistory_model = make_model("eventwise_predictive", spec, config, device).eval()
    with torch.no_grad():
        nohistory_output = nohistory_model(
            query_tokens=nohistory.query_tokens,
            candidate_tokens=nohistory.candidate_tokens,
            history_tokens=nohistory.history_tokens,
            history_mask=nohistory.history_mask,
        )
    nohistory_identity = torch.equal(
        nohistory_output.scores, nohistory_output.base_scores
    )
    decision = gate_decision(results, config)
    decision["checks"]["no_history_bitwise_identity"] = {
        "pass": nohistory_identity,
        "examples": data["no_history_examples"],
    }
    decision["pass"] = decision["pass"] and nohistory_identity
    report = {
        "protocol_id": config["protocol_id"],
        "status": "PASS" if decision["pass"] else "FAIL",
        "scope": "synthetic eventwise mechanism falsifier only",
        "manifest_sha256": sha256(LOCK_PATH),
        "construct_audit": audit,
        "decision": decision,
        "results": results,
        "device": {
            "physical_gpu": config["device"]["physical_gpu"],
            "name": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
        },
        "elapsed_seconds": time.time() - started,
        "forbidden_data_access": {
            "real_records": 0,
            "dev": 0,
            "test": 0,
            "qrels": 0,
        },
        "lock": manifest,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
