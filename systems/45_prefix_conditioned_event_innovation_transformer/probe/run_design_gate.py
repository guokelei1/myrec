"""Run and aggregate the locked C45 data-free GPU gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model import MODES, PrefixConditionedEventInnovationTransformer  # noqa: E402
from probe.synthetic import (  # noqa: E402
    SyntheticRows,
    complete_orders,
    generate,
    ndcg10,
    shuffled_history,
    wrong_history,
)


PRIMARY = "innovation"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value.get("candidate_id") != "c45":
        raise ValueError("C45 config identity differs")
    return value


def verify_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_any_c45_trained_model_outcome":
        raise PermissionError("C45 proposal is not locked")
    declarations = value["declarations"]
    if declarations != {
        "dev_test_qrels_read": False,
        "repository_data_read": False,
        "repository_labels_read": False,
        "shared_evaluator_calls": 0,
        "trained_c45_model_outcome_observed": False,
    }:
        raise PermissionError("C45 pre-outcome declarations differ")
    for relative, expected in value["files"].items():
        if sha256_file(SYSTEM_ROOT / relative) != expected:
            raise RuntimeError(f"locked C45 file changed: {relative}")
    return value, sha256_file(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def make_model(config: Mapping[str, Any], mode: str) -> PrefixConditionedEventInnovationTransformer:
    generator = config["generator"]
    row = config["model"]
    return PrefixConditionedEventInnovationTransformer(
        input_dim=int(generator["input_dim"]),
        width=int(row["width"]),
        heads=int(row["heads"]),
        ff_multiplier=int(row["ff_multiplier"]),
        max_history=int(row["max_history"]),
        mode=mode,
    )


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    positive = (scores * labels).sum(dim=1)
    return (torch.logsumexp(scores, dim=1) - positive).mean()


def parameter_groups(model: torch.nn.Module, names: set[str]) -> dict[str, bool]:
    prefixes = {
        "input_projections": ("query_projection", "candidate_projection", "event_projection"),
        "transition": ("transition", "initial_state", "state_role", "event_role", "position"),
        "base_transformer": ("base_transformer", "base_read", "base_head"),
        "evidence_transformer": ("evidence_transformer", "evidence_norm", "correction_head"),
    }
    return {
        group: any(any(name == prefix or name.startswith(prefix + ".") for prefix in values) for name in names)
        for group, values in prefixes.items()
    }


def train_model(
    model: PrefixConditionedEventInnovationTransformer,
    rows: SyntheticRows,
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    training = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    count = rows.query.shape[0]
    rng = np.random.default_rng(seed + 991)
    permutations = [rng.permutation(count) for _ in range((int(training["steps"]) * int(training["batch_size"]) // count) + 2)]
    stream = np.concatenate(permutations)
    cursor = 0
    losses: list[float] = []
    gradient_names: set[str] = set()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.train()
    for _ in range(int(training["steps"])):
        batch_size = int(training["batch_size"])
        raw = stream[cursor : cursor + batch_size]
        cursor += batch_size
        indices = torch.from_numpy(raw.astype(np.int64)).to(rows.query.device)
        batch = rows.subset(indices)
        output = model.forward_components(
            batch.query, batch.candidates, batch.history, batch.history_mask
        )
        loss = listwise_loss(output.score, batch.labels) + float(
            training["base_loss_weight"]
        ) * listwise_loss(output.base, batch.labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite C45 loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"nonfinite C45 gradient: {name}")
                if bool(parameter.grad.ne(0).any()):
                    gradient_names.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    updated = {
        name for name, value in model.named_parameters() if not torch.equal(value.detach(), initial[name])
    }
    groups = parameter_groups(model, gradient_names)
    return {
        "steps": len(losses),
        "finite": bool(np.isfinite(losses).all()),
        "loss_first_30": float(np.mean(losses[:30])),
        "loss_last_30": float(np.mean(losses[-30:])),
        "gradient_parameter_names": sorted(gradient_names),
        "gradient_groups": groups,
        "all_load_bearing_gradient_groups_active": all(groups.values()),
        "updated_parameter_names": sorted(updated),
        "parameters_updated": bool(updated),
    }


def mean_ndcg(scores: torch.Tensor, labels: torch.Tensor) -> float:
    return float(ndcg10(scores, labels).mean().cpu())


def evaluate(
    model: PrefixConditionedEventInnovationTransformer,
    rows: SyntheticRows,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        clean = model.forward_components(rows.query, rows.candidates, rows.history, rows.history_mask)
        clean_again = model.forward_components(rows.query, rows.candidates, rows.history, rows.history_mask)
        wrong = model.forward_components(
            rows.query, rows.candidates, wrong_history(rows), rows.history_mask
        )
        shuffled = model.forward_components(
            rows.query, rows.candidates, shuffled_history(rows), rows.history_mask
        )
        no_mask = torch.zeros_like(rows.history_mask)
        no_history = model.forward_components(rows.query, rows.candidates, rows.history, no_mask)
        query_absent = model.forward_components(
            torch.zeros_like(rows.query),
            rows.candidates,
            rows.history,
            rows.history_mask,
            query_present=torch.zeros(len(rows.query), dtype=torch.bool, device=rows.query.device),
        )
        permutation = torch.arange(rows.candidates.shape[1] - 1, -1, -1, device=rows.query.device)
        permuted = model.forward_components(
            rows.query, rows.candidates[:, permutation], rows.history, rows.history_mask
        )
        restored = permuted.score[:, permutation]
        item_only = torch.linspace(
            -1.0, 1.0, rows.candidates.shape[1], device=rows.query.device
        )[None].expand(len(rows.query), -1)
        repeat = model.rank(
            rows.query,
            rows.candidates,
            rows.history,
            rows.history_mask,
            repeat_present=torch.ones(len(rows.query), dtype=torch.bool, device=rows.query.device),
            item_only_scores=item_only,
        )
        state = model.initial_state[None].expand(min(32, len(rows.query)), -1)
        null = model.null_event[None].expand_as(state)
        first = model._transition_once(state, null, 0)
        second = model._transition_once(state, null, 0)

    base_ndcg = mean_ndcg(clean.base, rows.labels)
    clean_ndcg = mean_ndcg(clean.score, rows.labels)
    wrong_ndcg = mean_ndcg(wrong.score, rows.labels)
    shuffled_ndcg = mean_ndcg(shuffled.score, rows.labels)
    gain = clean_ndcg - base_ndcg
    positive = rows.labels.bool()
    clicked_direction = float(
        (
            clean.correction[positive]
            .reshape(len(rows.query), 1)
            .squeeze(1)
            - (clean.correction.masked_fill(positive, 0.0).sum(1) / (~positive).sum(1))
        ).mean().cpu()
    )
    activity = float(
        (complete_orders(clean.score) != complete_orders(clean.base)).any(dim=1).float().mean().cpu()
    )
    return {
        "ndcg10": {
            "base": base_ndcg,
            "clean": clean_ndcg,
            "wrong": wrong_ndcg,
            "shuffled": shuffled_ndcg,
        },
        "gain_over_base": gain,
        "wrong_gain_retention": (wrong_ndcg - base_ndcg) / gain if gain > 0 else None,
        "shuffle_gain_retention": (shuffled_ndcg - base_ndcg) / gain if gain > 0 else None,
        "clicked_minus_unclicked_correction": clicked_direction,
        "order_change_fraction": activity,
        "deterministic_max_abs": float((clean.score - clean_again.score).abs().max().cpu()),
        "candidate_permutation_max_abs": float((clean.score - restored).abs().max().cpu()),
        "nohistory_base_max_abs": float((no_history.score - no_history.base).abs().max().cpu()),
        "query_absent_correction_max_abs": float(query_absent.correction.abs().max().cpu()),
        "repeat_item_only_max_abs": float((repeat - item_only).abs().max().cpu()),
        "null_event_innovation_max_abs": float((first - second).abs().max().cpu()),
        "states_finite": bool(
            clean.event_tokens.isfinite().all()
            and clean.factual_states.isfinite().all()
            and clean.null_states.isfinite().all()
        ),
    }


def run_seed(config: Mapping[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    _, lock_hash = verify_lock(config)
    registered = int(config["resources"]["seed_to_physical_gpu"].get(str(seed), -1))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(registered):
        raise RuntimeError("C45 physical GPU registration differs")
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C45 seed requires exactly one visible cuda:0")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C45 deterministic CUBLAS setting absent")
    seed_all(seed)
    train_rows = generate(
        config,
        requests=int(config["generator"]["train_requests"]),
        split_offset=100_000,
    ).to(device)
    validation = generate(
        config,
        requests=int(config["generator"]["validation_requests"]),
        split_offset=200_000,
    ).to(device)
    artifact_root = REPO_ROOT / config["paths"]["artifact_root"]
    output_path = artifact_root / f"seed_{seed}_report.json"
    if output_path.exists():
        raise FileExistsError(output_path)
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    mode_reports: dict[str, Any] = {}
    initial_hashes: dict[str, str] = {}
    for mode in MODES:
        seed_all(seed)
        model = make_model(config, mode).to(device)
        initial_hashes[mode] = state_sha256(model)
        parameters = sum(value.numel() for value in model.parameters())
        training = train_model(model, train_rows, config, seed)
        evaluation = evaluate(model, validation, config)
        checkpoint_path = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        torch.save(
            {
                "candidate_id": "c45",
                "seed": seed,
                "mode": mode,
                "proposal_lock_sha256": lock_hash,
                "state_dict": model.state_dict(),
            },
            checkpoint_path,
        )
        mode_reports[mode] = {
            "parameters": parameters,
            "training": training,
            "evaluation": evaluation,
            "final_state_sha256": state_sha256(model),
            "checkpoint": {
                "path": str(checkpoint_path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(checkpoint_path),
            },
        }
        del model
        torch.cuda.empty_cache()
    report = {
        "candidate_id": "c45",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "physical_gpu": registered,
        "proposal_lock_sha256": lock_hash,
        "paired_initialization": len(set(initial_hashes.values())) == 1,
        "initial_state_sha256": initial_hashes,
        "mode_reports": mode_reports,
        "repository_data_read": False,
        "repository_labels_read": False,
        "dev_test_qrels_read": False,
        "shared_evaluator_calls": 0,
    }
    atomic_json(output_path, report)
    return report


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_lock(config)
    artifact_root = REPO_ROOT / config["paths"]["artifact_root"]
    output_path = artifact_root / "design_gate_report.json"
    promoted_path = REPO_ROOT / config["paths"]["promoted_report"]
    if output_path.exists() or promoted_path.exists():
        raise FileExistsError(output_path if output_path.exists() else promoted_path)
    seeds = [int(value) for value in config["training"]["seeds"]]
    reports = [json.loads((artifact_root / f"seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in seeds]
    params = {
        report["mode_reports"][mode]["parameters"]
        for report in reports
        for mode in MODES
    }
    tolerance = config["evaluation"]
    d0_checks = {
        "paired_initialization": all(report["paired_initialization"] for report in reports),
        "equal_parameter_count": len(params) == 1,
        "finite_training": all(report["mode_reports"][mode]["training"]["finite"] for report in reports for mode in MODES),
        "parameters_updated": all(report["mode_reports"][mode]["training"]["parameters_updated"] for report in reports for mode in MODES),
        "primary_gradient_groups_active": all(report["mode_reports"][PRIMARY]["training"]["all_load_bearing_gradient_groups_active"] for report in reports),
        "states_finite": all(report["mode_reports"][mode]["evaluation"]["states_finite"] for report in reports for mode in MODES),
        "deterministic": all(report["mode_reports"][mode]["evaluation"]["deterministic_max_abs"] <= float(tolerance["deterministic_tolerance"]) for report in reports for mode in MODES),
        "candidate_permutation": all(report["mode_reports"][mode]["evaluation"]["candidate_permutation_max_abs"] <= float(tolerance["candidate_permutation_tolerance"]) for report in reports for mode in MODES),
        "nohistory_exact_base": all(report["mode_reports"][mode]["evaluation"]["nohistory_base_max_abs"] == 0.0 for report in reports for mode in MODES),
        "query_absent_exact_base": all(report["mode_reports"][mode]["evaluation"]["query_absent_correction_max_abs"] == 0.0 for report in reports for mode in MODES),
        "repeat_exact_item_only": all(report["mode_reports"][mode]["evaluation"]["repeat_item_only_max_abs"] == 0.0 for report in reports for mode in MODES),
        "null_event_zero_innovation": all(report["mode_reports"][PRIMARY]["evaluation"]["null_event_innovation_max_abs"] == 0.0 for report in reports),
        "checkpoint_hashes": all(sha256_file(REPO_ROOT / report["mode_reports"][mode]["checkpoint"]["path"]) == report["mode_reports"][mode]["checkpoint"]["sha256"] for report in reports for mode in MODES),
        "repository_data_labels_closed": all(not report["repository_data_read"] and not report["repository_labels_read"] for report in reports),
        "dev_test_qrels_closed": all(not report["dev_test_qrels_read"] and report["shared_evaluator_calls"] == 0 for report in reports),
    }
    gate = config["gate"]
    gains = {
        mode: [report["mode_reports"][mode]["evaluation"]["gain_over_base"] for report in reports]
        for mode in MODES
    }
    primary_margins = {
        mode: [
            report["mode_reports"][PRIMARY]["evaluation"]["ndcg10"]["clean"]
            - report["mode_reports"][mode]["evaluation"]["ndcg10"]["clean"]
            for report in reports
        ]
        for mode in MODES
        if mode != PRIMARY
    }
    d1_checks = {
        "primary_gain_each_seed": all(value >= float(gate["primary_gain_over_base_min_each_seed"]) for value in gains[PRIMARY]),
        "primary_control_mean_margins": all(float(np.mean(values)) >= float(gate["primary_margin_over_each_control_mean_min"]) for values in primary_margins.values()),
        "primary_control_positive_seed_counts": all(sum(value > 0 for value in values) >= int(gate["primary_margin_over_each_control_positive_seeds_min"]) for values in primary_margins.values()),
        "wrong_retention": all(report["mode_reports"][PRIMARY]["evaluation"]["wrong_gain_retention"] is not None and report["mode_reports"][PRIMARY]["evaluation"]["wrong_gain_retention"] <= float(gate["primary_wrong_gain_retention_max"]) for report in reports),
        "shuffle_retention": all(report["mode_reports"][PRIMARY]["evaluation"]["shuffle_gain_retention"] is not None and report["mode_reports"][PRIMARY]["evaluation"]["shuffle_gain_retention"] <= float(gate["primary_shuffle_gain_retention_max"]) for report in reports),
        "clicked_direction": all(report["mode_reports"][PRIMARY]["evaluation"]["clicked_minus_unclicked_correction"] > float(gate["primary_clicked_direction_min_each_seed"]) for report in reports),
        "order_activity": all(report["mode_reports"][PRIMARY]["evaluation"]["order_change_fraction"] >= float(gate["primary_order_change_fraction_min_each_seed"]) for report in reports),
    }
    status = "passed_D1_synthetic_only" if all(d0_checks.values()) and all(d1_checks.values()) else ("failed_D0_terminal" if not all(d0_checks.values()) else "failed_D1_terminal")
    result = {
        "candidate_id": "c45",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "proposal_lock_sha256": lock_hash,
        "D0": {"status": "passed" if all(d0_checks.values()) else "failed", "checks": d0_checks},
        "D1": {
            "status": "passed" if all(d1_checks.values()) else "failed",
            "checks": d1_checks,
            "gains_over_mode_specific_base": gains,
            "primary_minus_control_clean_ndcg10": primary_margins,
            "seed_metrics": {
                str(seed): reports[index]["mode_reports"]
                for index, seed in enumerate(seeds)
            },
        },
        "repository_data_read": False,
        "repository_labels_read": False,
        "dev_test_qrels_read": False,
        "shared_evaluator_calls": 0,
        "authorization_after_gate": "new_train_internal_protocol_only" if status == "passed_D1_synthetic_only" else "none_terminal",
    }
    atomic_json(output_path, result)
    atomic_json(promoted_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("seed", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "seed":
        if args.seed is None:
            raise ValueError("C45 seed stage requires --seed")
        value = run_seed(config, int(args.seed), torch.device(args.device))
        status = "complete"
    else:
        value = aggregate(config)
        status = value["status"]
    print(json.dumps({"candidate_id": "c45", "stage": args.stage, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
