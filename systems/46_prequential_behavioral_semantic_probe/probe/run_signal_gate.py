"""Train, score, and aggregate the locked C46 signal gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model import BehavioralSemanticTransformer  # noqa: E402
from myrec.eval.metrics import ndcg_at_k  # noqa: E402
from probe.data import PackedTrain, atomic_json, candidate_key_sha256, sha256_file  # noqa: E402
from probe.metrics import bootstrap, clicked_direction, compare, order, order_change_fraction  # noqa: E402
from probe.protocol import load_config, state_sha256, verify_proposal_lock  # noqa: E402


MODES = ("true_pairs", "shuffled_pairs")
PRIMARY = "true_pairs"


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def make_model(config: Mapping[str, Any]) -> BehavioralSemanticTransformer:
    row = config["model"]
    return BehavioralSemanticTransformer(
        input_dim=int(row["input_dim"]),
        width=int(row["width"]),
        heads=int(row["heads"]),
        layers=int(row["layers"]),
        ff_multiplier=int(row["ff_multiplier"]),
        max_history=int(config["source"]["max_history"]),
        temperature=float(row["temperature"]),
    )


def load_g0(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    root = REPO_ROOT / config["paths"]["artifact_root"]
    report = json.loads((root / "g0_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed" or report.get("A_labels_opened") is not False:
        raise PermissionError("C46 G0 did not pass with A labels closed")
    values = {}
    for name, row in report["outputs"].items():
        path = REPO_ROOT / row["path"]
        if sha256_file(path) != row["sha256"]:
            raise RuntimeError(f"C46 G0 artifact changed: {name}")
        values[name] = np.load(path, mmap_mode="r")
    return report, values


def parameter_groups(names: set[str]) -> dict[str, bool]:
    return {
        "item_projection": any(name.startswith("item_projection.") for name in names),
        "transformer": any(name.startswith("transformer.") for name in names),
        "read_position": any(name in {"read_token", "position"} for name in names),
        "output_norm": any(name.startswith("output_norm.") for name in names),
    }


def batch_histories(
    positions: np.ndarray,
    offsets: np.ndarray,
    items: np.ndarray,
    max_history: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows = [np.asarray(items[int(offsets[p]):int(offsets[p + 1])], dtype=np.int64)[-max_history:] for p in positions]
    width = max((len(row) for row in rows), default=0)
    values = np.zeros((len(rows), width), dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=bool)
    for index, row in enumerate(rows):
        values[index, : len(row)] = row
        mask[index, : len(row)] = True
    return values, mask


def train_model(
    model: BehavioralSemanticTransformer,
    mode: str,
    arrays: Mapping[str, np.ndarray],
    raw_items: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    offsets = arrays["source_offsets.npy"]
    prefix_items = arrays["source_items.npy"]
    targets = arrays["source_targets.npy"] if mode == PRIMARY else arrays["source_shuffled_targets.npy"]
    negative_pool = np.asarray(arrays["negative_pool.npy"], dtype=np.int64)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    rng = np.random.default_rng(seed + 461)
    losses = []
    gradients: set[str] = set()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.train()
    for _ in range(int(training["steps"])):
        positions = rng.integers(0, len(targets), size=int(training["batch_size"]))
        history_indices, history_mask = batch_histories(positions, offsets, prefix_items, int(config["source"]["max_history"]))
        target = np.asarray(targets[positions], dtype=np.int64)
        negative_positions = rng.integers(0, len(negative_pool), size=(len(positions), int(training["negatives"])))
        negatives = negative_pool[negative_positions]
        equal = negatives == target[:, None]
        negatives[equal] = negative_pool[(negative_positions[equal] + 1) % len(negative_pool)]
        candidate_indices = np.concatenate((target[:, None], negatives), axis=1)
        history = torch.from_numpy(np.asarray(raw_items[history_indices], dtype=np.float32)).to(device)
        mask = torch.from_numpy(history_mask).to(device)
        candidates = torch.from_numpy(np.asarray(raw_items[candidate_indices], dtype=np.float32)).to(device)
        score = model.score(history, mask, candidates)
        loss = F.cross_entropy(score, torch.zeros(len(positions), dtype=torch.long, device=device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("nonfinite C46 training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"nonfinite C46 gradient: {name}")
                if bool(parameter.grad.ne(0).any()):
                    gradients.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    updated = [name for name, value in model.named_parameters() if not torch.equal(initial[name], value.detach())]
    groups = parameter_groups(gradients)
    return {
        "steps": len(losses),
        "finite": bool(np.isfinite(losses).all()),
        "loss_first_30": float(np.mean(losses[:30])),
        "loss_last_30": float(np.mean(losses[-30:])),
        "gradient_parameter_names": sorted(gradients),
        "gradient_groups": groups,
        "all_gradient_groups_active": all(groups.values()),
        "updated_parameter_names": updated,
        "parameters_updated": bool(updated),
    }


def pad_request_rows(
    histories: Sequence[np.ndarray], candidates: Sequence[np.ndarray], raw_items: np.ndarray, device: torch.device, max_history: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    history_width = max((min(len(row), max_history) for row in histories), default=0)
    candidate_width = max(len(row) for row in candidates)
    h_indices = np.zeros((len(histories), history_width), dtype=np.int64)
    h_mask = np.zeros((len(histories), history_width), dtype=bool)
    c_indices = np.zeros((len(candidates), candidate_width), dtype=np.int64)
    counts = []
    for row_index, row in enumerate(histories):
        row = np.asarray(row[-max_history:], dtype=np.int64)
        h_indices[row_index, : len(row)] = row
        h_mask[row_index, : len(row)] = True
    for row_index, row in enumerate(candidates):
        row = np.asarray(row, dtype=np.int64)
        c_indices[row_index, : len(row)] = row
        counts.append(len(row))
    return (
        torch.from_numpy(np.asarray(raw_items[h_indices], dtype=np.float32)).to(device),
        torch.from_numpy(h_mask).to(device),
        torch.from_numpy(np.asarray(raw_items[c_indices], dtype=np.float32)).to(device),
        counts,
    )


def score_model_rows(
    model: BehavioralSemanticTransformer,
    data: PackedTrain,
    indices: Sequence[int],
    donors: Sequence[int],
    raw_items: np.ndarray,
    device: torch.device,
    max_history: int,
    source: str,
    *,
    reverse_candidates: bool = False,
) -> list[np.ndarray]:
    output: list[np.ndarray] = []
    for start in range(0, len(indices), 64):
        block = indices[start : start + 64]
        donor_block = donors[start : start + 64]
        histories = []
        candidates = []
        for index, donor in zip(block, donor_block):
            history = data.history(index if source != "wrong" else donor)
            if source == "reverse":
                history = history[::-1].copy()
            histories.append(history)
            row = data.candidates(index)
            if reverse_candidates:
                row = row[::-1].copy()
            candidates.append(row)
        history, mask, candidate, counts = pad_request_rows(histories, candidates, raw_items, device, max_history)
        with torch.inference_mode():
            scores = model.score(history, mask, candidate).detach().cpu().numpy()
        output.extend(np.asarray(scores[row, :count], dtype=np.float32) for row, count in enumerate(counts))
    return output


def semantic_rows(
    data: PackedTrain,
    indices: Sequence[int],
    donors: Sequence[int],
    raw_items: np.ndarray,
    source: str,
) -> list[np.ndarray]:
    rows = []
    for index, donor in zip(indices, donors):
        history_indices = data.history(index if source == "true" else donor)
        candidates = np.asarray(raw_items[data.candidates(index)], dtype=np.float32)
        if len(history_indices):
            history = np.asarray(raw_items[history_indices], dtype=np.float32)
            history /= np.maximum(np.linalg.norm(history, axis=1, keepdims=True), 1e-6)
            profile = history.mean(0)
            profile /= max(float(np.linalg.norm(profile)), 1e-6)
            candidates /= np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-6)
            score = candidates @ profile
        else:
            score = np.zeros(len(candidates), dtype=np.float32)
        rows.append(np.asarray(score, dtype=np.float32))
    return rows


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    values = np.concatenate(rows).astype(np.float32, copy=False) if rows else np.empty(0, np.float32)
    return np.asarray(offsets, dtype=np.int64), values


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[int(offsets[i]):int(offsets[i + 1])], dtype=np.float32) for i in range(len(offsets) - 1)]


def run_seed(config: Mapping[str, Any], seed: int, device: torch.device) -> dict[str, Any]:
    _, lock_hash = verify_proposal_lock(config)
    g0, arrays = load_g0(config)
    physical = int(config["resources"]["seed_to_physical_gpu"].get(str(seed), -1))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical) or str(device) != "cuda:0":
        raise RuntimeError("C46 GPU registration differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C46 deterministic CUBLAS setting absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C46 requires exactly one visible GPU")
    raw_items = np.load(REPO_ROOT / config["paths"]["raw_item_embeddings"], mmap_mode="r")
    data = PackedTrain(REPO_ROOT / config["paths"]["packed_train_root"])
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    donors = [int(value) for value in selection["wrong_history_donors"]["indices"]]
    artifact_root = REPO_ROOT / config["paths"]["artifact_root"]
    report_path = artifact_root / f"seed_{seed}_report.json"
    score_path = artifact_root / f"seed_{seed}_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(report_path if report_path.exists() else score_path)
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    reports = {}
    models = {}
    initial_hashes = {}
    for mode in MODES:
        seed_all(seed)
        model = make_model(config).to(device)
        initial_hashes[mode] = state_sha256(model.state_dict())
        training = train_model(model, mode, arrays, raw_items, config, seed, device)
        checkpoint_path = checkpoint_root / f"seed_{seed}_{mode}.pt"
        if checkpoint_path.exists():
            raise FileExistsError(checkpoint_path)
        torch.save({"candidate_id": "c46", "seed": seed, "mode": mode, "proposal_lock_sha256": lock_hash, "state_dict": model.state_dict()}, checkpoint_path)
        reports[mode] = {
            "parameters": sum(value.numel() for value in model.parameters()),
            "training": training,
            "final_state_sha256": state_sha256(model.state_dict()),
            "checkpoint": {"path": str(checkpoint_path.relative_to(REPO_ROOT)), "sha256": sha256_file(checkpoint_path)},
        }
        models[mode] = model.eval()
    max_history = int(config["source"]["max_history"])
    primary_true = score_model_rows(models[PRIMARY], data, indices, donors, raw_items, device, max_history, "true")
    primary_true_again = score_model_rows(models[PRIMARY], data, indices, donors, raw_items, device, max_history, "true")
    primary_wrong = score_model_rows(models[PRIMARY], data, indices, donors, raw_items, device, max_history, "wrong")
    primary_reverse = score_model_rows(models[PRIMARY], data, indices, donors, raw_items, device, max_history, "reverse")
    shuffled_true = score_model_rows(models["shuffled_pairs"], data, indices, donors, raw_items, device, max_history, "true")
    semantic_true = semantic_rows(data, indices, donors, raw_items, "true")
    semantic_wrong = semantic_rows(data, indices, donors, raw_items, "wrong")
    reversed_candidates = score_model_rows(models[PRIMARY], data, indices, donors, raw_items, device, max_history, "true", reverse_candidates=True)
    restored = [row[::-1].copy() for row in reversed_candidates]
    deterministic = max(float(np.max(np.abs(a - b))) for a, b in zip(primary_true, primary_true_again))
    permutation = max(float(np.max(np.abs(a - b))) for a, b in zip(primary_true, restored))
    sample_history = torch.from_numpy(np.asarray(raw_items[np.zeros((8, max_history), dtype=np.int64)], dtype=np.float32)).to(device)
    sample_candidates = torch.from_numpy(np.asarray(raw_items[np.zeros((8, 3), dtype=np.int64)], dtype=np.float32)).to(device)
    with torch.inference_mode():
        nohistory = models[PRIMARY].score(sample_history, torch.zeros(8, max_history, dtype=torch.bool, device=device), sample_candidates)
    item_ids = [data.candidate_ids(index).astype(str).tolist() for index in indices]
    activity = {
        "true_vs_wrong": order_change_fraction(primary_true, primary_wrong, item_ids),
        "primary_vs_shuffled_pairs": order_change_fraction(primary_true, shuffled_true, item_ids),
        "true_vs_reverse": order_change_fraction(primary_true, primary_reverse, item_ids),
    }
    names = {
        "primary_true": primary_true,
        "primary_wrong": primary_wrong,
        "primary_reverse": primary_reverse,
        "shuffled_pairs_true": shuffled_true,
        "semantic_true": semantic_true,
        "semantic_wrong": semantic_wrong,
    }
    offsets, _ = flatten(primary_true)
    flattened = {name: flatten(rows)[1] for name, rows in names.items()}
    np.savez(score_path, offsets=offsets, **flattened)
    report = {
        "candidate_id": "c46",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "physical_gpu": physical,
        "proposal_lock_sha256": lock_hash,
        "g0_status": g0["status"],
        "paired_initialization": len(set(initial_hashes.values())) == 1,
        "initial_state_sha256": initial_hashes,
        "mode_reports": reports,
        "scoring": {
            "requests": len(indices),
            "candidate_key_sha256": candidate_key_sha256(data, indices),
            "deterministic_max_abs": deterministic,
            "candidate_permutation_max_abs": permutation,
            "nohistory_max_abs": float(nohistory.abs().max().cpu()),
            "activity": activity,
            "states_scores_finite": all(np.isfinite(row).all() for rows in names.values() for row in rows),
        },
        "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)},
        "source_labels_opened": True,
        "A_features_scores_opened": True,
        "A_labels_opened": False,
        "dev_test_qrels_read": False,
    }
    atomic_json(report_path, report)
    return report


def labels_for(data: PackedTrain, indices: Sequence[int], path: Path) -> list[np.ndarray]:
    source = np.load(path, mmap_mode="r")
    rows = []
    for index in indices:
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        rows.append(np.asarray(source[start:stop], dtype=np.float32).copy())
    return rows


def ndcg_rows(item_ids: Sequence[Sequence[str]], scores: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> np.ndarray:
    values = []
    for items, score, label in zip(item_ids, scores, labels):
        ranked = [str(items[i]) for i in order(np.asarray(score), items)]
        positives = {str(item) for item, value in zip(items, label) if value > 0}
        values.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(values, dtype=np.float64)


def average_rows(groups: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean([group[row] for group in groups], axis=0).astype(np.float32) for row in range(len(groups[0]))]


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_proposal_lock(config)
    g0, _ = load_g0(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    output = root / "signal_gate_report.json"
    promoted = REPO_ROOT / config["paths"]["promoted_report"]
    if output.exists() or promoted.exists():
        raise FileExistsError(output if output.exists() else promoted)
    seeds = [int(value) for value in config["training"]["seeds"]]
    reports = [json.loads((root / f"seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in seeds]
    score_rows = {}
    names = ("primary_true", "primary_wrong", "primary_reverse", "shuffled_pairs_true", "semantic_true", "semantic_wrong")
    for seed, report in zip(seeds, reports):
        path = REPO_ROOT / report["score_artifact"]["path"]
        if sha256_file(path) != report["score_artifact"]["sha256"]:
            raise RuntimeError("C46 score artifact changed")
        with np.load(path, allow_pickle=False) as values:
            offsets = np.asarray(values["offsets"], dtype=np.int64)
            score_rows[seed] = {name: unflatten(offsets, values[name]) for name in names}
    params = {report["mode_reports"][mode]["parameters"] for report in reports for mode in MODES}
    gate = config["gate"]
    evaluation = config["evaluation"]
    a0_checks = {
        "g0_passed": g0["status"] == "passed",
        "paired_initialization": all(report["paired_initialization"] for report in reports),
        "equal_parameters": len(params) == 1,
        "finite_training": all(report["mode_reports"][mode]["training"]["finite"] for report in reports for mode in MODES),
        "all_gradient_groups": all(report["mode_reports"][mode]["training"]["all_gradient_groups_active"] for report in reports for mode in MODES),
        "parameters_updated": all(report["mode_reports"][mode]["training"]["parameters_updated"] for report in reports for mode in MODES),
        "finite_scores": all(report["scoring"]["states_scores_finite"] for report in reports),
        "deterministic": all(report["scoring"]["deterministic_max_abs"] <= float(evaluation["deterministic_tolerance"]) for report in reports),
        "candidate_permutation": all(report["scoring"]["candidate_permutation_max_abs"] <= float(evaluation["candidate_permutation_tolerance"]) for report in reports),
        "nohistory_exact_zero": all(report["scoring"]["nohistory_max_abs"] == 0.0 for report in reports),
        "true_wrong_active": all(report["scoring"]["activity"]["true_vs_wrong"] >= float(gate["true_vs_wrong_order_fraction_min_each_seed"]) for report in reports),
        "shuffled_pair_active": all(report["scoring"]["activity"]["primary_vs_shuffled_pairs"] >= float(gate["primary_vs_shuffled_pair_order_fraction_min_each_seed"]) for report in reports),
        "checkpoint_hashes": all(sha256_file(REPO_ROOT / report["mode_reports"][mode]["checkpoint"]["path"]) == report["mode_reports"][mode]["checkpoint"]["sha256"] for report in reports for mode in MODES),
        "A_labels_closed_during_training_scoring": all(report["A_labels_opened"] is False for report in reports),
        "candidate_hash": all(report["scoring"]["candidate_key_sha256"] == config["integrity"]["A_candidate_key_sha256"] for report in reports),
        "dev_test_qrels_closed": all(report["dev_test_qrels_read"] is False for report in reports),
    }
    result: dict[str, Any] = {
        "candidate_id": "c46",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "proposal_lock_sha256": lock_hash,
        "A0": {"status": "passed" if all(a0_checks.values()) else "failed", "checks": a0_checks, "seed_activity": {str(seed): report["scoring"]["activity"] for seed, report in zip(seeds, reports)}},
        "source_labels_opened": True,
        "A_features_scores_opened": True,
        "A_labels_opened": False,
        "dev_test_qrels_read": False,
    }
    if not all(a0_checks.values()):
        result["status"] = "failed_A0_terminal"
        atomic_json(output, result)
        atomic_json(promoted, result)
        return result
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    data = PackedTrain(REPO_ROOT / config["paths"]["packed_train_root"])
    indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    request_ids = [data.request_ids[index] for index in indices]
    item_ids = [data.candidate_ids(index).astype(str).tolist() for index in indices]
    labels = labels_for(data, indices, REPO_ROOT / config["paths"]["train_candidate_labels"])
    averaged = {name: average_rows([score_rows[seed][name] for seed in seeds]) for name in names}
    averaged_ndcg = {name: ndcg_rows(item_ids, rows, labels) for name, rows in averaged.items()}
    seed_ndcg = {seed: {name: ndcg_rows(item_ids, rows, labels) for name, rows in score_rows[seed].items()} for seed in seeds}
    comparisons = compare(
        request_ids,
        averaged_ndcg["primary_true"],
        {"wrong_history": averaged_ndcg["primary_wrong"], "shuffled_pairs": averaged_ndcg["shuffled_pairs_true"], "semantic_mean": averaged_ndcg["semantic_true"]},
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    score_name = {"wrong_history": "primary_wrong", "shuffled_pairs": "shuffled_pairs_true", "semantic_mean": "semantic_true"}
    seed_differences = {name: {str(seed): float((seed_ndcg[seed]["primary_true"] - seed_ndcg[seed][target]).mean()) for seed in seeds} for name, target in score_name.items()}
    clicked_true = clicked_direction(averaged["primary_true"], labels)
    clicked_wrong = clicked_direction(averaged["primary_wrong"], labels)
    clicked = bootstrap(clicked_true, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 20)
    clicked_specific = bootstrap(clicked_true - clicked_wrong, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 21)

    def signs(name: str) -> bool:
        return all(value > 0 for value in seed_differences[name].values()) and all(row["mean_difference"] > 0 for row in comparisons[name]["hash_folds"])

    thresholds = {"wrong_history": float(gate["primary_minus_wrong_ndcg_min"]), "shuffled_pairs": float(gate["primary_minus_shuffled_pair_ndcg_min"]), "semantic_mean": float(gate["primary_minus_semantic_ndcg_min"])}
    a1_checks = {}
    for name, threshold in thresholds.items():
        a1_checks[f"{name}_effect"] = comparisons[name]["mean"] >= threshold
        a1_checks[f"{name}_ci"] = comparisons[name]["percentile_95_ci"][0] > 0
        a1_checks[f"{name}_seed_fold_signs"] = signs(name)
    a1_checks["clicked_direction_ci"] = clicked["percentile_95_ci"][0] > 0
    a1_checks["clicked_specificity_ci"] = clicked_specific["percentile_95_ci"][0] > 0
    result["A1"] = {
        "status": "passed" if all(a1_checks.values()) else "failed",
        "checks": a1_checks,
        "comparisons": comparisons,
        "seed_differences": seed_differences,
        "seed_averaged_ndcg10": {name: float(value.mean()) for name, value in averaged_ndcg.items()},
        "clicked_direction": clicked,
        "clicked_true_minus_wrong": clicked_specific,
    }
    result["A_labels_opened"] = True
    result["status"] = "passed_A1_behavioral_signal_only" if all(a1_checks.values()) else "failed_A1_terminal"
    atomic_json(output, result)
    atomic_json(promoted, result)
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
            raise ValueError("C46 seed stage requires --seed")
        run_seed(config, int(args.seed), torch.device(args.device))
        status = "complete"
    else:
        status = aggregate(config)["status"]
    print(json.dumps({"candidate_id": "c46", "stage": args.stage, "status": status}, sort_keys=True))


if __name__ == "__main__":
    main()
