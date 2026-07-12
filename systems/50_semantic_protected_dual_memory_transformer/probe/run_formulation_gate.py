"""Score C50 using the six frozen C49 predictors."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C49_ROOT = REPO_ROOT / "systems/49_prequential_innovation_memory_transformer"
C49_PROBE = C49_ROOT / "probe"
C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(REPO_ROOT / "src"), str(C38_ROOT), str(C47_ROOT), str(C49_PROBE)):
    if value not in sys.path:
        sys.path.insert(0, value)

from freeze_protocol import load_config, verify_lock, write_once  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C49 = _load("c50_c49_runtime", C49_PROBE / "run_learnability_gate.py")
DUAL = _load("c50_dual_runtime", SYSTEM_ROOT / "model/dual_memory.py")

SCORE_NAMES = (
    "base",
    "primary_true",
    "primary_wrong",
    "raw_semantic",
    "innovation",
    "unprojected_sum",
    "c47_plain",
    "c47_posterior",
    "c47_softmax",
    "primary_correction",
    "wrong_correction",
)


def dual_scores(model, query, history, candidates, c49_config, c50_config, device):
    with torch.inference_mode():
        keys, predictions = C49.prequential_states(model, history, device, int(c49_config["model"]["max_history"]))
        q = model.encode_items(torch.from_numpy(np.array(query, dtype=np.float32, copy=True))[None].to(device))
        c = model.encode_items(torch.from_numpy(np.array(candidates, dtype=np.float32, copy=True, order="C")).to(device))
        if len(keys):
            memory = C49.innovation_memory_reads(
                q,
                keys[None],
                predictions[None],
                torch.ones(1, len(keys), dtype=torch.bool, device=device),
                ridge=float(c49_config["memory"]["ridge"]),
                softmax_temperature=float(c49_config["memory"]["softmax_temperature"]),
                epsilon=float(c49_config["memory"]["normalization_epsilon"]),
            )
            dual = DUAL.semantic_protected_reads(memory.raw_krr, memory.primary, epsilon=float(c50_config["operator"]["epsilon"]))
        else:
            width = int(c49_config["model"]["width"])
            zero = torch.zeros(1, width, device=device)
            dual = DUAL.semantic_protected_reads(zero, zero, epsilon=float(c50_config["operator"]["epsilon"]))
        scale = float(c50_config["operator"]["correction_scale"])
        output = {
            name: (scale * (c @ getattr(dual, name)[0])).cpu().numpy().astype(np.float32, copy=False)
            for name in ("primary", "raw_semantic", "innovation", "unprojected_sum")
        }
        output["orthogonality"] = float((dual.raw_semantic * dual.orthogonal_innovation).sum().abs().cpu())
        output["semantic_norm"] = float(dual.raw_semantic.norm().cpu())
    return output


def flatten(rows):
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def unflatten(offsets, values):
    return [np.asarray(values[int(offsets[i]) : int(offsets[i + 1])], np.float32).copy() for i in range(len(offsets) - 1)]


def run_seed(config: Mapping[str, Any], domain: str, seed: int, device: torch.device) -> dict[str, Any]:
    _, lock_hash = verify_lock(config)
    c49 = C49.load_config(REPO_ROOT / config["paths"]["c49_config"])
    c47 = C49.load_c47_config(REPO_ROOT / c49["paths"]["c47_config"])
    c38 = C49.load_c38_config(REPO_ROOT / c49["paths"]["c38_config"])
    physical = int(config["resources"][f"{domain}_seed_to_physical_gpu"].get(str(seed), -1))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical) or str(device) != "cuda:0":
        raise RuntimeError("C50 GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C50 requires one visible GPU")
    source_report = json.loads((REPO_ROOT / c49["paths"]["artifact_root"] / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8"))
    checkpoint = REPO_ROOT / source_report["checkpoint"]["path"]
    if C49.sha256_file(checkpoint) != source_report["checkpoint"]["sha256"]:
        raise RuntimeError("C50 source checkpoint changed")
    store = C49.DomainStore(domain, c47, c38)
    model = C49.make_model(c49, store.input_dim).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    indices, donors = store.a_indices(), store.donors()
    expected = c47["integrity"][f"{domain}_candidate_key_sha256"]
    if C49.candidate_key_sha256(store.eval_store, indices) != expected:
        raise RuntimeError("C50 candidate hash differs")
    prior_report = json.loads((REPO_ROOT / c47["paths"]["artifact_root"] / f"{domain}_fixed_score_report.json").read_text(encoding="utf-8"))
    prior = C49.load_score_rows(REPO_ROOT / c47["paths"]["artifact_root"], prior_report)
    rows = {name: [] for name in SCORE_NAMES}
    deterministic_max = 0.0
    candidate_permutation_max = 0.0
    orthogonality_max = 0.0
    for position, (index, donor) in enumerate(zip(indices, donors)):
        query, candidates = store.query(index), store.candidates(index)
        true_history = store.eval_sequence(index, source="true")
        wrong_history = store.eval_sequence(index, source="wrong", donor=donor)
        true = dual_scores(model, query, true_history, candidates, c49, config, device)
        again = dual_scores(model, query, true_history, candidates, c49, config, device)
        wrong = dual_scores(model, query, wrong_history, candidates, c49, config, device)
        reverse = dual_scores(model, query, true_history, candidates[::-1], c49, config, device)
        base = prior["base"][position]
        deterministic_max = max(deterministic_max, float(np.max(np.abs(true["primary"] - again["primary"]))))
        candidate_permutation_max = max(candidate_permutation_max, float(np.max(np.abs(true["primary"] - reverse["primary"][::-1]))))
        orthogonality_max = max(orthogonality_max, true["orthogonality"], wrong["orthogonality"])
        rows["base"].append(base)
        rows["primary_true"].append(base + true["primary"])
        rows["primary_wrong"].append(base + wrong["primary"])
        rows["raw_semantic"].append(base + true["raw_semantic"])
        rows["innovation"].append(base + true["innovation"])
        rows["unprojected_sum"].append(base + true["unprojected_sum"])
        rows["c47_plain"].append(prior["plain_ridge"][position])
        rows["c47_posterior"].append(prior["posterior_supported"][position])
        rows["c47_softmax"].append(prior["softmax_attention"][position])
        rows["primary_correction"].append(true["primary"])
        rows["wrong_correction"].append(wrong["primary"])
    nohistory = dual_scores(
        model,
        np.ones(store.input_dim, dtype=np.float32),
        np.empty((0, store.input_dim), dtype=np.float32),
        np.eye(store.input_dim, dtype=np.float32)[:3],
        c49,
        config,
        device,
    )
    root = REPO_ROOT / config["paths"]["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / f"{domain}_seed_{seed}_scores.npz"
    report_path = root / f"{domain}_seed_{seed}_report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(report_path)
    offsets, _ = flatten(rows["base"])
    with score_path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(values)[1] for name, values in rows.items()})
    checks = {
        "candidate_hash": True,
        "finite": all(np.isfinite(row).all() for values in rows.values() for row in values),
        "deterministic": deterministic_max <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": candidate_permutation_max <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "orthogonality": orthogonality_max <= 2e-6,
        "nohistory_exact_zero": all(
            np.count_nonzero(nohistory[name]) == 0
            for name in ("primary", "raw_semantic", "innovation", "unprojected_sum")
        ),
        "optimizer_steps_zero": True,
        "fresh_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c50",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "seed": seed,
        "physical_gpu": physical,
        "proposal_lock_sha256": lock_hash,
        "checks": checks,
        "deterministic_max_abs": deterministic_max,
        "candidate_permutation_max_abs": candidate_permutation_max,
        "orthogonality_max_abs": orthogonality_max,
        "source_checkpoint": source_report["checkpoint"],
        "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": C49.sha256_file(score_path)},
        "optimizer_steps": 0,
        "fresh_reserve_dev_test_qrels_opened": False,
    }
    write_once(report_path, report)
    return report


def load_rows(report):
    path = REPO_ROOT / report["score_artifact"]["path"]
    if C49.sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C50 score artifact changed")
    with np.load(path, allow_pickle=False) as values:
        offsets = np.asarray(values["offsets"], np.int64)
        return {name: unflatten(offsets, values[name]) for name in SCORE_NAMES}


def run_a0(config):
    _, lock_hash = verify_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    reports = [
        json.loads((root / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8"))
        for domain in ("kuai", "amazon")
        for seed in C49.load_config(REPO_ROOT / config["paths"]["c49_config"])["training"][f"{domain}_seeds"]
    ]
    checks = {
        "all_seed_checks": all(all(report["checks"].values()) for report in reports),
        "score_hashes": all(C49.sha256_file(REPO_ROOT / report["score_artifact"]["path"]) == report["score_artifact"]["sha256"] for report in reports),
        "optimizer_steps_zero": all(report["optimizer_steps"] == 0 for report in reports),
        "fresh_dev_test_qrels_closed": all(report["fresh_reserve_dev_test_qrels_opened"] is False for report in reports),
    }
    value = {"candidate_id": "c50", "created_at": datetime.now(timezone.utc).isoformat(), "status": "passed" if all(checks.values()) else "failed_A0_terminal", "proposal_lock_sha256": lock_hash, "checks": checks, "fresh_reserve_dev_test_qrels_opened": False}
    write_once(root / "a0_report.json", value)
    return value


def average_rows(groups):
    return [np.mean(np.stack(values), axis=0).astype(np.float32) for values in zip(*groups)]


def aggregate_domain(config, domain, c49, c47, c38):
    store = C49.DomainStore(domain, c47, c38)
    indices = store.a_indices()
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    labels = C49.kuai_labels(c47, store.eval_store, indices) if domain == "kuai" else C49.amazon_labels(c47, store.eval_store, indices)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    seeds = c49["training"][f"{domain}_seeds"]
    reports = [json.loads((root / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in seeds]
    seed_rows = {seed: load_rows(report) for seed, report in zip(seeds, reports)}
    ensemble = {name: average_rows([seed_rows[seed][name] for seed in seeds]) for name in SCORE_NAMES}
    ndcg = {name: C49.ndcg_rows(request_ids, item_ids, rows, labels) for name, rows in ensemble.items() if name not in {"primary_correction", "wrong_correction"}}
    seed_ndcg = {seed: {name: C49.ndcg_rows(request_ids, item_ids, rows, labels) for name, rows in seed_rows[seed].items() if name not in {"primary_correction", "wrong_correction"}} for seed in seeds}
    refs = {"base": "base", "raw_semantic": "raw_semantic", "unprojected_sum": "unprojected_sum", "innovation": "innovation", "c47_plain": "c47_plain", "c47_posterior": "c47_posterior", "c47_softmax": "c47_softmax", "wrong_history": "primary_wrong"}
    evaluation = config["evaluation"]
    comparisons = C49.compare(request_ids, ndcg["primary_true"], {name: ndcg[target] for name, target in refs.items()}, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]), folds=int(evaluation["hash_folds"]))
    seed_diff = {name: {str(seed): float((seed_ndcg[seed]["primary_true"] - seed_ndcg[seed][target]).mean()) for seed in seeds} for name, target in refs.items()}
    clicked_true = C49.clicked_direction(ensemble["primary_correction"], labels)
    clicked_wrong = C49.clicked_direction(ensemble["wrong_correction"], labels)
    clicked = C49.bootstrap(clicked_true, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 20)
    specificity = C49.bootstrap(clicked_true - clicked_wrong, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 21)
    thresholds = {"base": float(evaluation["primary_minus_base_min"]), "raw_semantic": float(evaluation["primary_minus_raw_min"]), "unprojected_sum": float(evaluation["primary_minus_unprojected_min"]), "innovation": 0.0, "c47_plain": float(evaluation["primary_minus_c47_best_min"]), "c47_posterior": float(evaluation["primary_minus_c47_best_min"]), "c47_softmax": float(evaluation["primary_minus_c47_best_min"]), "wrong_history": float(evaluation["true_minus_wrong_min"])}
    checks = {}
    for name, threshold in thresholds.items():
        row = comparisons[name]
        checks[f"{name}_effect"] = row["mean"] >= threshold
        checks[f"{name}_ci"] = row["percentile_95_ci"][0] > 0
        checks[f"{name}_all_seed_fold_positive"] = all(value > 0 for value in seed_diff[name].values()) and all(fold["mean_difference"] > 0 for fold in row["hash_folds"])
    checks["clicked_direction_ci"] = clicked["percentile_95_ci"][0] > 0
    checks["clicked_specificity_ci"] = specificity["percentile_95_ci"][0] > 0
    return {"status": "passed" if all(checks.values()) else "failed", "requests": len(indices), "checks": checks, "mean_ndcg10": {name: float(values.mean()) for name, values in ndcg.items()}, "comparisons": comparisons, "seed_differences": seed_diff, "clicked_direction": clicked, "clicked_true_minus_wrong": specificity}


def aggregate(config):
    _, lock_hash = verify_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0 = json.loads((root / "a0_report.json").read_text(encoding="utf-8"))
    if a0.get("status") != "passed":
        raise PermissionError("C50 A0 failed")
    c49 = C49.load_config(REPO_ROOT / config["paths"]["c49_config"])
    c47 = C49.load_c47_config(REPO_ROOT / c49["paths"]["c47_config"])
    c38 = C49.load_c38_config(REPO_ROOT / c49["paths"]["c38_config"])
    domains = {domain: aggregate_domain(config, domain, c49, c47, c38) for domain in ("kuai", "amazon")}
    passed = all(row["status"] == "passed" for row in domains.values())
    value = {"candidate_id": "c50", "created_at": datetime.now(timezone.utc).isoformat(), "gate_id": config["gate_id"], "status": "passed_exposed_formulation_only" if passed else "failed_formulation_terminal", "decision": "authorize_separately_locked_C50_training_gate" if passed else "close_C50_before_training_or_fresh_reserve", "proposal_lock_sha256": lock_hash, "domains": domains, "optimizer_steps": 0, "fresh_reserve_dev_test_qrels_opened": False, "claims": {"exposed_formulation_only": True, "trained_result": False, "fresh_result": False, "dev_test_result": False}}
    write_once(root / "formulation_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("seed", "a0", "aggregate"))
    parser.add_argument("--domain", choices=("kuai", "amazon"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "seed":
        c49 = C49.load_config(REPO_ROOT / config["paths"]["c49_config"])
        if args.domain is None or args.seed is None or args.seed not in c49["training"][f"{args.domain}_seeds"]:
            raise ValueError("C50 seed/domain not registered")
        value = run_seed(config, args.domain, args.seed, torch.device(args.device))
    elif args.stage == "a0":
        value = run_a0(config)
    else:
        value = aggregate(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
