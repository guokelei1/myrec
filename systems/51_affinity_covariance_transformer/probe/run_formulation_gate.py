"""Evaluate C51 on the already-open C47 formulation cohorts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(REPO_ROOT / "src"), str(C38_ROOT), str(C47_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from freeze_lock import load_config, verify_lock, write_once  # noqa: E402
from probe.freeze_signal_lock import load_config as load_c47_config, verify_signal_lock  # noqa: E402
from probe.run_signal_gate import (  # noqa: E402
    AmazonStore,
    KuaiStore,
    amazon_labels,
    candidate_key_sha256,
    kuai_labels,
    load_score_rows,
    ndcg_rows,
)
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402


SPEC = importlib.util.spec_from_file_location("c51_runtime_operator", SYSTEM_ROOT / "model/affinity_covariance.py")
assert SPEC and SPEC.loader
OPERATOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = OPERATOR
SPEC.loader.exec_module(OPERATOR)


def score_one(query, history, candidates, config):
    q = torch.from_numpy(np.array(query, dtype=np.float32, copy=True))[None]
    c = torch.from_numpy(np.array(candidates, dtype=np.float32, copy=True, order="C"))[None]
    if len(history):
        h = torch.from_numpy(np.array(history, dtype=np.float32, copy=True, order="C"))[None]
        mask = torch.ones(1, len(history), dtype=torch.bool)
    else:
        h = torch.zeros(1, 1, candidates.shape[1])
        mask = torch.zeros(1, 1, dtype=torch.bool)
    with torch.inference_mode():
        output = OPERATOR.affinity_covariance(
            q,
            h,
            mask,
            c,
            normalization_epsilon=float(config["operator"]["normalization_epsilon"]),
            variance_epsilon=float(config["operator"]["variance_epsilon"]),
        )
    scale = float(config["operator"]["correction_scale"])
    return {
        "covariance": (scale * output.covariance[0]).numpy().astype(np.float32, copy=False),
        "uncentered": (scale * output.uncentered_second_moment[0]).numpy().astype(np.float32, copy=False),
        "pearson": (scale * output.pearson_control[0]).numpy().astype(np.float32, copy=False),
    }


def evaluate_domain(config, c47, domain):
    selection = json.loads((REPO_ROOT / c47["paths"]["selection"]).read_text(encoding="utf-8"))
    if domain == "kuai":
        store: Any = KuaiStore(c47)
        role = "kuai_internal_A"
        expected = c47["integrity"]["kuai_candidate_key_sha256"]
    else:
        store = AmazonStore(c47)
        role = "amazon_internal_A"
        expected = c47["integrity"]["amazon_candidate_key_sha256"]
    indices = [int(value) for value in selection["roles"][role]["indices"]]
    donors = [int(value) for value in selection["wrong_history_donors"][role]["indices"]]
    if candidate_key_sha256(store, indices) != expected:
        raise RuntimeError("C51 candidate hash differs")
    artifact = REPO_ROOT / c47["paths"]["artifact_root"]
    prior_report = json.loads((artifact / f"{domain}_fixed_score_report.json").read_text(encoding="utf-8"))
    prior = load_score_rows(artifact, prior_report)
    covariance = []
    wrong_covariance = []
    uncentered = []
    pearson = []
    deterministic_max = 0.0
    candidate_permutation_max = 0.0
    history_permutation_max = 0.0
    usable = 0
    for index, donor in zip(indices, donors):
        query, candidates = store.query(index), store.candidates(index)
        history = store.history(index) if domain == "kuai" else store.history(index, "true")
        wrong_history = store.history(donor) if domain == "kuai" else store.history(index, "wrong")
        output = score_one(query, history, candidates, config)
        again = score_one(query, history, candidates, config)
        wrong = score_one(query, wrong_history, candidates, config)
        candidate_reverse = score_one(query, history, candidates[::-1], config)
        history_reverse = score_one(query, history[::-1], candidates, config)
        deterministic_max = max(deterministic_max, float(np.max(np.abs(output["covariance"] - again["covariance"]))))
        candidate_permutation_max = max(candidate_permutation_max, float(np.max(np.abs(output["covariance"] - candidate_reverse["covariance"][::-1]))))
        history_permutation_max = max(history_permutation_max, float(np.max(np.abs(output["covariance"] - history_reverse["covariance"]))))
        usable += int(len(history) >= 2)
        covariance.append(output["covariance"])
        wrong_covariance.append(wrong["covariance"])
        uncentered.append(output["uncentered"])
        pearson.append(output["pearson"])
    empty = score_one(np.ones(store.query(indices[0]).shape[0], np.float32), np.empty((0, store.query(indices[0]).shape[0]), np.float32), np.eye(store.query(indices[0]).shape[0], dtype=np.float32)[:3], config)
    scores = {
        "primary": [base + correction for base, correction in zip(prior["base"], covariance)],
        "wrong": [base + correction for base, correction in zip(prior["base"], wrong_covariance)],
        "uncentered": [base + correction for base, correction in zip(prior["base"], uncentered)],
        "pearson": [base + correction for base, correction in zip(prior["base"], pearson)],
        "base": prior["base"],
        "plain_krr": prior["plain_ridge"],
        "posterior": prior["posterior_supported"],
        "softmax": prior["softmax_attention"],
    }
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    labels = kuai_labels(c47, store, indices) if domain == "kuai" else amazon_labels(c47, store, indices)
    ndcg = {name: ndcg_rows(request_ids, item_ids, values, labels) for name, values in scores.items()}
    evaluation = config["evaluation"]
    refs = {name: ndcg[name] for name in ("base", "uncentered", "pearson", "plain_krr", "posterior", "softmax", "wrong")}
    comparisons = compare(request_ids, ndcg["primary"], refs, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]), folds=int(evaluation["hash_folds"]))
    clicked_true = clicked_direction(covariance, labels)
    clicked_wrong = clicked_direction(wrong_covariance, labels)
    clicked = bootstrap(clicked_true, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 20)
    specificity = bootstrap(clicked_true - clicked_wrong, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 21)
    thresholds = {"base": float(evaluation["primary_minus_base_min"]), "wrong": float(evaluation["true_minus_wrong_min"]), **{name: float(evaluation["primary_minus_control_min"]) for name in ("uncentered", "pearson", "plain_krr", "posterior", "softmax")}}
    structural = {
        "candidate_hash": True,
        "finite": all(np.isfinite(row).all() for values in scores.values() for row in values),
        "deterministic": deterministic_max == 0.0,
        "candidate_permutation": candidate_permutation_max <= 1e-7,
        "history_permutation": history_permutation_max <= 1e-7,
        "nohistory_covariance_zero": np.count_nonzero(empty["covariance"]) == 0,
        "usable_surface": usable > len(indices) // 2,
    }
    utility = {}
    for name, threshold in thresholds.items():
        row = comparisons[name]
        utility[f"{name}_effect"] = row["mean"] >= threshold
        utility[f"{name}_ci"] = row["percentile_95_ci"][0] > 0
        utility[f"{name}_all_folds_positive"] = all(fold["mean_difference"] > 0 for fold in row["hash_folds"])
    utility["clicked_direction_ci"] = clicked["percentile_95_ci"][0] > 0
    utility["clicked_specificity_ci"] = specificity["percentile_95_ci"][0] > 0
    checks = {**structural, **utility}
    return {"status": "passed" if all(checks.values()) else "failed", "requests": len(indices), "usable_requests": usable, "checks": checks, "diagnostics": {"deterministic_max_abs": deterministic_max, "candidate_permutation_max_abs": candidate_permutation_max, "history_permutation_max_abs": history_permutation_max}, "mean_ndcg10": {name: float(values.mean()) for name, values in ndcg.items()}, "comparisons": comparisons, "clicked_direction": clicked, "clicked_true_minus_wrong": specificity}


def run(config):
    _, lock_hash = verify_lock(config)
    c47 = load_c47_config(REPO_ROOT / config["paths"]["c47_config"])
    verify_signal_lock(c47)
    domains = {domain: evaluate_domain(config, c47, domain) for domain in ("kuai", "amazon")}
    passed = all(row["status"] == "passed" for row in domains.values())
    value = {"candidate_id": "c51", "created_at": datetime.now(timezone.utc).isoformat(), "gate_id": config["gate_id"], "status": "passed_exposed_formulation_only" if passed else "failed_formulation_terminal", "decision": "authorize_separately_locked_C51_training_gate" if passed else "close_C51_before_training_or_fresh_reserve", "proposal_lock_sha256": lock_hash, "domains": domains, "fresh_reserve_dev_test_qrels_opened": False, "claims": {"exposed_formulation_only": True, "trained_result": False, "fresh_result": False, "dev_test_result": False}}
    root = REPO_ROOT / config["paths"]["artifact_root"]
    root.mkdir(parents=True, exist_ok=False)
    write_once(root / "formulation_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))
