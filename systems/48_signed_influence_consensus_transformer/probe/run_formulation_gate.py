"""Evaluate the locked C48 operator on already-open C47 formulation cohorts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(REPO_ROOT / "src"), str(C38_ROOT), str(C47_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

MODEL_SPEC = importlib.util.spec_from_file_location(
    "c48_runtime_signed_consensus", SYSTEM_ROOT / "model/signed_consensus.py"
)
assert MODEL_SPEC and MODEL_SPEC.loader
MODEL_MODULE = importlib.util.module_from_spec(MODEL_SPEC)
sys.modules[MODEL_SPEC.name] = MODEL_MODULE
MODEL_SPEC.loader.exec_module(MODEL_MODULE)
signed_consensus_mix = MODEL_MODULE.signed_consensus_mix

from freeze_formulation_lock import load_config, verify_formulation_lock  # noqa: E402
from probe.freeze_signal_lock import load_config as load_c47_config, verify_signal_lock  # noqa: E402
from probe.locking import sha256_file  # noqa: E402
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


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def score_one(
    query: np.ndarray,
    history: np.ndarray,
    candidates: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    settings = config["operator"]
    q = torch.from_numpy(np.asarray(query, dtype=np.float32))[None]
    c = torch.from_numpy(np.asarray(candidates, dtype=np.float32))[None]
    if len(history):
        h = torch.from_numpy(np.asarray(history, dtype=np.float32))[None]
        mask = torch.ones(1, len(history), dtype=torch.bool)
    else:
        h = torch.zeros(1, 1, candidates.shape[1], dtype=torch.float32)
        mask = torch.zeros(1, 1, dtype=torch.bool)
    with torch.inference_mode():
        out = signed_consensus_mix(
            q,
            h,
            mask,
            c,
            ridge=float(settings["ridge"]),
            normalization_epsilon=float(settings["normalization_epsilon"]),
            influence_epsilon=float(settings["influence_epsilon"]),
        )
    return {
        "correction": out.correction[0].numpy().astype(np.float32, copy=False),
        "plain": out.plain_correction[0].numpy().astype(np.float32, copy=False),
        "signed_l1": out.signed_l1_control[0].numpy().astype(np.float32, copy=False),
        "coherence": out.coherence[0].numpy().astype(np.float32, copy=False),
    }


def evaluate_domain(
    config: Mapping[str, Any], c47: Mapping[str, Any], domain: str
) -> dict[str, Any]:
    selection = json.loads((REPO_ROOT / c47["paths"]["selection"]).read_text(encoding="utf-8"))
    artifact = REPO_ROOT / c47["paths"]["artifact_root"]
    prior_report = json.loads((artifact / f"{domain}_fixed_score_report.json").read_text(encoding="utf-8"))
    prior = load_score_rows(artifact, prior_report)
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
        raise RuntimeError(f"C48 {domain} candidate hash differs")
    corrections: list[np.ndarray] = []
    wrong_corrections: list[np.ndarray] = []
    signed_l1: list[np.ndarray] = []
    coherence: list[np.ndarray] = []
    deterministic_max = 0.0
    candidate_permutation_max = 0.0
    history_permutation_max = 0.0
    nohistory_zero = True
    for index, donor in zip(indices, donors):
        query = store.query(index)
        candidates = store.candidates(index)
        history = store.history(index) if domain == "kuai" else store.history(index, "true")
        wrong_history = store.history(donor) if domain == "kuai" else store.history(index, "wrong")
        out = score_one(query, history, candidates, config)
        again = score_one(query, history, candidates, config)
        wrong = score_one(query, wrong_history, candidates, config)
        candidate_reverse = score_one(query, history, candidates[::-1], config)
        history_reverse = score_one(query, history[::-1], candidates, config)
        empty = score_one(query, np.empty((0, candidates.shape[1]), np.float32), candidates, config)
        deterministic_max = max(deterministic_max, float(np.max(np.abs(out["correction"] - again["correction"]))))
        candidate_permutation_max = max(candidate_permutation_max, float(np.max(np.abs(out["correction"] - candidate_reverse["correction"][::-1]))))
        history_permutation_max = max(history_permutation_max, float(np.max(np.abs(out["correction"] - history_reverse["correction"]))))
        nohistory_zero = nohistory_zero and all(np.count_nonzero(value) == 0 for value in empty.values())
        corrections.append(out["correction"])
        wrong_corrections.append(wrong["correction"])
        signed_l1.append(out["signed_l1"])
        coherence.append(out["coherence"])
    primary = [base + correction for base, correction in zip(prior["base"], corrections)]
    wrong_primary = [base + correction for base, correction in zip(prior["base"], wrong_corrections)]
    signed_l1_scores = [base + correction for base, correction in zip(prior["base"], signed_l1)]
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    labels = kuai_labels(c47, store, indices) if domain == "kuai" else amazon_labels(c47, store, indices)
    score_rows = {
        "primary": primary,
        "wrong_primary": wrong_primary,
        "signed_l1": signed_l1_scores,
        "base": prior["base"],
        "plain_ridge": prior["plain_ridge"],
        "posterior_supported": prior["posterior_supported"],
        "softmax_attention": prior["softmax_attention"],
    }
    ndcg = {name: ndcg_rows(request_ids, item_ids, rows, labels) for name, rows in score_rows.items()}
    evaluation = config["evaluation"]
    comparisons = compare(
        request_ids,
        ndcg["primary"],
        {name: ndcg[name] for name in ("base", "plain_ridge", "posterior_supported", "softmax_attention", "signed_l1", "wrong_primary")},
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    click_true = clicked_direction(corrections, labels)
    click_wrong = clicked_direction(wrong_corrections, labels)
    clicked = bootstrap(click_true, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 20)
    specificity = bootstrap(click_true - click_wrong, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 21)
    checks = {
        "candidate_hash": True,
        "finite": all(np.isfinite(row).all() for rows in score_rows.values() for row in rows),
        "coherence_bounds": all(np.all((row >= 0) & (row <= 1)) for row in coherence),
        "deterministic": deterministic_max == 0.0,
        "candidate_permutation": candidate_permutation_max <= 1e-6,
        "history_permutation": history_permutation_max <= 2e-5,
        "nohistory_zero": nohistory_zero,
        "base_effect": comparisons["base"]["mean"] >= float(evaluation["primary_minus_base_min"]),
        "base_ci": comparisons["base"]["percentile_95_ci"][0] > 0,
        "plain_ci": comparisons["plain_ridge"]["percentile_95_ci"][0] > 0,
        "plain_all_folds": all(row["mean_difference"] > 0 for row in comparisons["plain_ridge"]["hash_folds"]),
        "softmax_mean": comparisons["softmax_attention"]["mean"] > 0,
        "signed_l1_mean": comparisons["signed_l1"]["mean"] > 0,
        "true_wrong_effect": comparisons["wrong_primary"]["mean"] >= float(evaluation["true_minus_wrong_min"]),
        "true_wrong_ci": comparisons["wrong_primary"]["percentile_95_ci"][0] > 0,
        "clicked_direction_ci": clicked["percentile_95_ci"][0] > 0,
        "clicked_specificity_ci": specificity["percentile_95_ci"][0] > 0,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "requests": len(indices),
        "checks": checks,
        "diagnostics": {
            "deterministic_max_abs": deterministic_max,
            "candidate_permutation_max_abs": candidate_permutation_max,
            "history_permutation_max_abs": history_permutation_max,
            "coherence_mean": float(np.concatenate(coherence).mean()),
        },
        "mean_ndcg10": {name: float(values.mean()) for name, values in ndcg.items()},
        "comparisons": comparisons,
        "clicked_direction": clicked,
        "clicked_true_minus_wrong": specificity,
    }


def run(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_formulation_lock(config)
    c47 = load_c47_config(REPO_ROOT / config["paths"]["c47_config"])
    _, c47_hash = verify_signal_lock(c47)
    domains = {domain: evaluate_domain(config, c47, domain) for domain in ("kuai", "amazon")}
    passed = all(row["status"] == "passed" for row in domains.values())
    value = {
        "candidate_id": "c48",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_id": config["gate_id"],
        "status": "passed_exposed_formulation_only" if passed else "failed_formulation_terminal",
        "decision": "authorize_separately_locked_fresh_C48_design" if passed else "close_signed_influence_consensus_before_fresh_data",
        "formulation_lock_sha256": lock_hash,
        "c47_signal_execution_lock_sha256": c47_hash,
        "domains": domains,
        "claims": {
            "exposed_train_internal_formulation_only": True,
            "fresh_result": False,
            "trained_architecture_result": False,
            "dev_test_result": False,
        },
        "fresh_reserve_opened": False,
        "dev_test_records_labels_qrels_opened": False,
    }
    output_root = REPO_ROOT / config["paths"]["artifact_root"]
    output_root.mkdir(parents=True, exist_ok=False)
    atomic_json(output_root / "formulation_report.json", value)
    atomic_json(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(load_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))
