from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import atomic_json, load_config, sha256_file, timestamp, verify_execution_lock  # noqa: E402
from execution.runtime import DomainStore, unflatten  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from probe.run_signal_gate import amazon_labels, kuai_labels  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402


SCORE_NAMES = (
    "primary_true", "primary_wrong", "random_true", "semantic_true", "semantic_wrong"
)


def load_rows(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C69 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def mean_rows(seed_rows: Sequence[Sequence[np.ndarray]], name: str) -> list[np.ndarray]:
    output = []
    for request in range(len(seed_rows[0])):
        output.append(np.mean([rows[request] for rows in seed_rows], axis=0).astype(np.float32))
    return output


def ndcg_rows(
    request_ids: Sequence[str], item_ids: Sequence[Sequence[str]],
    scores: Sequence[np.ndarray], labels: Sequence[np.ndarray],
) -> np.ndarray:
    output = []
    for request_id, items, values, label in zip(request_ids, item_ids, scores, labels):
        ranked = [
            row.item_id
            for row in sort_candidates(
                request_id,
                [ScoredCandidate(str(item), float(score)) for item, score in zip(items, values)],
            )
        ]
        positives = {str(item) for item, value in zip(items, label) if value > 0}
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, dtype=np.float64)


def comparison_pass(row: Mapping[str, Any], minimum: float) -> bool:
    return bool(
        row["mean"] >= minimum
        and row["percentile_95_ci"][0] > 0
        and all(fold["mean_difference"] > 0 for fold in row["hash_folds"])
    )


def aggregate_domain(
    config: Mapping[str, Any], domain: str, reports: Sequence[Mapping[str, Any]],
    c47: Mapping[str, Any], c38: Mapping[str, Any],
) -> dict[str, Any]:
    store = DomainStore(domain, c47, c38)
    store.assert_candidate_hash()
    indices = store.a_indices()
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    labels = kuai_labels(c47, store.eval_store, indices) if domain == "kuai" else amazon_labels(c47, store.eval_store, indices)
    loaded = [load_rows(report) for report in reports]
    ensemble = {
        name: mean_rows([row[name] for row in loaded], name) for name in SCORE_NAMES
    }
    ensemble_ndcg = {
        name: ndcg_rows(request_ids, item_ids, rows, labels) for name, rows in ensemble.items()
    }
    eval_cfg = config["evaluation"]
    comparisons = compare(
        request_ids,
        ensemble_ndcg["primary_true"],
        {
            "semantic_true": ensemble_ndcg["semantic_true"],
            "random_true": ensemble_ndcg["random_true"],
            "primary_wrong": ensemble_ndcg["primary_wrong"],
        },
        samples=int(eval_cfg["bootstrap_samples"]),
        seed=int(eval_cfg["bootstrap_seed"]),
        folds=int(eval_cfg["hash_folds"]),
    )
    per_seed = []
    seed_signs = {"semantic_true": [], "random_true": [], "primary_wrong": []}
    for report, rows in zip(reports, loaded):
        metrics = {name: ndcg_rows(request_ids, item_ids, values, labels) for name, values in rows.items()}
        deltas = {
            "semantic_true": float((metrics["primary_true"] - metrics["semantic_true"]).mean()),
            "random_true": float((metrics["primary_true"] - metrics["random_true"]).mean()),
            "primary_wrong": float((metrics["primary_true"] - metrics["primary_wrong"]).mean()),
        }
        for name, value in deltas.items():
            seed_signs[name].append(value > 0)
        per_seed.append({
            "seed": report["seed"],
            "ndcg@10": {name: float(value.mean()) for name, value in metrics.items()},
            "deltas": deltas,
        })
    direction = bootstrap(
        clicked_direction(ensemble["primary_true"], labels),
        samples=int(eval_cfg["bootstrap_samples"]),
        seed=int(eval_cfg["bootstrap_seed"]) + 30,
    )
    checks = {
        "primary_beats_semantic": comparison_pass(
            comparisons["semantic_true"], float(eval_cfg["primary_minus_semantic_min"])
        ),
        "primary_beats_random": comparison_pass(
            comparisons["random_true"], float(eval_cfg["primary_minus_random_min"])
        ),
        "true_beats_wrong": comparison_pass(
            comparisons["primary_wrong"], float(eval_cfg["true_minus_wrong_min"])
        ),
        "all_seed_signs": all(all(values) for values in seed_signs.values()),
        "clicked_direction": direction["percentile_95_ci"][0] > 0,
    }
    return {
        "domain": domain,
        "requests": len(indices),
        "ensemble_ndcg@10": {name: float(value.mean()) for name, value in ensemble_ndcg.items()},
        "comparisons": comparisons,
        "clicked_direction": direction,
        "per_seed": per_seed,
        "checks": checks,
        "passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _, lock_hash = verify_execution_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    reports: dict[str, list[dict[str, Any]]] = {}
    source_reports = []
    a0_pass = True
    for domain in ("kuai", "amazon"):
        rows = []
        for seed in config["training"]["seeds"][domain]:
            path = root / f"{domain}_seed_{int(seed)}_report.json"
            report = json.loads(path.read_text())
            if report["execution_lock_sha256"] != lock_hash or report["domain"] != domain or report["seed"] != int(seed):
                raise RuntimeError("C69 seed report identity mismatch")
            if sha256_file(REPO_ROOT / report["score_artifact"]["path"]) != report["score_artifact"]["sha256"]:
                raise RuntimeError("C69 score hash mismatch")
            rows.append(report)
            source_reports.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)})
            a0_pass &= bool(report["passed_A0"])
        reports[domain] = rows
    target = REPO_ROOT / config["paths"]["promoted_report"]
    if not a0_pass:
        value = {
            "schema": "myrec.c69.gate.v1", "candidate_id": "c69", "created_at": timestamp(),
            "execution_lock_sha256": lock_hash, "source_reports": source_reports,
            "A0_passed": False, "A_labels_opened": False, "passed": False,
            "decision": "failed_A0_terminal",
            "isolation": {"fresh_reserve_opened": False, "dev_test_qrels_opened": False},
        }
        atomic_json(target, value)
        print(target.relative_to(REPO_ROOT)); print(value["decision"])
        return
    c47 = load_config(REPO_ROOT / config["paths"]["c47_config"])
    c38 = load_config(REPO_ROOT / config["paths"]["c38_config"])
    domains = {
        domain: aggregate_domain(config, domain, reports[domain], c47, c38)
        for domain in ("kuai", "amazon")
    }
    passed = all(value["passed"] for value in domains.values())
    value = {
        "schema": "myrec.c69.gate.v1", "candidate_id": "c69", "created_at": timestamp(),
        "execution_lock_sha256": lock_hash, "source_reports": source_reports,
        "A0_passed": True, "A_labels_opened_after_A0": True, "domains": domains,
        "passed": passed,
        "decision": "authorize_catalog_open_architecture_formulation" if passed else "failed_signal_terminal",
        "isolation": {"fresh_reserve_opened": False, "dev_test_qrels_opened": False},
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT)); print(sha256_file(target)); print(value["decision"])


if __name__ == "__main__":
    main()
