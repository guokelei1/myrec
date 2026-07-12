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
sys.path.append(str(REPO_ROOT / "src"))
sys.path.append(str(REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"))

from execution.locking import atomic_json, load_config, sha256_file, timestamp, verify_execution_lock  # noqa: E402
from execution.score_gate import SCORE_NAMES, flatten, rankings  # noqa: E402
from execution.selection import candidate_key_sha256, iter_structural_records  # noqa: E402
from myrec.eval.metrics import ndcg_at_k  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[offsets[i] : offsets[i + 1]], dtype=np.float32).copy() for i in range(len(offsets) - 1)]


def load_scores(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C71 score artifact changed")
    with np.load(path, allow_pickle=False) as values:
        offsets = np.asarray(values["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, values[name]) for name in SCORE_NAMES}


def open_target_labels(
    records_path: Path, targets: Sequence[Mapping[str, Any]], label_field: str
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    wanted = {row["request_id"] for row in targets}
    rows: dict[str, dict[str, Any]] = {}
    labels: dict[str, np.ndarray] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            request_id = str(row["request_id"])
            if request_id not in wanted:
                continue
            structural = {
                "request_id": request_id,
                "candidate_ids": [str(candidate["item_id"]) for candidate in row["candidates"]],
            }
            rows[request_id] = structural
            labels[request_id] = np.asarray(
                [float(candidate[label_field]) for candidate in row["candidates"]], dtype=np.float32
            )
    if set(rows) != wanted:
        raise RuntimeError("C71 target labels incomplete")
    ordered_rows, ordered_labels = [], []
    for target in targets:
        row = rows[target["request_id"]]
        if row["candidate_ids"] != target["candidate_ids"]:
            raise RuntimeError("C71 candidate IDs changed at label open")
        ordered_rows.append(row)
        ordered_labels.append(labels[target["request_id"]])
    return ordered_rows, ordered_labels


def ndcg_rows(
    request_ids: Sequence[str], item_ids: Sequence[Sequence[str]], scores: Sequence[np.ndarray], labels: Sequence[np.ndarray]
) -> np.ndarray:
    output = []
    for request_id, items, values, label in zip(request_ids, item_ids, scores, labels):
        ranked = rankings(items, values)
        positives = {str(item) for item, value in zip(items, label) if value > 0}
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, dtype=np.float64)


def comparison_pass(row: Mapping[str, Any], minimum: float) -> bool:
    return bool(
        row["mean"] >= minimum
        and row["percentile_95_ci"][0] > 0
        and all(fold["mean_difference"] > 0 for fold in row["hash_folds"])
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    _, lock_hash = verify_execution_lock(config)
    paths = config["paths"]
    root = REPO_ROOT / paths["artifact_root"]
    a0_path = root / "a0_report.json"
    a0 = json.loads(a0_path.read_text(encoding="utf-8"))
    if a0["execution_lock_sha256"] != lock_hash:
        raise RuntimeError("C71 A0 lock differs")
    target = REPO_ROOT / paths["promoted_report"]
    if not a0["passed_A0"]:
        value = {
            "schema": "myrec.c71.gate.v1",
            "candidate_id": "c71",
            "created_at": timestamp(),
            "execution_lock_sha256": lock_hash,
            "A0_passed": False,
            "target_labels_opened": False,
            "passed": False,
            "decision": "failed_A0_terminal",
            "source_A0": {"path": str(a0_path.relative_to(REPO_ROOT)), "sha256": sha256_file(a0_path)},
            "isolation": {"source_episode_labels_opened": False, "dev_test_qrels_opened": False},
        }
        atomic_json(target, value)
        print(target.relative_to(REPO_ROOT)); print(value["decision"])
        return
    selection_path = REPO_ROOT / paths["selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    structural, labels = open_target_labels(
        REPO_ROOT / paths["records_train"],
        selection["targets"],
        str(config["evaluation"]["label_field"]),
    )
    if candidate_key_sha256(structural) != selection["candidate_key_sha256"]:
        raise RuntimeError("C71 candidate hash changed immediately before evaluation")
    scores = load_scores(a0)
    request_ids = [row["request_id"] for row in selection["targets"]]
    item_ids = [row["candidate_ids"] for row in selection["targets"]]
    metrics = {
        name: ndcg_rows(request_ids, item_ids, values, labels)
        for name, values in scores.items()
        if name not in ("primary_correction", "wrong_correction")
    }
    evaluation = config["evaluation"]
    comparisons = compare(
        request_ids,
        metrics["primary_true"],
        {
            "base": metrics["base"],
            "positive_only": metrics["positive_only"],
            "uniform_slate": metrics["uniform_slate"],
            "semantic_history": metrics["semantic_history"],
            "primary_wrong": metrics["primary_wrong"],
        },
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    direction = bootstrap(
        clicked_direction(scores["primary_correction"], labels),
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]) + 20,
    )
    checks = {
        "primary_beats_base": comparison_pass(comparisons["base"], float(evaluation["primary_minus_base_min"])),
        "primary_beats_positive_only": comparison_pass(comparisons["positive_only"], float(evaluation["primary_minus_positive_only_min"])),
        "primary_beats_uniform_slate": comparison_pass(comparisons["uniform_slate"], float(evaluation["primary_minus_uniform_slate_min"])),
        "primary_beats_semantic_history": comparison_pass(comparisons["semantic_history"], float(evaluation["primary_minus_semantic_history_min"])),
        "true_beats_wrong": comparison_pass(comparisons["primary_wrong"], float(evaluation["true_minus_wrong_min"])),
        "clicked_direction": direction["percentile_95_ci"][0] > 0,
    }
    passed = all(checks.values())
    value = {
        "schema": "myrec.c71.gate.v1",
        "candidate_id": "c71",
        "created_at": timestamp(),
        "execution_lock_sha256": lock_hash,
        "proposal_lock_sha256": a0["proposal_lock_sha256"],
        "selection_sha256": sha256_file(selection_path),
        "A0_passed": True,
        "target_labels_opened_after_A0": True,
        "requests": len(request_ids),
        "mean_ndcg@10": {name: float(value.mean()) for name, value in metrics.items()},
        "comparisons": comparisons,
        "clicked_direction": direction,
        "checks": checks,
        "passed": passed,
        "decision": "authorize_c70_second_domain_acquisition" if passed else "failed_signal_terminal",
        "source_A0": {"path": str(a0_path.relative_to(REPO_ROOT)), "sha256": sha256_file(a0_path)},
        "isolation": {
            "source_episode_labels_opened": False,
            "dev_test_qrels_opened": False,
            "c70_architecture_implemented": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT)); print(sha256_file(target)); print(value["decision"])


if __name__ == "__main__":
    main()
