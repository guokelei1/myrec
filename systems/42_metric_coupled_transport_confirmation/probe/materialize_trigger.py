"""Materialize the post-terminal C41 control evidence that triggers C42."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C41_ROOT = REPO_ROOT / "systems/41_semantic_carrier_routing_transformer"
sys.path.insert(0, str(C41_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.run_train_gate import _average_rows, _unflatten, ndcg_rows  # noqa: E402
from train.store import FrozenStore, open_role_labels, read_json, sha256_file, write_json  # noqa: E402


OUTPUT = REPO_ROOT / "reports/pps_c41_coupled_control_diagnostic.json"


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    config = yaml.safe_load(
        (C41_ROOT / "configs/train_gate.yaml").read_text(encoding="utf-8")
    )
    report_path = Path(config["paths"]["artifact_root"]) / "train_gate_report.json"
    report = read_json(report_path)
    if report.get("status") != "failed_A1_terminal" or not all(report["A0"]["checks"].values()):
        raise RuntimeError("C41 terminal report is not a valid trigger source")
    store = FrozenStore(config)
    indices = store.role_indices("internal_A")
    labels = open_role_labels(
        records_train_path=config["paths"]["records_train"],
        records_train_sha256=config["integrity"]["records_train_sha256"],
        selection_path=config["paths"]["selection"],
        selection_sha256=config["paths"]["selection_sha256"],
        store=store,
        role="internal_A",
    )
    label_rows = [labels.row(index, store.candidate_count(index)) for index in indices]
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    seeds = [int(value) for value in config["training"]["seeds"]]
    names = (
        "coupled_content_true",
        "coupled_content_wrong",
        "coupled_content_correction",
        "c38_unprojected_true",
        "semantic_routing_true",
        "base",
    )
    rows = {}
    ndcg = {}
    score_hashes = {}
    for seed in seeds:
        seed_report = read_json(
            Path(config["paths"]["artifact_root"]) / f"seed_{seed}_report.json"
        )
        score_path = Path(seed_report["score_artifact"]["path"])
        if sha256_file(score_path) != seed_report["score_artifact"]["sha256"]:
            raise RuntimeError("C41 score artifact changed")
        score_hashes[str(seed)] = seed_report["score_artifact"]["sha256"]
        with np.load(score_path, allow_pickle=False) as values:
            offsets = np.asarray(values["offsets"], dtype=np.int64)
            rows[seed] = {name: _unflatten(offsets, values[name]) for name in names}
        ndcg[seed] = {
            name: ndcg_rows(request_ids, item_ids, rows[seed][name], label_rows)
            for name in (
                "coupled_content_true",
                "coupled_content_wrong",
                "c38_unprojected_true",
                "semantic_routing_true",
                "base",
            )
        }
    averaged = {
        name: np.mean(np.stack([ndcg[seed][name] for seed in seeds]), axis=0)
        for name in ndcg[seeds[0]]
    }
    comparisons = compare(
        request_ids,
        averaged["coupled_content_true"],
        {
            "c38_unprojected": averaged["c38_unprojected_true"],
            "semantic_routing": averaged["semantic_routing_true"],
            "base": averaged["base"],
            "wrong_history": averaged["coupled_content_wrong"],
        },
        samples=10000,
        seed=20262301,
        folds=3,
    )
    correction = _average_rows(
        [rows[seed]["coupled_content_correction"] for seed in seeds]
    )
    direction = bootstrap(
        clicked_direction(correction, label_rows), samples=10000, seed=20262401
    )
    per_seed = {
        str(seed): {
            "coupled_minus_c38": float(
                (ndcg[seed]["coupled_content_true"] - ndcg[seed]["c38_unprojected_true"]).mean()
            ),
            "coupled_true_minus_wrong": float(
                (ndcg[seed]["coupled_content_true"] - ndcg[seed]["coupled_content_wrong"]).mean()
            ),
        }
        for seed in seeds
    }
    checks = {
        "coupled_minus_c38_effect": comparisons["c38_unprojected"]["mean"] >= 0.003,
        "coupled_minus_c38_ci": comparisons["c38_unprojected"]["percentile_95_ci"][0] > 0,
        "coupled_minus_c38_all_seeds": all(row["coupled_minus_c38"] > 0 for row in per_seed.values()),
        "coupled_minus_c38_all_folds": all(
            row["mean_difference"] > 0 for row in comparisons["c38_unprojected"]["hash_folds"]
        ),
        "coupled_true_over_wrong_ci": comparisons["wrong_history"]["percentile_95_ci"][0] > 0,
        "coupled_true_over_wrong_all_seeds": all(
            row["coupled_true_minus_wrong"] > 0 for row in per_seed.values()
        ),
        "coupled_true_over_wrong_all_folds": all(
            row["mean_difference"] > 0 for row in comparisons["wrong_history"]["hash_folds"]
        ),
        "coupled_clicked_direction_ci": direction["percentile_95_ci"][0] > 0,
        "c42_cohort_unread": True,
        "dev_test_unread": True,
    }
    value = {
        "candidate_id": "c42_trigger",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_c42_trigger" if all(checks.values()) else "failed_terminal",
        "checks": checks,
        "source_report": str(report_path),
        "source_report_sha256": sha256_file(report_path),
        "source_score_sha256": score_hashes,
        "comparisons": comparisons,
        "clicked_direction": direction,
        "per_seed": per_seed,
        "seed_averaged_ndcg10": {
            name: float(value.mean()) for name, value in averaged.items()
        },
        "c42_cohort_read": False,
        "dev_test_read": False,
    }
    write_json(OUTPUT, value)
    print(json.dumps({"status": value["status"], "checks": checks}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
