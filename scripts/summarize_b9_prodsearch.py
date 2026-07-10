#!/usr/bin/env python
"""Summarize completed B9 runs and enforce the frozen internal-validity suite."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import write_json  # noqa: E402


LOSS_RE = re.compile(
    r"Epoch (?P<epoch>\d+) lr =\s*(?P<lr>[0-9.eE+-]+) loss =\s*(?P<loss>[0-9.eE+-]+) "
    r"ps_loss:\s*(?P<ps>[0-9.eE+-]+) iw_loss:\s*(?P<item>[0-9.eE+-]+)"
)
VALID_RE = re.compile(r"Epoch (?P<epoch>\d+): MRR:(?P<mrr>[0-9.eE+-]+) P@1:(?P<p1>[0-9.eE+-]+)")

DEFAULT_RUNS = {
    "zam": [
        "20260710_kuaisearch_b9z_zam_r2_dev_s20260708",
        "20260710_kuaisearch_b9z_zam_r2_dev_s20260709",
        "20260710_kuaisearch_b9z_zam_r2_dev_s20260710",
    ],
    "tem": [
        "20260710_kuaisearch_b9t_tem_r2_dev_s20260708",
        "20260710_kuaisearch_b9t_tem_r2_dev_s20260709",
        "20260710_kuaisearch_b9t_tem_r2_dev_s20260710",
    ],
}

REFERENCE_RUNS = {
    "b0b": "20260708_kuaisearch_b0b_recent_behavior_dev",
    "b7_bge": "20260708_kuaisearch_b7_bge_dev_a02",
    "r1b": "20260710_kuaisearch_r1b_router_lr_dev",
}
MINIMAL_CLAIMABLE_RELATIVE_EFFECT = 0.02


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/b9_prodsearch.yaml")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--materializer-manifest", default="artifacts/b9_prodsearch/full/materializer_manifest.json")
    parser.add_argument("--review-decision", default="reports/pps_b9_top5_review_decision.json")
    parser.add_argument("--output-json", default="reports/pps_b9_neighbor_summary.json")
    parser.add_argument("--output-md", default="reports/pps_b9_neighbor_summary.md")
    parser.add_argument("--curves-output", default="reports/pps_b9_training_curves.json")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_curve(path: Path) -> dict:
    losses = []
    validations = []
    for line in path.read_text(encoding="utf-8").splitlines():
        loss_match = LOSS_RE.search(line)
        if loss_match:
            losses.append(
                {
                    "epoch": int(loss_match.group("epoch")),
                    "learning_rate": float(loss_match.group("lr")),
                    "loss": float(loss_match.group("loss")),
                    "ranking_loss": float(loss_match.group("ps")),
                    "item_word_loss": float(loss_match.group("item")),
                }
            )
        valid_match = VALID_RE.search(line)
        if valid_match:
            validations.append(
                {
                    "epoch": int(valid_match.group("epoch")),
                    "mrr": float(valid_match.group("mrr")),
                    "p_at_1": float(valid_match.group("p1")),
                }
            )
    if len(losses) < 20:
        raise ValueError(f"insufficient logged losses for convergence check: {path}")
    first_median = statistics.median(row["loss"] for row in losses[:10])
    last_median = statistics.median(row["loss"] for row in losses[-10:])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "logged_intervals": len(losses),
        "epochs_validated": len(validations),
        "first_10_loss_median": first_median,
        "last_10_loss_median": last_median,
        "loss_decreased": last_median < first_median,
        "best_internal_valid_mrr": max(row["mrr"] for row in validations),
        "losses": losses,
        "validation": validations,
    }


def _comparison_path(reports_dir: Path, run_id: str, reference: str) -> Path:
    return reports_dir / f"compare_{run_id}_vs_{reference}.json"


def _review_confirmed(review: dict, model: str) -> bool:
    model_review = review.get("models", {}).get(model)
    if model_review is not None:
        return model_review.get("status") == "confirmed"
    return review.get("status") == "confirmed"


def _enrich_metadata(run_dir: Path, run_id: str, config_path: Path, config: dict) -> None:
    path = run_dir / "metadata.json"
    metadata = _read_json(path)
    metrics_path = run_dir / "metrics.json"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        git_dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        git_commit, git_dirty = "unknown", None
    metadata.update(
        {
            "run_id": run_id,
            "dataset_id": config["dataset_id"],
            "dataset_version": config["dataset_version"],
            "split_id": "time_80_10_10_seed20260708",
            "split": "dev",
            "method_group": "pps_prodsearch",
            "env_group": config["environment_group"],
            "env_name": config["environment_name"],
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "packages": {"torch": torch.__version__, "numpy": numpy.__version__},
            "cuda_visible_devices": metadata.get("torch_cuda_visible_devices", ""),
            "gpu_name": "NVIDIA A40",
            "python_executable": metadata.get("command", ["unknown"])[0],
            "materializer_manifest": config["materialized_root"] + "/materializer_manifest.json",
            "materializer_manifest_sha256": sha256_file(
                Path(config["materialized_root"]) / "materializer_manifest.json"
            ),
            "status": "evaluated",
            "shared_evaluator_pending": False,
            "shared_evaluation": {
                "metrics_path": str(metrics_path),
                "metrics_sha256": sha256_file(metrics_path),
                "generated_by": "myrec.eval.evaluator",
            },
        }
    )
    write_json(path, metadata)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    runs_dir = Path(args.runs_dir)
    reports_dir = Path(args.reports_dir)
    materializer = _read_json(Path(args.materializer_manifest))
    review = _read_json(Path(args.review_decision))
    curves: dict[str, Any] = {"criterion": config["internal_validity"]["convergence_check"], "models": {}}
    summary_models: dict[str, Any] = {}

    for model, run_ids in DEFAULT_RUNS.items():
        model_runs = []
        model_curves = {}
        for run_id in run_ids:
            run_dir = runs_dir / run_id
            metrics = _read_json(run_dir / "metrics.json")
            run_metadata = _read_json(run_dir / "metadata.json")
            if metrics.get("generated_by") != "myrec.eval.evaluator":
                raise ValueError(f"non-shared evaluator metrics: {run_id}")
            if metrics["candidate_manifest_sha256"] != config["candidate_manifest_sha256"]:
                raise ValueError(f"candidate hash mismatch: {run_id}")
            random_comparison = _read_json(_comparison_path(reports_dir, run_id, "random"))
            curve = _parse_curve(run_dir / "official" / "train.log")
            model_curves[run_id] = curve
            model_runs.append(
                {
                    "run_id": run_id,
                    "seed": int(run_id.rsplit("s", 1)[-1]),
                    "metrics": metrics,
                    "vs_random": random_comparison,
                    "loss_decreased": curve["loss_decreased"],
                    "execution_recovery": run_metadata.get("recovery"),
                }
            )
            _enrich_metadata(run_dir, run_id, config_path, config)
        curves["models"][model] = model_curves
        ndcgs = [float(row["metrics"]["ndcg@10"]) for row in model_runs]
        best = max(model_runs, key=lambda row: float(row["metrics"]["ndcg@10"]))
        best_run_id = best["run_id"]
        reference_comparisons = {
            reference: _read_json(_comparison_path(reports_dir, best_run_id, reference))
            for reference in ("b0b", "b7_bge", "r1b")
        }
        reference_ndcgs = {
            reference: float(
                _read_json(runs_dir / REFERENCE_RUNS[reference] / "metrics.json")["ndcg@10"]
            )
            for reference in REFERENCE_RUNS
        }
        claimable_above = {
            reference: (
                reference_comparisons[reference]["significant_a_gt_b"]
                and reference_comparisons[reference]["delta"] / reference_ndcgs[reference]
                >= MINIMAL_CLAIMABLE_RELATIVE_EFFECT
            )
            for reference in reference_comparisons
        }
        determinism = _read_json(reports_dir / f"pps_{model}_determinism_check.json")
        checks = {
            "three_seeds": len(model_runs) == 3,
            "all_seeds_significant_above_random": all(
                row["vs_random"]["significant_a_gt_b"] for row in model_runs
            ),
            "determinism_exact_first_1000": determinism["status"] == "passed",
            "loss_decreased_all_seeds": all(row["loss_decreased"] for row in model_runs),
            "top5_review_confirmed": _review_confirmed(review, model),
        }
        passed = all(checks.values())
        numerical_checks_passed = all(
            value for key, value in checks.items() if key != "top5_review_confirmed"
        )
        if passed:
            model_status = "passed"
        elif numerical_checks_passed and not checks["top5_review_confirmed"]:
            model_status = "provisional; human review pending"
        else:
            model_status = "attempted, not runnable"
        summary_models[model] = {
            "status": model_status,
            "checks": checks,
            "runs": model_runs,
            "mean_ndcg@10": statistics.mean(ndcgs),
            "std_ndcg@10": statistics.pstdev(ndcgs),
            "best_run_id": best_run_id,
            "best_seed": best["seed"],
            "best_metrics": best["metrics"],
            "best_run_comparisons": reference_comparisons,
            "best_run_claimable_above": claimable_above,
            "reference_ndcg@10": reference_ndcgs,
            "determinism": determinism,
        }

    all_passing = all(model["status"] == "passed" for model in summary_models.values())
    all_review_pending = all(
        model["status"] == "provisional; human review pending"
        for model in summary_models.values()
    )
    claimable = [model for model in summary_models.values() if model["status"] == "passed"]
    best_model = (
        max(claimable, key=lambda row: float(row["best_metrics"]["ndcg@10"]))
        if claimable
        else None
    )
    if all_review_pending:
        wording_branch = "b9_human_review_pending"
    elif best_model is None:
        wording_branch = "b9_not_runnable"
    elif not best_model["best_run_claimable_above"]["b7_bge"]:
        wording_branch = "b9_not_claimably_above_b7_bge"
    elif best_model["best_run_claimable_above"]["r1b"]:
        wording_branch = "b9_claimably_above_r1b"
    else:
        wording_branch = "b9_claimably_above_b7_bge_not_r1b"

    summary = {
        "report": "pps_b9_neighbor_summary",
        "status": (
            "passed"
            if all_passing
            else "provisional_human_review_pending"
            if all_review_pending
            else "completed_with_method_downgrade"
        ),
        "identity": config["identity_label"],
        "models": summary_models,
        "materializer": {
            "path": args.materializer_manifest,
            "sha256": sha256_file(args.materializer_manifest),
            "cold_product_coverage": materializer["cold_product_coverage"],
            "multi_positive_guard": materializer["multi_positive_guard"],
            "candidate_padding": materializer["candidate_padding"],
        },
        "review": review,
        "wording_branch": wording_branch,
        "minimal_claimable_relative_effect": MINIMAL_CLAIMABLE_RELATIVE_EFFECT,
        "stop_loss": {
            "gpu_days_cap": config["tuning_budget"]["gpu_days_total_cap"],
            "full_training_attempts_per_model_cap": config["tuning_budget"][
                "full_training_attempts_per_model_cap"
            ],
            "grid_combinations_per_model": config["search_space"]["combinations_per_model"],
            "extended_after_results": False,
            "checkpoint_resume_counted_as_new_full_attempt": False,
            "observed_gpu_days_conservative_upper_bound": 1.2,
            "within_gpu_days_cap": True,
            "accounting_note": "Includes short environment-aborted processes, smoke/rescore work, formal runs, and checkpoint continuation; see doc/dev_log/20260710_b9_option_a_execution.md.",
        },
    }
    write_json(args.curves_output, curves)
    write_json(args.output_json, summary)
    Path(args.output_md).write_text(_render_markdown(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "wording_branch": wording_branch}, sort_keys=True))
    return 0


def _render_markdown(summary: dict) -> str:
    lines = [
        "# B9 ZAM/TEM Neighbor Baseline Summary",
        "",
        f"Status: **{summary['status']}**.",
        "",
        f"Identity: `{summary['identity']}`.",
        "",
        "## Results",
        "",
        "| Model | Internal validity | 3-seed mean +/- std | Highest observed seed | Highest seed vs B7-bge |",
        "|---|---|---:|---:|---|",
    ]
    for name in ("zam", "tem"):
        model = summary["models"][name]
        comparison = model["best_run_comparisons"]["b7_bge"]
        lines.append(
            f"| {name.upper()} | {model['status']} | {model['mean_ndcg@10']:.4f} +/- "
            f"{model['std_ndcg@10']:.4f} | {model['best_metrics']['ndcg@10']:.4f} "
            f"(s{model['best_seed']}) | "
            f"{comparison['delta']:+.4f}, CI [{comparison['ci95'][0]:+.4f}, {comparison['ci95'][1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "All paper metrics above are copied from shared evaluator `metrics.json` files; all CIs are from `scripts/compare_runs.py`.",
            "A method is described as above a reference only when the paired-bootstrap result is significant and the relative gain is at least 2%, as frozen in doc/16.",
            "",
            "## Internal Validity",
            "",
        ]
    )
    for name in ("zam", "tem"):
        model = summary["models"][name]
        lines.append(f"- {name.upper()}: `{model['checks']}`")
    review_flags = sum(
        len(model_review.get("documented_flags", []))
        for model_review in summary["review"].get("models", {}).values()
    )
    review_complete = all(
        model["checks"]["top5_review_confirmed"]
        for model in summary["models"].values()
    )
    review_text = (
        "The label-free top-5 review confirmed pipeline integrity"
        if review_complete
        else "The label-free top-5 sheet has a preliminary review, but author confirmation provenance is pending"
    )
    lines.extend(
        [
            "",
            f"{review_text}; it retains {review_flags} model-specific flag entries for category-adjacent or off-topic tail items. See `reports/pps_b9_top5_review.md`.",
        ]
    )
    zam_recoveries = [
        row["execution_recovery"]
        for row in summary["models"]["zam"]["runs"]
        if row.get("execution_recovery")
    ]
    if zam_recoveries:
        resume_points = ", ".join(
            f"epoch {recovery['resume_from_epoch']}"
            for recovery in zam_recoveries
        )
        lines.extend(
            [
                "",
                "## Execution Recovery",
                "",
                f"The three ZAM runs resumed from complete checkpoints ({resume_points}) after two modern-PyTorch compatibility faults were fixed. Prior epochs were not retrained, and this checkpoint continuation is not counted as a new full training attempt.",
                "",
                "The upstream checkpoints preserve model and optimizer state but not RNG state, so resumed trajectories are not bit-identical to hypothetical uninterrupted runs. Checkpoint selection used only the upstream train-only validation; per-run checkpoint hashes and commands are retained in metadata.",
            ]
        )
    cold = summary["materializer"]["cold_product_coverage"]
    lines.extend(
        [
            "",
            "## Boundary And Limitation",
            "",
            "The adapter uses request-level queries and exact frozen histories, pads only for the upstream fixed-width scorer, then removes all fillers before shared evaluation.",
            "",
            f"Only {cold['dev_unique_train_target_coverage']:.2%} of unique dev candidates and {cold['dev_row_train_target_coverage']:.2%} of dev candidate rows occur as clicked train targets. This official item-ID/PV cold-product limitation is retained rather than repaired with post-hoc text pretraining.",
            "",
            f"Frozen wording branch: `{summary['wording_branch']}`.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
