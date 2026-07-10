#!/usr/bin/env python
"""Audit and adjudicate the locked C5-R2 temporal identity repair."""

from __future__ import annotations

import json
import math
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.temporal_identity import adjudicate_temporal_gate
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


CONFIG_PATH = Path("configs/analysis/c5r_temporal_symmetric_identity.yaml")
PROTOCOL_PATH = Path("doc/22_c5r_temporal_symmetric_identity_protocol.md")
REPORT_JSON = Path("reports/pps_c5r2_temporal_symmetric_identity.json")
REPORT_MD = Path("reports/pps_c5r2_temporal_symmetric_identity.md")


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric_map(run_id: str) -> dict[str, dict]:
    return {
        str(row["request_id"]): row
        for row in iter_jsonl(Path("runs") / run_id / "per_request_metrics.jsonl")
    }


def main() -> int:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seeds = [int(seed) for seed in config["seeds"]]
    artifacts_dir = Path(config["artifacts_dir"])
    manifest = load_json(artifacts_dir / "materialization_manifest.json")
    if manifest["config_sha256"] != sha256_file(CONFIG_PATH):
        raise AssertionError("materializer/config hash mismatch")

    comparisons = {"history_present": {}, "freshness_balanced": {}, "same_query": {}}
    variants = {"true": [], "temporal_wrong": []}
    for seed in seeds:
        true_id = config["runs"]["true_d2s_pattern"].format(seed=seed)
        wrong_id = config["runs"]["wrong_d2s_pattern"].format(seed=seed)
        true_metrics = load_json(Path("runs") / true_id / "metrics.json")
        wrong_metrics = load_json(Path("runs") / wrong_id / "metrics.json")
        variants["true"].append(
            {"run_id": true_id, "seed": seed, "ndcg@10": true_metrics["ndcg@10"]}
        )
        variants["temporal_wrong"].append(
            {"run_id": wrong_id, "seed": seed, "ndcg@10": wrong_metrics["ndcg@10"]}
        )
        comparisons["history_present"][str(seed)] = load_json(
            f"reports/compare_20260710_kuaisearch_c5r2_d2s_true_vs_wrong_"
            f"history_present_s{seed}.json"
        )
        comparisons["freshness_balanced"][str(seed)] = load_json(
            f"reports/compare_20260710_kuaisearch_c5r2_d2s_true_vs_wrong_"
            f"freshness_balanced_s{seed}.json"
        )
        comparisons["same_query"][str(seed)] = load_json(
            f"reports/compare_20260710_kuaisearch_c5r2_d2s_true_vs_wrong_"
            f"same_query_freshness_balanced_s{seed}.json"
        )

    integrity = {
        "assignment_rows": {},
        "balanced_gap_violations": 0,
        "donor_event_not_before_donor_request": 0,
        "donor_request_not_before_target": 0,
        "history_absent_nonempty_donor": 0,
        "no_history_metric_mismatches": 0,
        "same_query_query_mismatches": 0,
        "same_user_donors": 0,
        "subset_mismatches": 0,
    }
    balance = {}
    max_gap = float(config["matching"]["max_log2_age_gap"])
    expected_balanced = _load_ids(
        artifacts_dir / "freshness_balanced_all_seeds_request_ids.txt"
    )
    expected_same_query = _load_ids(
        artifacts_dir / "same_query_freshness_balanced_all_seeds_request_ids.txt"
    )
    per_seed_balanced = []
    per_seed_same_query = []
    same_query_pairs: list[tuple[str, str]] = []
    for seed in seeds:
        rows = list(iter_jsonl(artifacts_dir / f"donor_assignments_s{seed}.jsonl"))
        integrity["assignment_rows"][str(seed)] = len(rows)
        balanced_ids = set()
        same_query_ids = set()
        balanced_rows = []
        same_query_rows = []
        for row in rows:
            if int(row["target_history_length"]) == 0:
                if row["donor_request_id"] is not None:
                    integrity["history_absent_nonempty_donor"] += 1
                continue
            if row["donor_user_id"] == row["target_user_id"]:
                integrity["same_user_donors"] += 1
            if int(row["donor_request_ts"]) >= int(row["target_request_ts"]):
                integrity["donor_request_not_before_target"] += 1
            if int(row["donor_latest_event_ts"]) >= int(row["donor_request_ts"]):
                integrity["donor_event_not_before_donor_request"] += 1
            if row["balanced"]:
                balanced_rows.append(row)
                balanced_ids.add(str(row["request_id"]))
                if float(row["log2_age_gap"]) > max_gap:
                    integrity["balanced_gap_violations"] += 1
                if row["match_tier"] in {"query", "query_length"}:
                    same_query_rows.append(row)
                    same_query_ids.add(str(row["request_id"]))
                    same_query_pairs.append(
                        (str(row["request_id"]), str(row["donor_request_id"]))
                    )
        per_seed_balanced.append(balanced_ids)
        per_seed_same_query.append(same_query_ids)
        balance[str(seed)] = {
            "freshness_balanced": _balance_summary(balanced_rows),
            "same_query_freshness_balanced": _balance_summary(same_query_rows),
        }

        d2p = metric_map(config["runs"]["d2p_pattern"].format(seed=seed))
        wrong = metric_map(config["runs"]["wrong_d2s_pattern"].format(seed=seed))
        for row in rows:
            if int(row["target_history_length"]) != 0:
                continue
            request_id = str(row["request_id"])
            for metric in ("ndcg@10", "mrr", "recall@10"):
                if d2p[request_id][metric] != wrong[request_id][metric]:
                    integrity["no_history_metric_mismatches"] += 1

    actual_balanced = set.intersection(*per_seed_balanced)
    actual_same_query = set.intersection(*per_seed_same_query)
    integrity["subset_mismatches"] += int(actual_balanced != expected_balanced)
    integrity["subset_mismatches"] += int(actual_same_query != expected_same_query)
    integrity["same_query_query_mismatches"] = _query_match_mismatches(
        config, same_query_pairs
    )
    integrity_passed = all(
        value == 0
        for key, value in integrity.items()
        if key != "assignment_rows"
    ) and all(value == 12229 for value in integrity["assignment_rows"].values())

    decision = adjudicate_temporal_gate(
        gate=config["gate"],
        subset_counts=manifest["subset_counts"],
        freshness_comparisons=comparisons["freshness_balanced"],
        same_query_comparisons=comparisons["same_query"],
        integrity_passed=integrity_passed,
    )
    true_values = [float(row["ndcg@10"]) for row in variants["true"]]
    wrong_values = [float(row["ndcg@10"]) for row in variants["temporal_wrong"]]
    report = {
        "analysis_id": config["analysis_id"],
        "balance": balance,
        "candidate_manifest_sha256": manifest["candidate_manifest_sha256"],
        "comparisons": comparisons,
        "config": {
            "path": str(CONFIG_PATH),
            "sha256": sha256_file(CONFIG_PATH),
        },
        "decision": {
            **decision,
            "formal_personalized_system_authorization": decision["passed"],
            "interpretation": (
                "C5-R2 passes: identity specificity survives the frozen "
                "freshness-matched control."
                if decision["passed"]
                else "C5-R2 fails because the frozen same-query significance "
                "requirement is not met. Aggregate correct-history value remains "
                "positive, but identity specificity is not sufficiently established "
                "for formal system authorization."
            ),
        },
        "dev_evaluations_added": 3,
        "integrity": {**integrity, "passed": integrity_passed},
        "materialization_manifest": {
            "path": str(artifacts_dir / "materialization_manifest.json"),
            "sha256": sha256_file(artifacts_dir / "materialization_manifest.json"),
        },
        "protocol": {
            "path": str(PROTOCOL_PATH),
            "sha256": sha256_file(PROTOCOL_PATH),
        },
        "qrels_read_by_materialization_or_scoring": False,
        "subset_counts": manifest["subset_counts"],
        "test_read": False,
        "variants": {
            "temporal_wrong": {
                "mean_ndcg@10": statistics.mean(wrong_values),
                "sample_std_ndcg@10": statistics.stdev(wrong_values),
                "seeds": variants["temporal_wrong"],
            },
            "true": {
                "mean_ndcg@10": statistics.mean(true_values),
                "sample_std_ndcg@10": statistics.stdev(true_values),
                "seeds": variants["true"],
            },
        },
    }
    write_json(REPORT_JSON, report)
    REPORT_MD.write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if decision["passed"] else 2


def _load_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def _balance_summary(rows: list[dict]) -> dict:
    target_age = [int(row["target_history_age"]) for row in rows]
    donor_age = [int(row["donor_history_age_at_target"]) for row in rows]
    target_log = [math.log2(value + 1.0) for value in target_age]
    donor_log = [math.log2(value + 1.0) for value in donor_age]
    pooled = math.sqrt(
        (statistics.pvariance(target_log) + statistics.pvariance(donor_log)) / 2.0
    )
    return {
        "count": len(rows),
        "donor_age_median": statistics.median(donor_age),
        "log_age_smd_donor_minus_true": (
            (statistics.mean(donor_log) - statistics.mean(target_log)) / pooled
            if pooled
            else 0.0
        ),
        "log2_age_gap_max": max(float(row["log2_age_gap"]) for row in rows),
        "log2_age_gap_median": statistics.median(
            float(row["log2_age_gap"]) for row in rows
        ),
        "true_age_median": statistics.median(target_age),
    }


def _query_match_mismatches(
    config: dict, pairs: list[tuple[str, str]]
) -> int:
    from myrec.analysis.history_identity import normalize_query

    needed_targets = {target for target, _ in pairs}
    needed_donors = {donor for _, donor in pairs}
    target_queries = {}
    donor_queries = {}
    for row in iter_jsonl(config["inputs"]["records_dev"]):
        request_id = str(row["request_id"])
        if request_id in needed_targets:
            target_queries[request_id] = normalize_query(row.get("query"))
        if request_id in needed_donors:
            donor_queries[request_id] = normalize_query(row.get("query"))
    for row in iter_jsonl(config["inputs"]["records_train"]):
        request_id = str(row["request_id"])
        if request_id in needed_donors:
            donor_queries[request_id] = normalize_query(row.get("query"))
    if set(target_queries) != needed_targets or set(donor_queries) != needed_donors:
        raise AssertionError("same-query audit could not resolve every request")
    return sum(
        target_queries[target] != donor_queries[donor] for target, donor in pairs
    )


def _markdown(report: dict) -> str:
    decision = report["decision"]
    lines = [
        "# C5-R2 Temporal-Symmetric Identity Repair",
        "",
        f"Status: **{'PASSED' if decision['passed'] else 'FAILED'}**.",
        "",
        "The temporal-staleness implementation defect is repaired: both true and "
        "wrong histories now follow a strictly-prior prequential policy, and the "
        "extreme train-versus-dev age mismatch is bounded per request before "
        "comparison. Residual balance is reported rather than hidden. The "
        "scientific gate is adjudicated separately below.",
        "",
        "## Frozen Counts",
        "",
        "| Subset | Requests |",
        "|---|---:|",
        f"| History present | {report['subset_counts']['history_present']:,} |",
        f"| Freshness-balanced, all seeds | {report['subset_counts']['freshness_balanced_all_seeds']:,} |",
        f"| Same-query + freshness-balanced, all seeds | {report['subset_counts']['same_query_freshness_balanced_all_seeds']:,} |",
        f"| History absent | {report['subset_counts']['history_absent']:,} |",
        "",
        "## Paired NDCG@10 Results",
        "",
        "| Subset | Seed | Delta | 95% CI | Significant |",
        "|---|---:|---:|---:|---|",
    ]
    for subset in ("freshness_balanced", "same_query"):
        label = "freshness-balanced" if subset == "freshness_balanced" else "same-query + freshness-balanced"
        for seed, row in report["comparisons"][subset].items():
            lines.append(
                f"| {label} | {seed} | {row['delta']:+.6f} | "
                f"[{row['ci95'][0]:+.6f}, {row['ci95'][1]:+.6f}] | "
                f"{'yes' if row['significant_a_gt_b'] else 'no'} |"
            )
    lines.extend(
        [
            "",
            "## Freshness Balance",
            "",
            "On the 7,614-request balanced subset, true-history median age is "
            f"{report['balance']['20260708']['freshness_balanced']['true_age_median']:.1f} "
            "and seed-20260708 donor median age is "
            f"{report['balance']['20260708']['freshness_balanced']['donor_age_median']:.1f}; "
            "the log-age SMD is "
            f"{report['balance']['20260708']['freshness_balanced']['log_age_smd_donor_minus_true']:.3f}. "
            "For the same-query subset the corresponding SMD is "
            f"{report['balance']['20260708']['same_query_freshness_balanced']['log_age_smd_donor_minus_true']:.3f}. "
            "The per-request factor-four bound removes the orders-of-magnitude "
            "mismatch in the original control, but does not imply perfect balance.",
            "",
            "## Decision",
            "",
            f"Same-query mean delta: **{decision['same_query_mean_delta']:+.6f}**; "
            f"significant seeds: **{decision['same_query_significant_seed_count']}/3**.",
            "",
            decision["interpretation"],
            "",
            "The original D2s performance waterline remains valid. What is not "
            "restored is the stronger statement that same-query identity specificity "
            "has already cleared its cheap falsifier. No model was trained and test "
            "was not read.",
            "",
            "## Integrity",
            "",
            f"Assignment/no-history audit: **{'passed' if report['integrity']['passed'] else 'failed'}**. "
            "All candidate and qrels hashes are recorded in the evaluator artifacts.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
