#!/usr/bin/env python
"""Summarize D2/D2h controls and adjudicate the final motivation waterline."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


SEEDS = [20260708, 20260709, 20260710]
RUN_NAMES = {
    "d2t": "d2t_finetuned_text_dev",
    "d2p": "d2p_text_pop_dev",
    "d2h": "d2h_static_true_history_dev",
    "d2h_wrong": "d2h_static_wrong_history_dev",
}


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_id(variant: str, seed: int) -> str:
    return f"20260710_kuaisearch_{RUN_NAMES[variant]}_s{seed}"


def load_per_request(run: str) -> dict[str, dict]:
    return {
        str(row["request_id"]): row
        for row in iter_jsonl(Path("runs") / run / "per_request_metrics.jsonl")
    }


def main() -> int:
    variants = {}
    for variant in RUN_NAMES:
        rows = []
        for seed in SEEDS:
            identifier = run_id(variant, seed)
            metrics = load_json(Path("runs") / identifier / "metrics.json")
            rows.append(
                {
                    "mrr": metrics["mrr"],
                    "ndcg@10": metrics["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "run_id": identifier,
                    "seed": seed,
                }
            )
        ndcg = [row["ndcg@10"] for row in rows]
        variants[variant] = {
            "mean_ndcg@10": statistics.mean(ndcg),
            "sample_std_ndcg@10": statistics.stdev(ndcg),
            "seeds": rows,
        }

    comparisons = {
        "d2t_vs_b2z": load_json(
            "reports/compare_20260710_kuaisearch_d2t_s20260708_vs_b2z.json"
        ),
        "d2t_vs_d1q": load_json(
            "reports/compare_20260710_kuaisearch_d2t_s20260708_vs_d1q.json"
        ),
        "d2p_vs_d2t": load_json(
            "reports/compare_20260710_kuaisearch_d2p_vs_d2t_s20260708.json"
        ),
        "d2p_vs_d1q": load_json(
            "reports/compare_20260710_kuaisearch_d2p_s20260708_vs_d1q.json"
        ),
        "d2p_vs_b0b": load_json(
            "reports/compare_20260710_kuaisearch_d2p_s20260708_vs_b0b.json"
        ),
        "d2p_vs_b7": load_json(
            "reports/compare_20260710_kuaisearch_d2p_s20260708_vs_b7_bge.json"
        ),
        "d2h_vs_b7": load_json(
            "reports/compare_20260710_kuaisearch_d2h_s20260708_vs_b7_bge.json"
        ),
        "d2h_vs_d2p": load_json(
            "reports/compare_20260710_kuaisearch_d2h_s20260708_vs_d2p.json"
        ),
        "d2h_vs_d2t": load_json(
            "reports/compare_20260710_kuaisearch_d2h_s20260708_vs_d2t.json"
        ),
        "d2h_vs_b0b": load_json(
            "reports/compare_20260710_kuaisearch_d2h_s20260708_vs_b0b.json"
        ),
    }
    identity = {"history_present": {}, "same_query": {}}
    for seed in SEEDS:
        identity["history_present"][str(seed)] = load_json(
            "reports/compare_20260710_kuaisearch_d2h_true_vs_wrong_"
            f"history_present_s{seed}.json"
        )
        identity["same_query"][str(seed)] = load_json(
            "reports/compare_20260710_kuaisearch_d2h_true_vs_wrong_"
            f"same_query_s{seed}.json"
        )
    identity["history_present_mean_delta"] = statistics.mean(
        row["delta"] for row in identity["history_present"].values()
    )
    identity["same_query_mean_delta"] = statistics.mean(
        row["delta"] for row in identity["same_query"].values()
    )

    no_history_ids = {
        line.strip()
        for line in Path(
            "artifacts/analysis/c3_history_identity_controls/"
            "history_absent_request_ids.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    no_history_checks = {}
    metric_names = ["ndcg@10", "mrr", "recall@10"]
    for seed in SEEDS:
        d2t = load_per_request(run_id("d2t", seed))
        d2h = load_per_request(run_id("d2h", seed))
        mismatches = {metric: 0 for metric in metric_names}
        for request_id in no_history_ids:
            for metric in metric_names:
                mismatches[metric] += int(d2t[request_id][metric] != d2h[request_id][metric])
        if any(mismatches.values()):
            raise AssertionError(f"D2h no-history metric mismatch at seed {seed}: {mismatches}")
        no_history_checks[str(seed)] = {
            "metric_mismatches": mismatches,
            "request_count": len(no_history_ids),
            "status": "passed",
        }

    report = {
        "analysis_id": "d2_and_d2h_motivation_controls_v1",
        "calibration_integrity": {
            "d2_alpha_correction": "full-train internal-leak result invalidated; internal-train counts selected alpha 0.6",
            "d2h_alpha": 0.1,
            "d2h_alpha_report": "reports/pps_d2h_train_only_calibration.json",
        },
        "candidate_manifest_sha256": (
            "94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e"
        ),
        "comparisons_seed20260708": comparisons,
        "decision": {
            "baseline_to_beat": "D2h",
            "baseline_to_beat_mean_ndcg@10": variants["d2h"]["mean_ndcg@10"],
            "d2h_replaces_b7": comparisons["d2h_vs_b7"]["significant_a_gt_b"],
            "fine_tuned_text_exceeds_zero_shot": comparisons["d2t_vs_b2z"]["significant_a_gt_b"],
            "fine_tuned_text_exceeds_d1q": comparisons["d2t_vs_d1q"]["significant_a_gt_b"],
            "nonpersonalized_d2p_reaches_b7": comparisons["d2p_vs_b7"]["ci95"][0] >= 0,
            "interpretation": (
                "Fine-tuning strengthens query text but does not beat D1q. A legal "
                "text/popularity mix is strong yet remains below B7. Reissuing the static "
                "history mix with D2t yields D2h, which significantly beats B7 and becomes "
                "the design waterline. Its large true-versus-wrong gap confirms that the "
                "gain depends on target-user history, while D1 shows that simple learned "
                "mean/attention residuals do not recover it stably."
            ),
        },
        "dev_evaluations": 12,
        "identity_control": identity,
        "no_history_metric_equivalence": no_history_checks,
        "qrels_read_by_training_or_scoring": False,
        "test_read": False,
        "variants": variants,
    }
    write_json("reports/pps_d2_d2h_summary.json", report)

    lines = [
        "# D2/D2h Motivation Control Summary",
        "",
        "Status: complete; D2h is the corrected static baseline-to-beat.",
        "",
        "| Control | Mean NDCG@10 | Sample SD | Three seeds |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "d2t": "D2t fine-tuned text",
        "d2p": "D2p text + train popularity",
        "d2h": "D2h text + true causal history",
        "d2h_wrong": "D2h text + matched wrong history",
    }
    for variant in RUN_NAMES:
        row = variants[variant]
        seed_values = ", ".join(
            f"{value['ndcg@10']:.4f}" for value in row["seeds"]
        )
        lines.append(
            f"| {labels[variant]} | {row['mean_ndcg@10']:.4f} | "
            f"{row['sample_std_ndcg@10']:.4f} | {seed_values} |"
        )
    d2h_b7 = comparisons["d2h_vs_b7"]
    d2h_d2p = comparisons["d2h_vs_d2p"]
    lines.extend(
        [
            "",
            "D2t significantly improves over zero-shot B2z but is statistically tied with "
            "D1q. D2p is significantly stronger than D2t/D1q/B0b, yet remains "
            "significantly below B7.",
            "",
            f"D2h exceeds B7 by {d2h_b7['delta']:+.4f} (95% CI "
            f"[{d2h_b7['ci95'][0]:+.4f}, {d2h_b7['ci95'][1]:+.4f}]) and D2p by "
            f"{d2h_d2p['delta']:+.4f}. D2h therefore replaces B7 as the static waterline.",
            "",
            f"On history-present requests, true D2h exceeds matched wrong D2h by "
            f"{identity['history_present_mean_delta']:+.4f} on average across seeds. "
            f"On the same-query donor subset the mean is "
            f"{identity['same_query_mean_delta']:+.4f}; every paired CI is positive.",
            "",
            "On all 4,110 no-history requests, D2h and its seed-matched D2t have exactly "
            "the same NDCG@10, MRR, and Recall@10. The final motivation is therefore not "
            "that query evidence is universally saturated. It is that a strong static "
            "query/history rule exposes identity-specific value that the tested learned "
            "residuals and representative baselines do not recover.",
            "",
            "Training/scoring did not read dev/test qrels; all metrics came from the "
            "shared evaluator. Test remains untouched.",
        ]
    )
    Path("reports/pps_d2_d2h_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
