#!/usr/bin/env python
"""Summarize all fixed D1 supervised motivation diagnostics."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


SEEDS = [20260708, 20260709, 20260710]
RUN_NAMES = {
    "d1q": "d1q_supervised_query_dev",
    "d1m": "d1m_mean_history_residual_dev",
    "d1a": "d1a_query_attn_residual_dev",
    "d1a_wrong_history": "d1a_wrong_history_dev",
}


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_id(variant: str, seed: int) -> str:
    return f"20260710_kuaisearch_{RUN_NAMES[variant]}_s{seed}"


def comparison(name: str) -> dict:
    return load_json(Path("reports") / name)


def main() -> int:
    variants = {}
    for variant in RUN_NAMES:
        seed_rows = []
        for seed in SEEDS:
            identifier = run_id(variant, seed)
            metrics = load_json(Path("runs") / identifier / "metrics.json")
            seed_rows.append(
                {
                    "mrr": metrics["mrr"],
                    "ndcg@10": metrics["ndcg@10"],
                    "recall@10": metrics["recall@10"],
                    "run_id": identifier,
                    "seed": seed,
                }
            )
        values = [row["ndcg@10"] for row in seed_rows]
        variants[variant] = {
            "mean_ndcg@10": statistics.mean(values),
            "sample_std_ndcg@10": statistics.stdev(values),
            "seeds": seed_rows,
        }

    residual_directions = {}
    for residual in ("d1m", "d1a"):
        residual_directions[residual] = [
            variants[residual]["seeds"][index]["ndcg@10"]
            - variants["d1q"]["seeds"][index]["ndcg@10"]
            for index in range(len(SEEDS))
        ]
    attention_mean_directions = [
        variants["d1a"]["seeds"][index]["ndcg@10"]
        - variants["d1m"]["seeds"][index]["ndcg@10"]
        for index in range(len(SEEDS))
    ]
    true_wrong_directions = [
        variants["d1a"]["seeds"][index]["ndcg@10"]
        - variants["d1a_wrong_history"]["seeds"][index]["ndcg@10"]
        for index in range(len(SEEDS))
    ]

    comparisons = {
        "d1q_vs_b0a_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1q_s20260708_vs_b0a.json"
        ),
        "d1q_vs_b0b_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1q_s20260708_vs_b0b.json"
        ),
        "d1q_vs_b2z_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1q_s20260708_vs_b2z.json"
        ),
        "d1q_vs_b7_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1q_s20260708_vs_b7_bge.json"
        ),
        "d1m_vs_d1q_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1m_vs_d1q_s20260708.json"
        ),
        "d1a_vs_d1q_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1a_vs_d1q_s20260708.json"
        ),
        "d1a_vs_d1m_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1a_vs_d1m_s20260708.json"
        ),
        "d1a_vs_d1q_history_present_seed20260708": comparison(
            "compare_20260710_kuaisearch_d1a_vs_d1q_history_present_s20260708.json"
        ),
        "d1a_true_vs_wrong_history_present": {
            str(seed): comparison(
                "compare_20260710_kuaisearch_d1a_true_vs_wrong_"
                f"history_present_s{seed}.json"
            )
            for seed in SEEDS
        },
    }

    report = {
        "analysis_id": "supervised_motivation_diagnostics_v1",
        "calibration_report": "reports/pps_supervised_diagnostics_calibration.md",
        "candidate_manifest_sha256": (
            "94eb667000e0d0f389d0a2a4d4730683b71c129043edbfcf627590376e9c123e"
        ),
        "comparisons": comparisons,
        "decision": {
            "query_only_weakness_is_not_only_zero_shot": "supported for the tested supervised frozen-embedding adapter",
            "stable_history_residual_gain": "failed",
            "query_attention_above_mean_history": "failed",
            "wrong_history_identity_in_d1a": "visible at the preselected seed but not stable across seeds",
            "interpretation": (
                "D1q significantly improves over zero-shot B2z but remains significantly "
                "below B7 and statistically tied with B0b. D1m and D1a do not deliver a "
                "stable seed-matched gain over D1q. The matched wrong-history effect is "
                "small and seed-unstable in D1a, so train-fitted history use is not a "
                "positive claim for this representation family."
            ),
        },
        "dev_evaluations": 12,
        "final_config_path": "configs/analysis/supervised_motivation_diagnostics_final.yaml",
        "final_config_sha256": sha256_file(
            "configs/analysis/supervised_motivation_diagnostics_final.yaml"
        ),
        "history_controls": {
            "d1a_minus_d1m_by_seed": attention_mean_directions,
            "d1a_true_minus_wrong_by_seed": true_wrong_directions,
            "residual_minus_d1q_by_seed": residual_directions,
        },
        "qrels_read_by_training_or_scoring": False,
        "score_audit": load_json(
            "reports/pps_supervised_diagnostics_score_audit.json"
        ),
        "slice_report_path": "reports/pps_supervised_diagnostics_slices.json",
        "test_read": False,
        "variants": variants,
    }
    output_json = Path("reports/pps_supervised_diagnostics_summary.json")
    write_json(output_json, report)

    lines = [
        "# Supervised Motivation Diagnostics Summary",
        "",
        "Status: complete under the frozen doc 18 protocol; test untouched.",
        "",
        "## Main results",
        "",
        "| Variant | Mean NDCG@10 | Sample SD | Seed values |",
        "|---|---:|---:|---|",
    ]
    labels = {
        "d1q": "D1q supervised query base",
        "d1m": "D1m mean-history residual",
        "d1a": "D1a query-attentive residual",
        "d1a_wrong_history": "D1a matched wrong history",
    }
    for variant in RUN_NAMES:
        row = variants[variant]
        values = ", ".join(f"{value['ndcg@10']:.4f}" for value in row["seeds"])
        lines.append(
            f"| {labels[variant]} | {row['mean_ndcg@10']:.4f} | "
            f"{row['sample_std_ndcg@10']:.4f} | {values} |"
        )
    d1q_b2z = comparisons["d1q_vs_b2z_seed20260708"]
    d1q_b7 = comparisons["d1q_vs_b7_seed20260708"]
    d1a_d1q = comparisons["d1a_vs_d1q_seed20260708"]
    lines.extend(
        [
            "",
            "## Adjudication",
            "",
            f"D1q improves over B2z by {d1q_b2z['delta']:+.4f} "
            f"(95% CI [{d1q_b2z['ci95'][0]:+.4f}, {d1q_b2z['ci95'][1]:+.4f}]) "
            "but remains below B7 by "
            f"{d1q_b7['delta']:+.4f} (95% CI [{d1q_b7['ci95'][0]:+.4f}, "
            f"{d1q_b7['ci95'][1]:+.4f}]).",
            "",
            "D1m and D1a each move above D1q in two seeds and below it in one. "
            f"At the preselected seed, D1a-D1q is {d1a_d1q['delta']:+.4f} with "
            f"CI [{d1a_d1q['ci95'][0]:+.4f}, {d1a_d1q['ci95'][1]:+.4f}]. "
            "D1a does not consistently exceed D1m.",
            "",
            "Matched wrong-history rescoring shows a significant true-history advantage "
            "only at seed 20260708; the other seeds do not reproduce it. The correct "
            "conclusion is therefore that identity signal is visible but this simple "
            "train-fitted residual does not use it stably.",
            "",
            "This is a representation/training negative result, not evidence against the "
            "locked C3-R identity effect. It prohibits claiming that query-attentive event "
            "selection is already established before proposed-system development.",
            "",
            "## Integrity",
            "",
            "All 12 dev evaluations use the shared evaluator and candidate hash. Training "
            "and scoring never read dev/test qrels; test remains untouched. The score audit "
            "verifies exact candidate coverage and exact D1q fallback on 4,110 no-history "
            "requests.",
        ]
    )
    Path("reports/pps_supervised_diagnostics_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["decision"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
