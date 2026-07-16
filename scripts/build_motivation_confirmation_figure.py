#!/usr/bin/env python
"""Render the frozen confirmation result as a paper-ready two-panel figure."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


INPUT = Path("reports/pps_motivation_confirmation_decision.json")
OUTPUT_DIR = Path("reports/figures/motivation_repair")
STEM = "motivation_frozen_confirmation"


def main() -> int:
    report = json.loads(INPUT.read_text(encoding="utf-8"))
    if report.get("evidence_mode") != "frozen_confirmation":
        raise ValueError("input is not a frozen confirmation decision")
    primary = report["primary_scientific_endpoints"]
    secondary = report["secondary_diagnostics"]["surfaces"]
    surfaces = [
        ("Target recurrence", primary["target_repeat"]),
        (
            "Other-candidate\noverlap",
            secondary["target_nonrepeat_other_candidate_overlap"],
        ),
        ("New-slate transfer", primary["target_nonrepeat_no_candidate_overlap"]),
    ]

    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "font.size": 10,
            "figure.dpi": 150,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1))

    means = [row[1]["true_minus_null_ndcg@10"] for row in surfaces]
    cis = [
        row[1]["true_minus_null_ndcg@10_query_cluster_ci95"] for row in surfaces
    ]
    errors = [
        [mean - ci[0] for mean, ci in zip(means, cis)],
        [ci[1] - mean for mean, ci in zip(means, cis)],
    ]
    colors = ["#2878B5", "#D9534F", "#F2A104"]
    axes[0].bar(range(3), means, color=colors, width=0.64, alpha=0.9)
    axes[0].errorbar(
        range(3),
        means,
        yerr=errors,
        fmt="none",
        ecolor="#202020",
        elinewidth=1.2,
        capsize=4,
    )
    axes[0].axhline(0.0, color="#555555", linewidth=0.9)
    bound = float(primary["practical_nonrepeat_upper_bound"])
    axes[0].axhline(
        bound,
        color="#777777",
        linewidth=0.9,
        linestyle="--",
        label=f"pre-registered materiality bound ({bound:.2f})",
    )
    axes[0].set_xticks(range(3), [row[0] for row in surfaces])
    axes[0].set_ylabel(r"Same-checkpoint recovery ($\Delta$NDCG@10)")
    axes[0].set_title("(a) History value is surface-selective")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")

    endpoints = report["all_request_and_conditional_positive"]
    names = ["BM25", "QC", "FULL-true"]
    all_values = [
        endpoints["bm25"]["ndcg@10_all_requests"],
        endpoints["qc"]["ndcg@10_all_requests"],
        endpoints["full"]["true_ndcg@10_all_requests"],
    ]
    positive_values = [
        endpoints["bm25"]["ndcg@10_positive_requests"],
        endpoints["qc"]["ndcg@10_positive_requests"],
        endpoints["full"]["true_ndcg@10_positive_requests"],
    ]
    x = range(3)
    axes[1].bar(
        [value - 0.18 for value in x],
        all_values,
        width=0.36,
        color="#4C78A8",
        label="All requests",
    )
    axes[1].bar(
        [value + 0.18 for value in x],
        positive_values,
        width=0.36,
        color="#72B7B2",
        label="Observed-positive",
    )
    axes[1].set_xticks(list(x), names)
    axes[1].set_ylabel("Graded NDCG@10")
    axes[1].set_title("(b) Task endpoint and estimand accounting")
    axes[1].legend(frameon=False, fontsize=8)

    decision = report["hierarchical_decision"]["decision"].replace("_", " ")
    fig.suptitle(f"Disjoint-window frozen confirmation: {decision}", y=1.02)
    fig.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{STEM}.png"
    pdf = OUTPUT_DIR / f"{STEM}.pdf"
    fig.savefig(png, dpi=240, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "analysis_type": "frozen_confirmation_paper_figure",
        "decision": report["hierarchical_decision"]["decision"],
        "input": {"path": str(INPUT), "sha256": sha256_file(INPUT)},
        "outputs": {
            "pdf": {"path": str(pdf), "sha256": sha256_file(pdf)},
            "png": {"path": str(png), "sha256": sha256_file(png)},
        },
    }
    metadata_path = OUTPUT_DIR / f"{STEM}_metadata.json"
    write_json(metadata_path, metadata)
    print(json.dumps({"metadata": str(metadata_path), **metadata["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
