#!/usr/bin/env python
"""Audit D2h true/wrong score coverage and no-history fallback without qrels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.core import _zscore_map
from myrec.eval.evaluator import (
    _assert_score_coverage,
    _load_candidate_manifest,
    _load_scores,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


SEEDS = [20260708, 20260709, 20260710]


def main() -> int:
    candidate_manifest = Path(
        "data/standardized/kuaisearch/v0_lite/candidate_manifest.json"
    )
    candidate_sha = sha256_file(candidate_manifest)
    candidates = _load_candidate_manifest(candidate_manifest, "dev")
    no_history_ids = {
        line.strip()
        for line in Path(
            "artifacts/analysis/c3_history_identity_controls/"
            "history_absent_request_ids.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    final_config = Path(
        "configs/analysis/d2h_static_history_control_final.yaml"
    )
    final_sha = sha256_file(final_config)
    alpha = 0.1
    runs = {}
    fallback = {}
    for seed in SEEDS:
        query_path = Path(
            f"runs/20260710_kuaisearch_d2t_finetuned_text_dev_s{seed}/"
            "scores.jsonl"
        )
        query_scores, _ = _load_scores(query_path)
        condition_scores = {}
        for condition in ("true", "wrong"):
            run_id = (
                f"20260710_kuaisearch_d2h_static_{condition}_history_dev_s{seed}"
            )
            run_dir = Path("runs") / run_id
            with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("qrels_read") is not False or metadata.get("test_read") is not False:
                raise AssertionError(f"label-isolation metadata failed: {run_id}")
            if metadata.get("candidate_manifest_sha256") != candidate_sha:
                raise AssertionError(f"candidate hash mismatch: {run_id}")
            if metadata.get("final_config_sha256") != final_sha:
                raise AssertionError(f"final config hash mismatch: {run_id}")
            scores_path = run_dir / "scores.jsonl"
            scores, method_id = _load_scores(scores_path)
            _assert_score_coverage(candidates, scores)
            condition_scores[condition] = scores
            runs[run_id] = {
                "method_id": method_id,
                "request_count": len(scores),
                "score_rows": sum(len(items) for items in scores.values()),
                "scores_sha256": sha256_file(scores_path),
                "status": "passed",
            }

        max_affine_error = 0.0
        true_wrong_mismatches = 0
        candidate_rows = 0
        for request_id in no_history_ids:
            query_z = _zscore_map(query_scores[request_id])
            for item_id, value in condition_scores["true"][request_id].items():
                max_affine_error = max(
                    max_affine_error, abs(value - alpha * query_z[item_id])
                )
                if value != condition_scores["wrong"][request_id][item_id]:
                    true_wrong_mismatches += 1
                candidate_rows += 1
        if max_affine_error > 1e-12 or true_wrong_mismatches:
            raise AssertionError(
                f"no-history fallback failed for seed {seed}: "
                f"error={max_affine_error} mismatches={true_wrong_mismatches}"
            )
        fallback[str(seed)] = {
            "candidate_rows": candidate_rows,
            "max_affine_error": max_affine_error,
            "request_count": len(no_history_ids),
            "true_wrong_exact_mismatches": true_wrong_mismatches,
            "status": "passed",
        }

    report = {
        "analysis_id": "d2h_static_history_waterline_v1",
        "candidate_manifest_sha256": candidate_sha,
        "final_config_sha256": final_sha,
        "no_history_fallback": fallback,
        "qrels_read": False,
        "runs": runs,
        "status": "passed",
        "test_read": False,
    }
    write_json("reports/pps_d2h_score_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
