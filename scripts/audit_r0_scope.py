#!/usr/bin/env python
"""Run the label-safe R0-A cross-dataset scope and holdout/power audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.r0_scope import (  # noqa: E402
    mde_from_reference_ci,
    required_units_for_mde,
    split_user_overlap,
    summarize_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", default="reports/pps_r0_scope_and_holdout_audit.json"
    )
    return parser.parse_args()


def load_json(path: str) -> dict[str, Any]:
    with (ROOT / path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jd_sample_summary() -> dict[str, Any]:
    behavior = ROOT / "data/raw/jdsearch/repository/user_behavior_data_sample.txt"
    product = ROOT / "data/raw/jdsearch/repository/product_meta_data_sample.txt"
    with behavior.open("r", encoding="utf-8", errors="replace") as handle:
        behavior_rows = max(0, sum(1 for _ in handle) - 1)
    with product.open("r", encoding="utf-8", errors="replace") as handle:
        product_rows = max(0, sum(1 for _ in handle) - 1)
    return {
        "local_availability": "schema_samples_only",
        "sample_behavior_rows": behavior_rows,
        "sample_product_rows": product_rows,
        "query": "anonymized_term_ids_not_plaintext",
        "history": ["anonymized_query_term_ids", "item_id", "event_type", "time_interval"],
        "candidates": ["item_id", "graded_label_in_raw_source"],
        "catalog": ["anonymized_name_terms", "brand_id", "category_ids", "shop_id"],
        "historical_slate": True,
        "timestamp_semantics": "relative_interaction_intervals",
        "declared_full_scale": {
            "users": 173831,
            "items": 12872636,
            "interactions": 26667260,
            "test_queries": 171728,
        },
        "claim_role": "conditional_id_and_interaction_robustness_only",
        "token_semantic_claim_supported": False,
        "full_data_blocker": "JD Cloud login/QR acquisition required; not binding for semantic R0 parity",
        "sample_sha256": {
            "behavior": sha256(behavior),
            "product": sha256(product),
        },
    }


def main() -> int:
    args = parse_args()
    kuai_root = ROOT / "data/standardized/kuaisearch/v0_lite"
    amazon_root = ROOT / "data/standardized/amazon_c4/v0_history_bm25_100"
    kuai_train, kuai_train_users = summarize_records([kuai_root / "records_train.jsonl"])
    kuai_dev, kuai_dev_users = summarize_records([kuai_root / "records_dev.jsonl"])
    amazon_train, amazon_train_users = summarize_records(
        [amazon_root / "records_train_blind.jsonl"]
    )
    amazon_dev, amazon_dev_users = summarize_records([amazon_root / "records_dev.jsonl"])

    kuai_c0 = load_json("reports/pps_c0_data_audit.json")
    kuai_c1 = load_json("reports/pps_c1_protocol.json")
    amazon_c1 = load_json("reports/pps_c1_amazon_c4_protocol.json")
    amazon_token = load_json("reports/pps_amazon_token_history_observability_v1.json")
    c5r3 = load_json("reports/pps_c5r3_candidate_history_alignment.json")
    c80 = load_json("reports/pps_c80_amazon_real_gate.json")

    kuai_reference = c5r3["comparisons"]["item_vs_d2p"]["20260708"]
    amazon_reference = amazon_token["comparisons"]["true_minus_null"]
    kuai_baseline = 0.3454
    amazon_baseline = float(amazon_token["metrics"]["ensemble"]["true"])
    kuai_claim_mde = 0.02 * kuai_baseline
    amazon_claim_mde = 0.02 * amazon_baseline
    raw_requests = int(kuai_c0["audits"]["recall"]["rows"])
    selected_requests = int(
        kuai_c0["audits"]["recall"]["time_window_sample"]["actual_window_size"]
    )

    report = {
        "report_id": "pps_r0_scope_and_holdout_audit",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "research_phase": "R0-A",
        "label_boundary": {
            "qrels_read_by_this_audit": False,
            "dev_records_label_free": True,
            "test_scores_or_metrics_computed": False,
            "test_opened": False,
            "c80_fresh_labels_opened": bool(
                not c80["label_boundary"]["fresh_labels_opened"] is False
            ),
            "amazon_historical_reserve_is_fresh_confirmation": False,
        },
        "datasets": {
            "kuaisearch": {
                "role": "main",
                "dataset_version": "v0_lite",
                "information_objects": {
                    "query": "real_plaintext",
                    "history": "strictly_prior_item_text_id_event_timestamp",
                    "candidates": "logged_fixed_slate_with_plaintext_and_attributes",
                    "labels": "click_primary_purchase_secondary_physically_separated_on_dev_test",
                },
                "train_surface": kuai_train,
                "dev_surface": kuai_dev,
                "train_dev_user_isolation": split_user_overlap(
                    kuai_train_users, kuai_dev_users
                ),
                "candidate_manifest_sha256": kuai_c1["candidate_manifest_sha256"],
                "supported_claims": [
                    "full_token_query_history_candidate_interaction",
                    "exact_item_recurrence",
                    "real_query_personalized_ranking",
                ],
                "unsupported_without_new_evidence": [
                    "coarse_semantic_history_transfer",
                    "ordinary_transformer_failure",
                ],
            },
            "amazon_c4": {
                "role": "comparable_secondary",
                "dataset_version": "v0_history_bm25_100",
                "information_objects": {
                    "query": "english_rewritten_plaintext",
                    "history": "strictly_prior_purchase_item_text_id_timestamp",
                    "candidates": "constructed_bm25_top100_plus_target_with_plaintext",
                    "labels": "single_target_physically_separated_on_dev_test",
                },
                "train_surface": amazon_train,
                "dev_surface": amazon_dev,
                "train_dev_user_isolation": split_user_overlap(
                    amazon_train_users, amazon_dev_users
                ),
                "candidate_manifest_sha256": amazon_c1["files"]["candidate_manifest"][
                    "manifest_sha256"
                ],
                "supported_claims": [
                    "full_token_query_history_candidate_interaction",
                    "strict_nonrepeat_semantic_history_value",
                ],
                "unsupported_without_new_evidence": ["real_query_ecological_validity"],
            },
            "jdsearch": jd_sample_summary(),
        },
        "cross_dataset_claim_boundary": {
            "kuai_amazon_common": "plaintext full-token Q-H-C interaction",
            "all_three_common": "anonymized item/query IDs, interaction types, temporal history, candidate slate",
            "jd_exclusion": "JDsearch cannot bind a plaintext token-semantic primitive",
            "dataset_name_branch_allowed": False,
        },
        "holdout_and_power": {
            "kuaisearch": {
                "dev_requests": kuai_c1["split_summary"]["dev"]["request_count"],
                "dev_users": kuai_c1["split_summary"]["dev"]["user_count"],
                "test_requests_locked": kuai_c1["split_summary"]["test"]["request_count"],
                "test_users_locked": kuai_c1["split_summary"]["test"]["user_count"],
                "test_role": "one_shot_final_only_not_controller_confirmation",
                "raw_requests": raw_requests,
                "current_window_requests": selected_requests,
                "unused_raw_request_capacity": raw_requests - selected_requests,
                "independent_confirmation_status": "feasible_from_unused_raw_but_not_materialized",
                "power_proxy": {
                    "source": "C5-R3 item-only vs D2p paired request CI; surface proxy, not a Failure Card effect",
                    "reference_requests": kuai_reference["num_requests"],
                    "reference_ci95": kuai_reference["ci95"],
                    "claim_mde_absolute": kuai_claim_mde,
                    "required_requests_approx": required_units_for_mde(
                        kuai_reference["ci95"],
                        kuai_reference["num_requests"],
                        kuai_claim_mde,
                    ),
                    "mde_at_3000_requests_approx": mde_from_reference_ci(
                        kuai_reference["ci95"], kuai_reference["num_requests"], 3000
                    ),
                },
            },
            "amazon_c4": {
                "dev_requests": amazon_c1["split_summary"]["dev"]["request_count"],
                "dev_users": amazon_c1["split_summary"]["dev"]["user_count"],
                "test_requests_locked": amazon_c1["split_summary"]["test"]["request_count"],
                "test_users_locked": amazon_c1["split_summary"]["test"]["user_count"],
                "historical_full_token_reserve_requests": amazon_token["reserve_requests"],
                "historical_full_token_reserve_users": amazon_token["reserve_users"],
                "historical_reserve_status": "opened_for_observability_not_independent_confirmation",
                "independent_confirmation_status": "test_locked; no separate fresh local cohort registered",
                "power_proxy": {
                    "source": "historical full-token true-minus-null user-cluster CI",
                    "reference_users": amazon_reference["users"],
                    "reference_ci95": amazon_reference["user_cluster_95_ci"],
                    "claim_mde_absolute": amazon_claim_mde,
                    "required_users_approx": required_units_for_mde(
                        amazon_reference["user_cluster_95_ci"],
                        amazon_reference["users"],
                        amazon_claim_mde,
                    ),
                    "mde_at_locked_test_users_approx": mde_from_reference_ci(
                        amazon_reference["user_cluster_95_ci"],
                        amazon_reference["users"],
                        amazon_c1["split_summary"]["test"]["user_count"],
                    ),
                },
            },
        },
        "r0_a_decision": {
            "status": "passed",
            "reason": "KuaiSearch and Amazon support full-token parity; JD is a conditional non-plaintext anchor; a new Kuai confirmation cohort is locally feasible and must be frozen before CONF.",
            "next_phase": "R0-B",
            "next_action": "establish KuaiSearch full-token observability parity and retain the existing comparable Amazon full-token result as source evidence",
        },
    }
    if report["label_boundary"]["c80_fresh_labels_opened"]:
        raise RuntimeError("C80 fresh-label lock is not intact")
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"output": str(output.relative_to(ROOT)), "status": "passed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

