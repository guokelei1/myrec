#!/usr/bin/env python
"""Rescore D1a with a frozen matched wrong-user history assignment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.supervised_diagnostics import (
    PackedRequestData,
    build_permuted_history_data,
    score_diagnostic,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/analysis/supervised_motivation_diagnostics.yaml",
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--assignments", required=True)
    parser.add_argument(
        "--donor-bank",
        default=(
            "artifacts/analysis/supervised_motivation_diagnostics/"
            "wrong_history_donor_bank.jsonl"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_root = Path(config["materialized_data"]["output_dir"])
    data = PackedRequestData.load(data_root, "dev")
    records_path = Path(config["standardized_dir"]) / "records_dev.jsonl"
    permuted, evidence = build_permuted_history_data(
        data, records_path, args.assignments, args.donor_bank
    )
    result = score_diagnostic(
        config=config,
        config_path=args.config,
        variant="d1a",
        seed=args.seed,
        checkpoint_path=args.checkpoint,
        device=args.device,
        data_override=permuted,
        run_id_override=f"20260710_kuaisearch_d1a_wrong_history_dev_s{args.seed}",
        method_id_override="d1a_query_attn_wrong_history",
        metadata_extra={
            "assignment_path": args.assignments,
            "history_condition": "matched wrong-user history",
            "permutation_evidence": evidence,
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
