#!/usr/bin/env python
"""Build the shared train-donor bank for D1a wrong-history rescoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.supervised_diagnostics import (
    materialize_wrong_history_donor_bank,
)
from myrec.utils.jsonl import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/analysis/supervised_motivation_diagnostics.yaml",
    )
    parser.add_argument(
        "--assignment-dir",
        default="artifacts/analysis/c3_history_identity_controls",
    )
    parser.add_argument(
        "--output",
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
    assignment_dir = Path(args.assignment_dir)
    assignments = [
        assignment_dir / f"donor_assignments_s{seed}.jsonl"
        for seed in config["seeds"]
    ]
    embedding_root = Path(config["embedding_artifact"]["root"])
    result = materialize_wrong_history_donor_bank(
        train_records_path=Path(config["standardized_dir"]) / "records_train.jsonl",
        assignments_paths=assignments,
        item_index_path=embedding_root / config["embedding_artifact"]["item_index"],
        output_path=args.output,
    )
    result.update(
        {
            "config_path": args.config,
            "qrels_read": False,
            "test_read": False,
        }
    )
    manifest_path = Path(args.output).with_suffix(".manifest.json")
    write_json(manifest_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
