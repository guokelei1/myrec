#!/usr/bin/env python
"""Materialize qrels-free M1 history interventions on internal-dev only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.mechanism.history_interventions import (  # noqa: E402
    MECHANISM_INTERVENTION_SEED,
    materialize_history_interventions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records-train",
        required=True,
        help="Frozen KuaiSearch records_train.jsonl donor population.",
    )
    parser.add_argument(
        "--records-dev",
        required=True,
        help="Label-free KuaiSearch records_dev.jsonl intervention population.",
    )
    parser.add_argument(
        "--feature-store",
        required=True,
        help="Frozen BGE query/context/item-semantic feature store.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="New empty run-local directory for condition JSONL and manifest.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_history_interventions(
        args.records_train,
        args.records_dev,
        args.feature_store,
        args.output_dir,
        seed=MECHANISM_INTERVENTION_SEED,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
