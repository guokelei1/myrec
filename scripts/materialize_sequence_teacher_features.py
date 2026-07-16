#!/usr/bin/env python
"""Materialize frozen SASRec representations for the LLM-SRec baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines" / "hstu"))

from myrec.baselines.sequence_teacher_features import (
    materialize_sequence_teacher_features,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dev-assignments", nargs="+", required=True)
    parser.add_argument(
        "--evaluation-split", default="dev", choices=("dev", "internal", "confirmation")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    result = materialize_sequence_teacher_features(
        args.standardized_dir,
        args.feature_store,
        args.checkpoint_dir,
        args.output_dir,
        dev_assignment_paths=args.dev_assignments,
        evaluation_split=args.evaluation_split,
        device=args.device,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
