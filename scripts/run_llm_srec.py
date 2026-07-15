#!/usr/bin/env python
"""Score one history condition with a fixed LLM-SRec checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.llm_srec_training import write_llm_srec_scores


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--teacher-store", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--history-assignments", required=True)
    parser.add_argument("--history-condition", choices=("true", "null", "wrong"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = write_llm_srec_scores(
        args.standardized_dir,
        args.teacher_store,
        args.checkpoint_dir,
        args.history_assignments,
        args.run_id,
        history_condition=args.history_condition,
        runs_dir=args.runs_dir,
        device=args.device,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

