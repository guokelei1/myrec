#!/usr/bin/env python
"""Train/resume the frozen V1.2 CoPPS-style structural transfer witness."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.copps_transfer_witness import (  # noqa: E402
    SAFE_EXIT_SECONDS,
    train_copps_transfer_witness,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--feature-store", required=True)
    parser.add_argument("--output-model-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--config",
        required=True,
        help="Frozen W0 config; production training refuses an implicit recipe.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--safe-exit-seconds", type=int, default=SAFE_EXIT_SECONDS)
    args = parser.parse_args()
    result = train_copps_transfer_witness(
        args.standardized_dir,
        args.feature_store,
        args.output_model_dir,
        args.run_id,
        runs_dir=args.runs_dir,
        config_path=args.config,
        device=args.device,
        resume=args.resume,
        safe_exit_seconds=args.safe_exit_seconds,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
