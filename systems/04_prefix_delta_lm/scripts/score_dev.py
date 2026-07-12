#!/usr/bin/env python
"""Score frozen label-free KuaiSearch dev records with C04."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from cpdlr.io import load_yaml
from cpdlr.score import score_dev


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CANDIDATE_ROOT / "configs/probe.yaml"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-requests", type=int, default=None)
    parser.add_argument("--no-diagnostics", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    train_run = config["run_ids"]["paired_delta"]
    checkpoint = args.checkpoint or str(Path(config["paths"]["model_dir"]) / f"{train_run}.pt")
    run_id = args.run_id or config["run_ids"]["screening"]
    result = score_dev(
        config,
        args.config,
        checkpoint,
        run_id,
        args.device,
        output_dir=args.output_dir,
        limit_requests=args.limit_requests,
        diagnostics=not args.no_diagnostics,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
