#!/usr/bin/env python3
"""Run the qrels-blind Q2 production attention-edge smoke."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.attention_edge_smoke import run_q2_attention_edge_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--block", type=int, required=True, choices=(13, 20, 27))
    parser.add_argument("--device", required=True)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    result = run_q2_attention_edge_smoke(
        args.standardized_dir,
        args.config,
        args.checkpoint_root,
        args.run_id,
        block=args.block,
        device=args.device,
        runs_dir=args.runs_dir,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "block_zero_based": result["block_zero_based"],
                "maximum_identity_error": result["maximum_identity_error"],
                "status": result["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
