#!/usr/bin/env python
"""Freeze final Motivation V1.2 checkpoints and authorize holdout creation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.motivation_v12_release_lock import (  # noqa: E402
    build_motivation_v12_release_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol", default="experiments/motivation/protocol.yaml"
    )
    parser.add_argument(
        "--q0-config",
        default=(
            "configs/methods/"
            "kuaisearch_motivation_v12_q0_qwen3_reranker_06b.yaml"
        ),
    )
    parser.add_argument(
        "--q1-config",
        default=(
            "configs/methods/"
            "kuaisearch_motivation_v12_q1_instructrec_generalqwen.yaml"
        ),
    )
    parser.add_argument(
        "--q2-config",
        default=(
            "configs/methods/"
            "kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
        ),
    )
    parser.add_argument(
        "--q3-config",
        default=(
            "configs/methods/"
            "kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
        ),
    )
    parser.add_argument(
        "--w0-config",
        default=(
            "configs/baselines/"
            "kuaisearch_motivation_v12_copps_transfer_witness.yaml"
        ),
    )
    parser.add_argument("--q0-checkpoint-root", required=True)
    parser.add_argument("--q1-checkpoint-root", required=True)
    parser.add_argument("--q2-checkpoint-root", required=True)
    parser.add_argument("--q3-checkpoint-root", required=True)
    parser.add_argument("--w0-checkpoint-dir", required=True)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch_full")
    parser.add_argument(
        "--development-dir",
        default="data/standardized/kuaisearch/full_confirm_preceding40k_v11",
    )
    parser.add_argument(
        "--subsequent-scout-dir",
        default="data/standardized/kuaisearch/full_scout10k_query_history_v1",
    )
    parser.add_argument("--output-lock-dir", required=True)
    parser.add_argument(
        "--lock-id", default="motivation_v1_2_first_round_post_selection"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_motivation_v12_release_lock(
        protocol_path=args.protocol,
        config_paths={
            "q0_qwen3_reranker_06b": args.q0_config,
            "q1_instructrec_generalqwen": args.q1_config,
            "q2_recranker_generalqwen": args.q2_config,
            "q3_tallrec_generalqwen": args.q3_config,
            "w0_copps_style_transfer_witness": args.w0_config,
        },
        q_checkpoint_roots={
            "q0_qwen3_reranker_06b": args.q0_checkpoint_root,
            "q1_instructrec_generalqwen": args.q1_checkpoint_root,
            "q2_recranker_generalqwen": args.q2_checkpoint_root,
            "q3_tallrec_generalqwen": args.q3_checkpoint_root,
        },
        w0_checkpoint_dir=args.w0_checkpoint_dir,
        raw_dir=args.raw_dir,
        development_dir=args.development_dir,
        subsequent_scout_dir=args.subsequent_scout_dir,
        output_lock_dir=args.output_lock_dir,
        lock_id=args.lock_id,
        command_argv=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
