#!/usr/bin/env python
"""Build the recipe-locked Motivation V1.2 KuaiSearch 4k holdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.kuaisearch_holdout import (  # noqa: E402
    materialize_motivation_v12_kuaisearch_holdout,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch_full")
    parser.add_argument(
        "--development-dir",
        default="data/standardized/kuaisearch/full_confirm_preceding40k_v11",
        help="Frozen 32k train + 8k internal-dev + legacy 2k population.",
    )
    parser.add_argument(
        "--subsequent-scout-dir",
        default="data/standardized/kuaisearch/full_scout10k_query_history_v1",
        help="Later source-train scout that the new holdout must not overlap.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "data/standardized/kuaisearch/"
            "full_confirm_preceding40k_newholdout4k_v12"
        ),
    )
    parser.add_argument(
        "--dataset-version",
        default="full_confirm_preceding40k_newholdout4k_v12",
    )
    parser.add_argument(
        "--protocol",
        default="experiments/motivation/protocol.yaml",
        help="Frozen pre-pilot V1.2 protocol containing data.new_holdout_rule.",
    )
    parser.add_argument(
        "--recipe-checkpoint-lock",
        required=True,
        help=(
            "Independent post-selection JSON lock with protocol/input/config "
            "hashes, one checkpoint-selection identity JSON per method, and "
            "explicit holdout_materialization authorization."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = materialize_motivation_v12_kuaisearch_holdout(
        raw_dir=args.raw_dir,
        development_dir=args.development_dir,
        subsequent_scout_dir=args.subsequent_scout_dir,
        output_dir=args.output_dir,
        protocol_path=args.protocol,
        recipe_checkpoint_lock_path=args.recipe_checkpoint_lock,
        dataset_version=args.dataset_version,
        command_argv=sys.argv,
        enforce_registered_v12_recipe=True,
    )
    print(
        json.dumps(
            {
                "dataset_version": manifest["dataset_version"],
                "protocol_sha256": manifest["freeze_gate"]["protocol"]["sha256"],
                "recipe_checkpoint_lock_sha256": manifest["freeze_gate"][
                    "post_selection_recipe_checkpoint_lock"
                ]["sha256"],
                "confirmation_requests": manifest["selection"][
                    "confirmation_requests"
                ],
                "population_isolation": manifest["population_isolation"],
                "manifest_path": str(Path(args.output_dir) / "manifest.json"),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
