#!/usr/bin/env python
"""Train and score the B4o RecBole SASRec baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.recbole_adapter import (  # noqa: E402
    train_and_score_b4o_sasrec,
    write_recbole_atomic_interactions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/b4o_sasrec_recbole.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override RecBole epochs. Omit for the formal official-default run.",
    )
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--hidden-dropout-prob", type=float, default=None)
    parser.add_argument("--attn-dropout-prob", type=float, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-item-list-length", type=int, default=None)
    parser.add_argument(
        "--max-score-requests",
        type=int,
        default=None,
        help="Optional smoke limit for scoring without running the shared evaluator.",
    )
    parser.add_argument("--prepare-atomic-only", action="store_true")
    return parser.parse_args()


def _load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def _recbole_overrides(args: argparse.Namespace) -> dict:
    pairs = {
        "attn_dropout_prob": args.attn_dropout_prob,
        "hidden_dropout_prob": args.hidden_dropout_prob,
        "hidden_size": args.hidden_size,
        "learning_rate": args.learning_rate,
        "MAX_ITEM_LIST_LENGTH": args.max_item_list_length,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
    }
    return {key: value for key, value in pairs.items() if value is not None}


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    if args.prepare_atomic_only:
        manifest = write_recbole_atomic_interactions(
            interactions_path=config["train_interactions"]["path"],
            output_root=config["recbole"]["data_dir"],
            dataset_name=config["recbole"]["dataset_name"],
            report_path=Path("artifacts/batch2b/b4o_recbole/atomic_manifest.json"),
        )
        print(json.dumps({"atomic_manifest": manifest}, ensure_ascii=False, sort_keys=True))
        return 0

    metadata = train_and_score_b4o_sasrec(
        config=config,
        run_id=args.run_id,
        split=args.split,
        seed=args.seed,
        runs_dir=args.runs_dir,
        epochs=args.epochs,
        recbole_overrides=_recbole_overrides(args),
        saved=True,
        max_score_requests=args.max_score_requests,
    )
    print(json.dumps({"metadata": metadata}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
