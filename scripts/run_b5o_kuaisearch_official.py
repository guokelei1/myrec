#!/usr/bin/env python
"""Train and score the B5o KuaiSearch official ranking baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.kuaisearch_official_adapter import (  # noqa: E402
    ensure_b5o_stageb_embeddings,
    materialize_b5o_stageb_format,
    train_and_score_b5o,
)
from myrec.eval.evaluator import evaluate_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/b5o_kuaisearch_din_dcnv2.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--model", required=True, choices=["DNN", "DCNv2", "DIN"])
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifact-root", default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--embeddings-only", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--overwrite-materialized", action="store_true")
    parser.add_argument("--overwrite-embeddings", action="store_true")
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-score-records", type=int, default=None)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--mixed-precision", choices=["no", "fp16", "bf16"], default=None)
    return parser.parse_args()


def _load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def _hyperparam_overrides(args: argparse.Namespace) -> dict:
    return {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "mixed_precision": args.mixed_precision,
        "num_epochs": args.num_epochs,
        "weight_decay": args.weight_decay,
    }


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)
    artifact_root = args.artifact_root or config["stage_b"]["artifact_root"]
    if args.prepare_only or args.embeddings_only:
        manifest = materialize_b5o_stageb_format(
            standardized_dir=config["standardized_dir"],
            output_root=artifact_root,
            split=args.split,
            max_train_records=args.max_train_records,
            max_score_records=args.max_score_records,
            overwrite=args.overwrite_materialized,
        )
        result = {"materializer_manifest": manifest}
        if args.embeddings_only and not args.skip_embeddings:
            result["embedding_manifest"] = ensure_b5o_stageb_embeddings(
                artifact_root,
                batch_size=int(config["stage_b"]["embedding"]["batch_size"]),
                max_length=int(config["stage_b"]["embedding"]["max_length"]),
                overwrite=args.overwrite_embeddings,
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0

    metadata = train_and_score_b5o(
        config=config,
        run_id=args.run_id,
        model_name=args.model,
        seed=args.seed,
        split=args.split,
        runs_dir=args.runs_dir,
        artifact_root=artifact_root,
        max_train_records=args.max_train_records,
        max_score_records=args.max_score_records,
        overwrite_materialized=args.overwrite_materialized,
        overwrite_embeddings=args.overwrite_embeddings,
        skip_embeddings=args.skip_embeddings,
        hyperparams=_hyperparam_overrides(args),
    )
    result = {"metadata": metadata}
    if args.evaluate:
        if args.max_score_records is not None:
            raise ValueError("--evaluate is only valid for full-score runs")
        metrics = evaluate_run(
            run_id=args.run_id,
            split=args.split,
            candidate_manifest_path=Path(config["standardized_dir"]) / "candidate_manifest.json",
            standardized_dir=config["standardized_dir"],
            runs_dir=args.runs_dir,
        )
        result["metrics"] = metrics
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
