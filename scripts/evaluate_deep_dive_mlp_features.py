#!/usr/bin/env python3
"""Aggregate all six fixed-grid SwiGLU formation bundles without qrels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from myrec.mechanism.mlp_feature_evaluator import evaluate_mlp_feature_bundles
from myrec.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for model in ("q2", "q3"):
        for block in (13, 20, 27):
            parser.add_argument(f"--{model}-b{block}", required=True)
    parser.add_argument(
        "--output-dir",
        default="runs/20260719_kuaisearch_mech_d4_mlp_formation_eval_v1",
    )
    parser.add_argument(
        "--analysis-run-id",
        default="20260719_kuaisearch_mech_d4_mlp_formation_eval_v1",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"MLP formation evaluation output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    bundles = {
        "q2_recranker_generalqwen": {
            block: getattr(args, f"q2_b{block}") for block in (13, 20, 27)
        },
        "q3_tallrec_generalqwen": {
            block: getattr(args, f"q3_b{block}") for block in (13, 20, 27)
        },
    }
    result = evaluate_mlp_feature_bundles(bundles)
    result.update(
        {
            "analysis_run_id": args.analysis_run_id,
            "command": [str(value) for value in os.sys.argv],
        }
    )
    output_path = output_dir / "metrics.json"
    temporary = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "cells": len(result["cells"]),
                "qrels_read": result["qrels_read"],
                "sha256": sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
