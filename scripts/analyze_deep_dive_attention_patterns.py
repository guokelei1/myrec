#!/usr/bin/env python3
"""Build the qrels-blind D3 cross-model attention concentration synthesis."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from myrec.mechanism.attention_pattern_analysis import summarize_attention_patterns
from myrec.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="runs/20260718_kuaisearch_mech_d3_attention_heads_eval_v1/metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/20260719_kuaisearch_mech_d3_attention_patterns_v1",
    )
    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"attention pattern output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = json.loads(input_path.read_text(encoding="utf-8"))
    result = summarize_attention_patterns(metrics)
    result.update(
        {
            "input_path": str(input_path),
            "input_sha256": sha256_file(input_path),
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
