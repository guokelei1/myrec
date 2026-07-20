#!/usr/bin/env python3
"""Evaluate the frozen Q2/Q3 reverse component-state removal family."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.component_necessity_evaluator import (
    evaluate_component_necessity,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--qrels-split-dir", required=True)
    parser.add_argument("--q2-bundle")
    parser.add_argument("--q2-gate-contract")
    parser.add_argument("--q3-bundle")
    parser.add_argument("--q3-gate-contract")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    model_inputs = {}
    for short, method_id in (
        ("q2", "q2_recranker_generalqwen"),
        ("q3", "q3_tallrec_generalqwen"),
    ):
        bundle = getattr(args, f"{short}_bundle")
        gate = getattr(args, f"{short}_gate_contract")
        if (bundle is None) == (gate is None):
            parser.error(
                f"{short} requires exactly one of --{short}-bundle or "
                f"--{short}-gate-contract"
            )
        model_inputs[method_id] = (
            {"bundle": bundle} if bundle is not None else {"gate_contract": gate}
        )
    result = evaluate_component_necessity(
        args.standardized_dir,
        args.qrels_split_dir,
        model_inputs,
        args.output_dir,
        args.analysis_run_id,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                "analysis_run_id": result["analysis_run_id"],
                "status": result["status"],
                "strict_transfer_requests": result["strict_transfer_requests"],
                "family_rows": len(result["family_rows"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
