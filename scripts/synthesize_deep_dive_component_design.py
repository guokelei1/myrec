#!/usr/bin/env python3
"""Combine component sufficiency, specificity, and reverse necessity gates."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.component_design_synthesis import (
    synthesize_component_design_gates,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--necessity-metrics", required=True)
    parser.add_argument("--selected-synthesis", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    args = parser.parse_args()
    result = synthesize_component_design_gates(
        args.necessity_metrics,
        args.selected_synthesis,
        args.output_dir,
        args.analysis_run_id,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                "cross_model_design_prioritized_nodes": result[
                    "cross_model_functional_support"
                ]["design_prioritized_nodes"],
                "status": result["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
