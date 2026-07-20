#!/usr/bin/env python3
"""Synthesize the frozen 96-cell D1 region-decoding family."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.deep_dive_representation_synthesis import (
    synthesize_d1_region_decoding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-evaluation", required=True)
    parser.add_argument("--q3-evaluation", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = synthesize_d1_region_decoding(
        {"q2": args.q2_evaluation, "q3": args.q3_evaluation},
        args.output_dir,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                "family_size": result["family"]["observed_size"],
                "passed_cells": result["passed_cells"],
                "status": result["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

