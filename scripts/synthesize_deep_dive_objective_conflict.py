#!/usr/bin/env python3
"""Synthesize the registered Q2 D7 objective-conflict family."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.objective_conflict_synthesis import (
    synthesize_q2_objective_conflict,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--final-run", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = synthesize_q2_objective_conflict(
        {
            "base_initialization": args.base_run,
            "frozen_final_checkpoint": args.final_run,
        },
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "family_size": result["family"]["observed_size"],
                "status": result["status"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
