#!/usr/bin/env python3
"""Synthesize all 17 frozen retrospective Transformer supplements."""

from __future__ import annotations

import argparse
import sys

from myrec.mechanism.cross_component_descriptive_synthesis import (
    build_cross_component_descriptive_synthesis,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default=(
            "runs/20260719_kuaisearch_mech_cross_component_descriptive_v1/"
            "metrics.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_cross_component_descriptive_synthesis(
        args.root,
        output_path=args.output,
        overwrite=args.overwrite,
        command=sys.argv,
    )


if __name__ == "__main__":
    main()
