#!/usr/bin/env python3
"""Fit D1 all-position, all-layer train-only ridge probes."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.deep_dive_representation_analysis import (
    fit_deep_dive_representation_probes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--activation-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--method-id", required=True)
    args = parser.parse_args()
    result = fit_deep_dive_representation_probes(
        args.standardized_dir,
        args.activation_bundle,
        args.output_dir,
        expected_method_id=args.method_id,
        command=sys.argv,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("method_id", "probe_checkpoint_id", "status")
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

