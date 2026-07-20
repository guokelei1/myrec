#!/usr/bin/env python3
"""Fit frozen train-only M2 brand/category ridge readouts."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.representation_probe import fit_train_representation_probes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--activation-bundle", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--method-id",
        required=True,
        choices=("q2_recranker_generalqwen", "q3_tallrec_generalqwen"),
    )
    args = parser.parse_args()
    result = fit_train_representation_probes(
        args.standardized_dir,
        args.activation_bundle,
        args.output_dir,
        expected_method_id=args.method_id,
        command=sys.argv,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
