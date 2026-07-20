#!/usr/bin/env python3
"""Synthesize registered D2 full-layer and transition families."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.postblock_sweep_synthesis import synthesize_postblock_sweeps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q2-fold0-selection")
    parser.add_argument("--q2-fold1-confirmation")
    parser.add_argument("--q3-fold0-selection")
    parser.add_argument("--q3-fold1-confirmation")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--analysis-run-id", required=True)
    parser.add_argument("--dev-eval-log", default="reports/dev_eval_log.jsonl")
    args = parser.parse_args()
    def pair(prefix):
        selection = getattr(args, f"{prefix}_fold0_selection")
        confirmation = getattr(args, f"{prefix}_fold1_confirmation")
        if bool(selection) != bool(confirmation):
            parser.error(f"{prefix} selection and confirmation must be both present or absent")
        return None if not selection else {
            "fold0_selection": selection,
            "fold1_confirmation": confirmation,
        }
    result = synthesize_postblock_sweeps(
        {
            "q2_recranker_generalqwen": pair("q2"),
            "q3_tallrec_generalqwen": pair("q3"),
        },
        args.output_dir,
        args.analysis_run_id,
        dev_eval_log_path=args.dev_eval_log,
    )
    print(json.dumps(result["localization"], sort_keys=True))


if __name__ == "__main__":
    main()
