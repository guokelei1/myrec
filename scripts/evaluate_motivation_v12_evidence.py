#!/usr/bin/env python
"""Audit, evaluate, or re-summarize a Motivation V1.2 score bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.eval.motivation_v12_evidence import (  # noqa: E402
    DEFAULT_PROTOCOL_PATH,
    audit_motivation_v12_score_bundle,
    build_motivation_v12_evidence,
    evaluate_motivation_v12_evidence,
)
from myrec.utils.jsonl import write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser(
        "audit", help="run the qrels-free score-bundle integrity audit"
    )
    _add_bundle_args(audit)
    audit.add_argument("--output", required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="audit, invoke the shared evaluator, and summarize"
    )
    _add_bundle_args(evaluate)
    evaluate.add_argument("--analysis-run-id", required=True)
    evaluate.add_argument("--activity-epsilon", default=0.01, type=float)
    evaluate.add_argument("--utility-epsilon", default=0.0, type=float)
    evaluate.add_argument(
        "--expected-qrels-sha256",
        default=None,
        help=(
            "Optional assertion only. Internal-dev/legacy derive the hash from "
            "the frozen protocol; the registered holdout derives it from its "
            "integrity lock."
        ),
    )
    evaluate.add_argument(
        "--label-mode",
        default="graded",
        choices=("click", "purchase", "graded"),
    )
    evaluate.add_argument(
        "--dev-eval-log", default="reports/dev_eval_log.jsonl"
    )

    summarize = subparsers.add_parser(
        "summarize", help="consume an existing shared-evaluator analysis"
    )
    summarize.add_argument("--analysis-run-id", required=True)
    summarize.add_argument("--standardized-dir", required=True)
    summarize.add_argument("--runs-dir", default="runs")
    summarize.add_argument("--score-audit", default=None)
    summarize.add_argument("--output", default=None)
    return parser.parse_args()


def _add_bundle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--full-run-id", required=True)
    parser.add_argument("--null-run-id", required=True)
    parser.add_argument("--wrong-run-id", default=None)
    parser.add_argument("--split", required=True, choices=("dev", "confirmation"))
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--candidate-manifest", default=None)
    parser.add_argument(
        "--protocol",
        default=str(DEFAULT_PROTOCOL_PATH),
        help="Current frozen Motivation V1.2 protocol to bind into the audit.",
    )
    parser.add_argument("--runs-dir", default="runs")


def main() -> int:
    args = parse_args()
    if args.command == "summarize":
        result = build_motivation_v12_evidence(
            analysis_run_id=args.analysis_run_id,
            standardized_dir=args.standardized_dir,
            runs_dir=args.runs_dir,
            score_audit_path=args.score_audit,
            output_path=args.output,
        )
    else:
        candidate_manifest = args.candidate_manifest or str(
            Path(args.standardized_dir) / "candidate_manifest.json"
        )
        common = {
            "full_run_id": args.full_run_id,
            "null_run_id": args.null_run_id,
            "wrong_run_id": args.wrong_run_id,
            "split": args.split,
            "standardized_dir": args.standardized_dir,
            "candidate_manifest_path": candidate_manifest,
            "protocol_path": args.protocol,
            "runs_dir": args.runs_dir,
        }
        if args.command == "audit":
            result = audit_motivation_v12_score_bundle(**common)
            write_json(args.output, result)
        else:
            result = evaluate_motivation_v12_evidence(
                **common,
                analysis_run_id=args.analysis_run_id,
                activity_epsilon=args.activity_epsilon,
                utility_epsilon=args.utility_epsilon,
                label_mode=args.label_mode,
                expected_qrels_sha256=args.expected_qrels_sha256,
                dev_eval_log_path=args.dev_eval_log,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
