#!/usr/bin/env python3
"""Build the fail-closed final comprehensive Transformer report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myrec.mechanism.comprehensive_report_builder import build_comprehensive_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--decisions", required=True)
    parser.add_argument(
        "--formal-report",
        default="reports/motivation_transformer_deep_dive.json",
    )
    parser.add_argument(
        "--json-output",
        default="reports/motivation_transformer_comprehensive.json",
    )
    parser.add_argument(
        "--markdown-output",
        default="reports/motivation_transformer_comprehensive.md",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    decisions = json.loads(Path(args.decisions).read_text(encoding="utf-8"))
    build_comprehensive_report(
        args.root,
        decisions,
        formal_report_path=args.formal_report,
        json_output=args.json_output,
        markdown_output=args.markdown_output,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
