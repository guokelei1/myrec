#!/usr/bin/env python3
"""Materialize the deliberately non-final Transformer decision worksheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from myrec.mechanism.deep_dive_decision_worksheet import (
    build_deep_dive_decision_worksheet,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    worksheet = build_deep_dive_decision_worksheet(args.root)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(worksheet, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
