#!/usr/bin/env python3
"""Freeze a minimal D2 selected-branch scoring contract."""

from __future__ import annotations

import argparse
import json
import sys

from myrec.mechanism.selected_branch_contract import (
    materialize_selected_branch_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = materialize_selected_branch_contract(
        args.selection, args.confirmation, args.output, command=sys.argv
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
