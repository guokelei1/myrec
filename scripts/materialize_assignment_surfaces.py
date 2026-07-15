#!/usr/bin/env python
"""Write request-ID surfaces for wrong-history assignment match types."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.assignment_surfaces import materialize_assignment_surfaces


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--intersect-surface",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Also materialize each assignment group intersected with this surface.",
    )
    args = parser.parse_args()
    intersections = {}
    for value in args.intersect_surface:
        if "=" not in value:
            parser.error("--intersect-surface must use NAME=PATH")
        name, path = value.split("=", 1)
        intersections[name] = path
    result = materialize_assignment_surfaces(
        args.assignments,
        args.output_dir,
        intersection_surfaces=intersections,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
