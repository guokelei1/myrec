#!/usr/bin/env python3
"""Report readiness for the final comprehensive Transformer exploration report."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.comprehensive_readiness import (
    build_comprehensive_readiness,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(
        json.dumps(
            build_comprehensive_readiness(args.root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
