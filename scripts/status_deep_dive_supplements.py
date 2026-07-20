#!/usr/bin/env python3
"""Report outcome-independent completion of the supplemental evidence registry."""

from __future__ import annotations

import argparse
import json

from myrec.mechanism.supplemental_evidence_registry import (
    audit_supplemental_evidence_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    print(
        json.dumps(
            audit_supplemental_evidence_registry(args.root),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
