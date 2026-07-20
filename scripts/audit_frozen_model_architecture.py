#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from myrec.mechanism.frozen_model_architecture_audit import (
    audit_frozen_model_architecture,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    result = audit_frozen_model_architecture(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
