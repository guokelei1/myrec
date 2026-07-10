#!/usr/bin/env python
"""Materialize D2 query tokens aligned with the packed train/dev requests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.finetuned_query_tower import materialize_query_tokens


def main() -> int:
    config_path = Path("configs/analysis/finetuned_nonpersonalized_control.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    result = materialize_query_tokens(config, config_path)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
