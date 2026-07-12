"""Freeze C24 roles before any delayed label access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.structure import (  # noqa: E402
    PackedStructure,
    build_selection,
    load_config,
    read_json,
    sha256_file,
    write_json_once,
)


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    if sha256_file(paths["c23_selection"]) != paths["c23_selection_sha256"]:
        raise ValueError("C23 selection changed")
    if sha256_file(paths["c23_g0_report"]) != paths["c23_g0_report_sha256"]:
        raise ValueError("C23 G0 report changed")
    if read_json(paths["c23_g0_report"]).get("internal_A_labels_opened") is not False:
        raise ValueError("C23 delayed-label boundary differs")
    data = PackedStructure(paths["packed_train_root"])
    c23 = read_json(paths["c23_selection"])
    result = build_selection(data, c23, seed=int(config["selection_seed"]))
    result["sources"] = {
        "c23_selection_path": str(paths["c23_selection"]),
        "c23_selection_sha256": paths["c23_selection_sha256"],
        "c23_g0_report_path": str(paths["c23_g0_report"]),
        "c23_g0_report_sha256": paths["c23_g0_report_sha256"],
        "packed_request_ids_sha256": sha256_file(
            Path(paths["packed_train_root"]) / "request_ids.jsonl"
        ),
    }
    write_json_once(paths["selection"], result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = materialize(args.config)
    print(
        json.dumps(
            {
                "selection": result["selection_id"],
                "counts": {
                    role: len(row["indices"]) for role, row in result["roles"].items()
                },
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
