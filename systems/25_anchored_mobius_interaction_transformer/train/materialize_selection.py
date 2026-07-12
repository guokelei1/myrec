"""Freeze C25 roles and wrong-history donors before labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.structure import PackedStructure, build_selection, load_config, sha256_file, write_json_once  # noqa: E402


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    for name, expected_name in (
        ("candidate_manifest", "candidate_manifest_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C25 registered source changed: {name}")
    data = PackedStructure(paths["packed_train_root"])
    if len(data.request_ids) != int(config["integrity"]["packed_train_requests"]):
        raise ValueError("C25 packed request count changed")
    result = build_selection(data, config)
    result["sources"] = {
        "candidate_manifest_sha256": paths["candidate_manifest_sha256"],
        "packed_manifest_sha256": paths["packed_manifest_sha256"],
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
                "pool_counts": result["pool_counts"],
                "role_counts": {role: len(row["indices"]) for role, row in result["roles"].items()},
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
