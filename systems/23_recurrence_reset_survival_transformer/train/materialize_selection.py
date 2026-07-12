"""Freeze the C23 cohort from label-free structure only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.structure import (
    PackedStructure,
    build_selection,
    load_config,
    selected_ids_from_prior,
    sha256_file,
    write_json_once,
)


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path, require_frozen_selection=False)
    paths = config["paths"]
    if sha256_file(paths["packed_manifest"]) != paths["packed_manifest_sha256"]:
        raise ValueError("packed manifest changed")
    if sha256_file(paths["candidate_manifest"]) != paths["candidate_manifest_sha256"]:
        raise ValueError("candidate manifest changed")
    if sha256_file(paths["c05_selection"]) != paths["c05_selection_sha256"]:
        raise ValueError("C05 selection changed")
    if sha256_file(paths["c06_selection"]) != paths["c06_selection_sha256"]:
        raise ValueError("C06 selection changed")
    blacklist = selected_ids_from_prior(paths["c05_selection"])
    blacklist |= selected_ids_from_prior(paths["c06_selection"])
    data = PackedStructure.load(paths["packed_train_root"])
    if len(data) != int(config["integrity"]["packed_train_requests"]):
        raise ValueError("packed train request count changed")
    result = build_selection(
        data,
        cut=int(config["integrity"]["packed_cut_request_index"]),
        seed=int(config["selection_seed"]),
        blacklist=blacklist,
    )
    result["sources"] = {
        "packed_request_ids": {
            "path": str(data.root / "request_ids.jsonl"),
            "sha256": sha256_file(data.root / "request_ids.jsonl"),
        },
        "packed_candidate_offsets": {
            "path": str(data.root / "candidate_offsets.npy"),
            "sha256": sha256_file(data.root / "candidate_offsets.npy"),
        },
        "packed_candidate_embedding_indices": {
            "path": str(data.root / "candidate_embedding_indices.npy"),
            "sha256": sha256_file(data.root / "candidate_embedding_indices.npy"),
        },
        "packed_candidate_item_ids": {
            "path": str(data.root / "candidate_item_ids.npy"),
            "sha256": sha256_file(data.root / "candidate_item_ids.npy"),
        },
        "packed_history_offsets": {
            "path": str(data.root / "history_offsets.npy"),
            "sha256": sha256_file(data.root / "history_offsets.npy"),
        },
        "packed_history_embedding_indices": {
            "path": str(data.root / "history_embedding_indices.npy"),
            "sha256": sha256_file(data.root / "history_embedding_indices.npy"),
        },
        "c05_selection": {
            "path": str(paths["c05_selection"]),
            "sha256": paths["c05_selection_sha256"],
        },
        "c06_selection": {
            "path": str(paths["c06_selection"]),
            "sha256": paths["c06_selection_sha256"],
        },
    }
    result["blacklisted_registered_request_count"] = len(blacklist)
    write_json_once(paths["selection"], result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = materialize(args.config)
    print(
        json.dumps(
            {
                "path": str(load_config(args.config)["paths"]["selection"]),
                "selection": result["selection_id"],
                "counts": {
                    role: len(row["indices"]) for role, row in result["roles"].items()
                },
                "pool_counts": result["pool_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
