"""Materialize untouched C38 escrow as C42 confirmation A."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def build(
    *,
    c38_selection_path: Path,
    prior_feature_index_paths: list[Path],
) -> dict[str, Any]:
    c38 = read_json(c38_selection_path)
    role = c38["roles"]["escrow"]
    indices = set(int(value) for value in role["indices"])
    prior = set()
    for path in prior_feature_index_paths:
        prior.update(int(value) for value in np.load(path, allow_pickle=False))
    other_roles = set(
        int(value)
        for name in ("fit", "internal_A", "delayed_B")
        for value in c38["roles"][name]["indices"]
    )
    isolation = {
        "internal_A_from_c38_escrow": len(indices),
        "internal_A_overlap_c38_other_roles": len(indices & other_roles),
        "internal_A_overlap_any_prior_feature_materialized": len(indices & prior),
    }
    if isolation != {
        "internal_A_from_c38_escrow": 1200,
        "internal_A_overlap_c38_other_roles": 0,
        "internal_A_overlap_any_prior_feature_materialized": 0,
    }:
        raise RuntimeError(f"C42 isolation failed: {isolation}")
    donors = {
        str(index): c38["wrong_donors"][str(index)] for index in sorted(indices)
    }
    return {
        "candidate_id": "c42",
        "seed": 20262500,
        "source_selection": str(c38_selection_path),
        "source_selection_sha256": sha256_file(c38_selection_path),
        "records_path": c38["records_path"],
        "records_sha256": c38["records_sha256"],
        "standardized_manifest_path": c38["standardized_manifest_path"],
        "standardized_manifest_sha256": c38["standardized_manifest_sha256"],
        "candidate_manifest_path": c38["candidate_manifest_path"],
        "candidate_manifest_sha256": c38["candidate_manifest_sha256"],
        "c0_report_path": c38["c0_report_path"],
        "c0_report_sha256": c38["c0_report_sha256"],
        "history_length_bins": c38["history_length_bins"],
        "train_requests": c38["train_requests"],
        "roles": {"internal_A": role},
        "wrong_donors": donors,
        "wrong_donor_audit": {
            "requests": len(donors),
            "coverage_fraction": len(donors) / len(indices),
            "same_length_bin_fraction": sum(int(row["same_bin"]) for row in donors.values()) / len(donors),
            "same_user_assignments": 0,
        },
        "outcome_isolation": isolation,
        "prior_feature_indices_sha256": {
            str(path): sha256_file(path) for path in prior_feature_index_paths
        },
        "label_access": {
            "records_train_blind_opened": True,
            "records_train_labels_opened": False,
            "dev_test_records_labels_qrels_opened": False,
        },
        "authorization": {
            "feature_roles": ["internal_A"],
            "training": False,
            "internal_A_labels_after_A0": True,
            "dev_test": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c38-selection", required=True)
    parser.add_argument("--prior-feature-indices", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = build(
        c38_selection_path=Path(args.c38_selection),
        prior_feature_index_paths=[Path(value) for value in args.prior_feature_indices],
    )
    write_json(args.output, value)
    print(json.dumps(value["outcome_isolation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
