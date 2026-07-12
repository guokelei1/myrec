"""Materialize the label-free C41 role remapping from untouched C38 roles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def build(
    *,
    c38_selection_path: Path,
    c39_selection_path: Path,
    c38_feature_indices_path: Path,
) -> dict[str, Any]:
    c38 = read_json(c38_selection_path)
    c39 = read_json(c39_selection_path)
    c38_featured = set(
        int(value)
        for value in np.load(c38_feature_indices_path, allow_pickle=False)
    )
    role_source = {
        "fit": "fit",
        "internal_A": "delayed_B",
        "delayed_B": "escrow",
    }
    roles = {target: c38["roles"][source] for target, source in role_source.items()}
    fit = set(int(value) for value in roles["fit"]["indices"])
    internal_a = set(int(value) for value in roles["internal_A"]["indices"])
    delayed_b = set(int(value) for value in roles["delayed_B"]["indices"])
    c38_a = set(int(value) for value in c38["roles"]["internal_A"]["indices"])
    c39_a = set(int(value) for value in c39["roles"]["internal_A"]["indices"])
    if fit & internal_a or fit & delayed_b or internal_a & delayed_b:
        raise ValueError("C41 roles overlap")
    selected = fit | internal_a | delayed_b
    donors = {
        str(index): c38["wrong_donors"][str(index)] for index in sorted(selected)
    }
    same_bin = sum(int(row["same_bin"]) for row in donors.values())
    isolation = {
        "internal_A_from_c38_delayed_B": len(internal_a),
        "internal_A_overlap_c38_internal_A": len(internal_a & c38_a),
        "internal_A_overlap_c39_internal_A": len(internal_a & c39_a),
        "internal_A_overlap_c38_feature_materialized": len(
            internal_a & c38_featured
        ),
        "delayed_B_overlap_any_prior_feature_materialized": len(
            delayed_b & c38_featured
        ),
        "fit_exactly_c38_fit": int(fit == set(c38["roles"]["fit"]["indices"])),
    }
    if isolation != {
        "internal_A_from_c38_delayed_B": 1200,
        "internal_A_overlap_c38_internal_A": 0,
        "internal_A_overlap_c39_internal_A": 0,
        "internal_A_overlap_c38_feature_materialized": 0,
        "delayed_B_overlap_any_prior_feature_materialized": 0,
        "fit_exactly_c38_fit": 1,
    }:
        raise RuntimeError(f"C41 isolation failed: {isolation}")
    return {
        "candidate_id": "c41",
        "seed": 20262100,
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
        "roles": roles,
        "wrong_donors": donors,
        "wrong_donor_audit": {
            "requests": len(donors),
            "coverage_fraction": len(donors) / len(selected),
            "same_length_bin_fraction": same_bin / len(donors),
            "same_user_assignments": 0,
        },
        "outcome_isolation": isolation,
        "label_access": {
            "records_train_blind_opened": True,
            "records_train_labels_opened": False,
            "dev_test_records_labels_qrels_opened": False,
        },
        "authorization": {
            "feature_roles": ["fit", "internal_A"],
            "fit_labels_after_proposal_lock": True,
            "internal_A_labels_after_A0": True,
            "delayed_B_features_scores_labels": False,
            "dev_test": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c38-selection", required=True)
    parser.add_argument("--c39-selection", required=True)
    parser.add_argument("--c38-feature-indices", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = build(
        c38_selection_path=Path(args.c38_selection),
        c39_selection_path=Path(args.c39_selection),
        c38_feature_indices_path=Path(args.c38_feature_indices),
    )
    write_json(args.output, value)
    print(json.dumps(value["outcome_isolation"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
