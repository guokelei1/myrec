"""Subselect C26's untouched role partition into a C27-owned selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.structure import (  # noqa: E402
    ROLE_COUNTS,
    PackedStructure,
    candidate_key_sha256,
    donor_key_sha256,
    load_config,
    read_json,
    sha256_file,
    stable_key,
    write_json_once,
)


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    for name, expected_name in (
        ("c26_selection", "c26_selection_sha256"),
        ("c26_g0_report", "c26_g0_report_sha256"),
        ("c26_train_report", "c26_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C27 registered source changed: {name}")
    source = read_json(paths["c26_selection"])
    outcome = read_json(paths["c26_train_report"])
    if outcome.get("internal_A_labels_opened") is not False or outcome.get(
        "delayed_B_labels_opened"
    ) is not False:
        raise PermissionError("C27 source A/B labels are not untouched")
    data = PackedStructure(paths["packed_train_root"])
    seed = int(config["selection"]["seed"])
    roles: dict[str, list[int]] = {}
    selected_positions: dict[str, list[int]] = {}
    for role, count in ROLE_COUNTS.items():
        row = source["roles"][role]
        ordered = sorted(
            range(len(row["indices"])),
            key=lambda position: (
                stable_key(seed, role, str(row["request_ids"][position])),
                int(row["indices"][position]),
            ),
        )
        positions = ordered[:count]
        selected_positions[role] = positions
        roles[role] = [int(row["indices"][position]) for position in positions]
    donors = {
        role: [
            int(source["wrong_history_donors"][role]["indices"][position])
            for position in selected_positions[role]
        ]
        for role in ("fit", "internal_A", "delayed_B")
    }
    if len({index for values in roles.values() for index in values}) != sum(
        len(values) for values in roles.values()
    ):
        raise AssertionError("C27 roles are not disjoint")
    for role, values in donors.items():
        if any(
            not set(int(value) for value in data.candidate_indices(recipient)).isdisjoint(
                int(value) for value in data.history_indices(donor)
            )
            for recipient, donor in zip(roles[role], values)
        ):
            raise AssertionError("C27 donor repeats recipient candidate")
    result = {
        "candidate_id": "c27",
        "selection_id": "c27_evidence_contest_selection_v1",
        "status": "frozen_before_any_c27_label_or_outcome",
        "seed": seed,
        "roles": {
            role: {
                "indices": values,
                "request_ids": [data.request_ids[index] for index in values],
                "candidate_key_sha256": candidate_key_sha256(data, values),
            }
            for role, values in roles.items()
        },
        "wrong_history_donors": {
            role: {
                "indices": values,
                "request_ids": [data.request_ids[index] for index in values],
                "mapping_sha256": donor_key_sha256(data, roles[role], values),
            }
            for role, values in donors.items()
        },
        "donor_matching": source["donor_matching"],
        "sources": {
            "c26_selection_sha256": paths["c26_selection_sha256"],
            "c26_g0_report_sha256": paths["c26_g0_report_sha256"],
            "c26_train_report_sha256": paths["c26_train_report_sha256"],
        },
        "checks": {
            "c26_fit_labels_previously_opened": True,
            "c26_internal_A_labels_opened": False,
            "c26_delayed_B_labels_opened": False,
            "c27_internal_A_labels_opened": False,
            "c27_delayed_B_labels_opened": False,
            "roles_pairwise_disjoint": True,
            "donor_candidate_overlap_zero": True,
            "dev_test_qrels_metrics_read": False,
        },
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
                "role_counts": {
                    name: len(row["indices"]) for name, row in result["roles"].items()
                },
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
