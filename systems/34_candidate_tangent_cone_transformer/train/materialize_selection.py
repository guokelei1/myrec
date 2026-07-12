"""Freeze wholly fresh C34 fit/outcome roles without label access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.authentication import load_user_ids  # noqa: E402
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


def length_bin(length: int, edges: list[int]) -> int:
    return next((position for position, edge in enumerate(edges) if length <= edge), len(edges))


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    seed = int(config["selection"]["seed"])
    for name, expected in (
        ("c33_selection", "c33_selection_sha256"),
        ("c33_g0_report", "c33_g0_report_sha256"),
        ("c33_train_report", "c33_train_report_sha256"),
        ("c32_selection", "c32_selection_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("label_free_request_manifest", "label_free_request_manifest_sha256"),
        ("schema_incident_report", "schema_incident_report_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected]:
            raise RuntimeError(f"C34 source changed: {name}")

    c33_selection = read_json(paths["c33_selection"])
    c32_selection = read_json(paths["c32_selection"])
    c33_g0 = read_json(paths["c33_g0_report"])
    c33 = read_json(paths["c33_train_report"])
    if c33.get("status") != "failed_A1_terminal" or c33.get(
        "internal_A_labels_opened"
    ) is not True:
        raise PermissionError("C34 C33 terminal state differs")
    if c33_g0.get("delayed_B_features_labels_scores_opened") is not False or c33.get(
        "delayed_B_features_labels_scores_opened"
    ) is not False:
        raise PermissionError("C34 C33 reserved boundary differs")

    data = PackedStructure(paths["packed_train_root"])
    users = load_user_ids(paths["label_free_request_metadata"], data)
    prior_roles = {
        int(value)
        for selection in (c32_selection, c33_selection)
        for row in selection["roles"].values()
        for value in row["indices"]
    }
    prior_donors = {
        int(value)
        for selection in (c32_selection, c33_selection)
        for row in selection["wrong_history_donors"].values()
        for value in row["indices"]
    }

    strict_nonrepeat = [
        index
        for index in range(len(data.request_ids))
        if index not in prior_roles
        and data.history_count(index) > 0
        and data.repeat_candidate_count(index) == 0
    ]
    repeat_pool = [
        index
        for index in range(len(data.request_ids))
        if index not in prior_roles and data.repeat_candidate_count(index) > 0
    ]
    nohistory_pool = [
        index
        for index in range(len(data.request_ids))
        if index not in prior_roles and data.history_count(index) == 0
    ]

    def take(pool: list[int], role: str, count: int, excluded: set[int]) -> list[int]:
        ordered = sorted(
            (index for index in pool if index not in excluded),
            key=lambda index: (stable_key(seed, role, data.request_ids[index]), index),
        )
        if len(ordered) < count:
            raise RuntimeError(f"C34 insufficient label-free pool for {role}")
        return ordered[:count]

    used: set[int] = set()
    roles: dict[str, list[int]] = {}
    for role, pool in (
        ("fit", strict_nonrepeat),
        ("internal_A", strict_nonrepeat),
        ("delayed_B", strict_nonrepeat),
        ("escrow", strict_nonrepeat),
        ("structural_repeat", repeat_pool),
        ("structural_nohistory", nohistory_pool),
    ):
        values = take(pool, role, ROLE_COUNTS[role], used)
        roles[role] = values
        used.update(values)

    if {name: len(values) for name, values in roles.items()} != ROLE_COUNTS:
        raise AssertionError("C34 role counts differ")
    flat = [value for values in roles.values() for value in values]
    if len(flat) != len(set(flat)) or set(flat) & prior_roles:
        raise AssertionError("C34 roles overlap or reuse a prior target role")

    outcome = set(flat)
    reserve = [index for index in strict_nonrepeat if index not in outcome]
    edges = [int(value) for value in config["selection"]["donor_length_bins"]]
    quantiles = int(config["selection"]["donor_time_quantiles"])
    time_edges = np.quantile(
        np.asarray(data.timestamps[reserve], dtype=np.float64),
        np.linspace(0, 1, quantiles + 1)[1:-1],
    )

    def bucket(index: int) -> tuple[int, int]:
        return (
            length_bin(data.history_count(index), edges),
            int(np.searchsorted(time_edges, float(data.timestamps[index]), side="right")),
        )

    grouped: dict[tuple[int, int], list[int]] = {}
    by_length: dict[int, list[int]] = {}
    for index in reserve:
        grouped.setdefault(bucket(index), []).append(index)
        by_length.setdefault(bucket(index)[0], []).append(index)
    for key, values in grouped.items():
        values.sort(key=lambda index: (stable_key(seed, f"donor:{key}", data.request_ids[index]), index))
    for key, values in by_length.items():
        values.sort(
            key=lambda index: (stable_key(seed, f"donor_length:{key}", data.request_ids[index]), index)
        )
    reserve.sort(key=lambda index: (stable_key(seed, "donor_fallback", data.request_ids[index]), index))

    def donor_for(recipient: int) -> int:
        candidates = grouped.get(bucket(recipient), []) or by_length.get(bucket(recipient)[0], []) or reserve
        start = int.from_bytes(
            stable_key(seed, "donor_start", data.request_ids[recipient])[0][:8], "big"
        ) % len(candidates)
        recipient_candidates = set(int(value) for value in data.candidate_indices(recipient))
        for offset in range(len(candidates)):
            donor = int(candidates[(start + offset) % len(candidates)])
            if users[donor] != users[recipient] and recipient_candidates.isdisjoint(
                int(value) for value in data.history_indices(donor)
            ):
                return donor
        raise RuntimeError("C34 donor unavailable")

    donors = {
        role: [donor_for(index) for index in roles[role]]
        for role in ("fit", "internal_A", "delayed_B")
    }
    if any(donor in outcome or donor in prior_roles for values in donors.values() for donor in values):
        raise AssertionError("C34 donor intersects a C34 or prior target role")
    for role, values in donors.items():
        for recipient, donor in zip(roles[role], values):
            if users[recipient] == users[donor] or not set(
                int(value) for value in data.candidate_indices(recipient)
            ).isdisjoint(int(value) for value in data.history_indices(donor)):
                raise AssertionError("C34 donor contract differs")

    result = {
        "candidate_id": "c34",
        "selection_id": "c34_fresh_candidate_tangent_cone_selection_v1",
        "status": "frozen_before_any_c34_feature_score_or_label",
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
        "sources": {
            "c33_selection_sha256": paths["c33_selection_sha256"],
            "c33_g0_report_sha256": paths["c33_g0_report_sha256"],
            "c33_train_report_sha256": paths["c33_train_report_sha256"],
            "c32_selection_sha256": paths["c32_selection_sha256"],
        },
        "checks": {
            "fresh_fit_not_reused": True,
            "c33_A_labels_previously_opened_but_not_reused": True,
            "c33_delayed_B_and_escrow_not_reused": True,
            "c32_c33_target_roles_not_reused": True,
            "prior_wrong_donors_may_become_targets_without_prior_target_exposure": bool(
                set(flat) & prior_donors
            ),
            "prior_wrong_donors_may_be_reused_as_donors": bool(
                set(value for values in donors.values() for value in values) & prior_donors
            ),
            "c34_internal_A_features_scores_labels_opened": False,
            "c34_delayed_B_features_scores_labels_opened": False,
            "roles_pairwise_disjoint": True,
            "strict_nonrepeat_fit_A_B_escrow": True,
            "donor_candidate_overlap_zero": True,
            "donor_user_overlap_zero": True,
            "selection_label_access": False,
            "c34_code_dev_test_qrels_metrics_read": False,
        },
        "donor_matching": {
            "history_length_edges": edges,
            "timestamp_quantiles": quantiles,
            "same_user_forbidden": True,
            "recipient_candidate_overlap_forbidden": True,
            "reserve_requests": len(reserve),
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
