"""Freeze C32 roles before any C32-A feature, score, or label access."""

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
    return next((index for index, edge in enumerate(edges) if length <= edge), len(edges))


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    seed = int(config["selection"]["seed"])
    for name, expected in (
        ("c31_selection", "c31_selection_sha256"),
        ("c31_g0_report", "c31_g0_report_sha256"),
        ("c31_train_report", "c31_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("label_free_request_manifest", "label_free_request_manifest_sha256"),
        ("schema_incident_report", "schema_incident_report_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected]:
            raise RuntimeError(f"C32 source changed: {name}")
    source = read_json(paths["c31_selection"])
    g0 = read_json(paths["c31_g0_report"])
    c31 = read_json(paths["c31_train_report"])
    if g0.get("delayed_B_features_labels_scores_opened") is not False:
        raise PermissionError("C32 source delayed-B was materialized")
    if c31.get("status") != "failed_A1_terminal":
        raise PermissionError("C32 C31 terminal state differs")
    if c31.get("internal_A_labels_opened") is not True:
        raise PermissionError("C32 C31 diagnostic state differs")
    if c31.get("delayed_B_features_labels_scores_opened") is not False:
        raise PermissionError("C32 source A/B boundary differs")

    data = PackedStructure(paths["packed_train_root"])
    users = load_user_ids(paths["label_free_request_metadata"], data)
    nonrepeat = [
        index
        for index in range(len(data.request_ids))
        if data.history_count(index) > 0 and data.repeat_candidate_count(index) == 0
    ]
    prior_roles = {
        role: [int(value) for value in row["indices"]]
        for role, row in source["roles"].items()
    }
    prior_donors = {
        int(value)
        for row in source["wrong_history_donors"].values()
        for value in row["indices"]
    }
    prior_footprint = {value for rows in prior_roles.values() for value in rows} | prior_donors
    roles = {
        "fit": prior_roles["fit"],
        "internal_A": prior_roles["delayed_B"],
        "delayed_B": prior_roles["escrow"],
        "structural_repeat": prior_roles["structural_repeat"],
        "structural_nohistory": prior_roles["structural_nohistory"],
    }
    new_pool = [index for index in nonrepeat if index not in prior_footprint]
    ordered = sorted(
        new_pool,
        key=lambda index: (stable_key(seed, "escrow", data.request_ids[index]), index),
    )
    roles["escrow"] = ordered[: ROLE_COUNTS["escrow"]]
    if {name: len(values) for name, values in roles.items()} != ROLE_COUNTS:
        raise AssertionError("C32 role counts differ")
    flat = [value for values in roles.values() for value in values]
    if len(flat) != len(set(flat)):
        raise AssertionError("C32 roles overlap")

    outcome = set(flat)
    reserve = [
        index
        for index in nonrepeat
        if index not in outcome and index not in prior_footprint
    ]
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
        for step in range(len(candidates)):
            donor = int(candidates[(start + step) % len(candidates)])
            if users[donor] != users[recipient] and recipient_candidates.isdisjoint(
                int(value) for value in data.history_indices(donor)
            ):
                return donor
        raise RuntimeError("C32 donor unavailable")

    donors = {
        "fit": [int(value) for value in source["wrong_history_donors"]["fit"]["indices"]],
        "internal_A": [
            int(value) for value in source["wrong_history_donors"]["delayed_B"]["indices"]
        ],
        "delayed_B": [donor_for(index) for index in roles["delayed_B"]],
    }
    if any(donor in outcome for values in donors.values() for donor in values):
        raise AssertionError("C32 donor intersects roles")
    for role, values in donors.items():
        for recipient, donor in zip(roles[role], values):
            if users[recipient] == users[donor] or not set(
                int(value) for value in data.candidate_indices(recipient)
            ).isdisjoint(int(value) for value in data.history_indices(donor)):
                raise AssertionError("C32 donor contract differs")

    result = {
        "candidate_id": "c32",
        "selection_id": "c32_authenticated_tangent_query_transport_selection_v1",
        "status": "frozen_before_any_c32_A_feature_score_or_label",
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
        "donor_matching": {
            "history_length_edges": edges,
            "timestamp_quantiles": quantiles,
            "same_user_forbidden": True,
            "recipient_candidate_overlap_forbidden": True,
            "reserve_requests": len(reserve),
        },
        "sources": {
            "c31_selection_sha256": paths["c31_selection_sha256"],
            "c31_g0_report_sha256": paths["c31_g0_report_sha256"],
            "c31_train_report_sha256": paths["c31_train_report_sha256"],
        },
        "checks": {
            "fit_labels_previously_opened": True,
            "c31_A_labels_previously_opened_but_not_reused": True,
            "c32_internal_A_features_scores_labels_opened": False,
            "c32_delayed_B_features_scores_labels_opened": False,
            "roles_pairwise_disjoint": True,
            "strict_nonrepeat_fit_A_B_escrow": True,
            "donor_candidate_overlap_zero": True,
            "donor_user_overlap_zero": True,
            "selection_label_access": False,
            "c32_code_dev_test_qrels_metrics_read": False,
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
                    key: len(value["indices"]) for key, value in result["roles"].items()
                },
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
