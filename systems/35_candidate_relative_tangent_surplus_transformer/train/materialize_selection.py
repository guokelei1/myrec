"""Freeze C35 roles from C34's untouched reserves plus a fresh escrow."""

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


def target_roles(selection: dict) -> set[int]:
    return {
        int(value)
        for row in selection["roles"].values()
        for value in row["indices"]
    }


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    seed = int(config["selection"]["seed"])
    for name, expected in (
        ("c34_selection", "c34_selection_sha256"),
        ("c34_g0_report", "c34_g0_report_sha256"),
        ("c34_train_report", "c34_train_report_sha256"),
        ("c33_selection", "c33_selection_sha256"),
        ("c32_selection", "c32_selection_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("label_free_request_manifest", "label_free_request_manifest_sha256"),
        ("schema_incident_report", "schema_incident_report_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected]:
            raise RuntimeError(f"C35 source changed: {name}")

    c34_selection = read_json(paths["c34_selection"])
    c33_selection = read_json(paths["c33_selection"])
    c32_selection = read_json(paths["c32_selection"])
    c34_g0 = read_json(paths["c34_g0_report"])
    c34 = read_json(paths["c34_train_report"])
    if c34.get("status") != "failed_A0_terminal" or c34.get(
        "internal_A_labels_opened"
    ) is not False:
        raise PermissionError("C35 C34 terminal state differs")
    if c34_g0.get("delayed_B_features_labels_scores_opened") is not False or c34.get(
        "delayed_B_features_labels_scores_opened"
    ) is not False:
        raise PermissionError("C35 C34 reserved boundary differs")

    data = PackedStructure(paths["packed_train_root"])
    users = load_user_ids(paths["label_free_request_metadata"], data)
    prior_targets = (
        target_roles(c32_selection)
        | target_roles(c33_selection)
        | target_roles(c34_selection)
    )
    c34_roles = {
        role: [int(value) for value in row["indices"]]
        for role, row in c34_selection["roles"].items()
    }
    roles = {
        "fit": c34_roles["fit"],
        "internal_A": c34_roles["delayed_B"],
        "delayed_B": c34_roles["escrow"],
    }
    used = {value for values in roles.values() for value in values}

    strict_pool = [
        index
        for index in range(len(data.request_ids))
        if index not in prior_targets
        and data.history_count(index) > 0
        and data.repeat_candidate_count(index) == 0
    ]
    repeat_pool = [
        index
        for index in range(len(data.request_ids))
        if index not in prior_targets and data.repeat_candidate_count(index) > 0
    ]
    nohistory_pool = [
        index
        for index in range(len(data.request_ids))
        if index not in prior_targets and data.history_count(index) == 0
    ]

    def take(pool: list[int], role: str, count: int, excluded: set[int]) -> list[int]:
        ordered = sorted(
            (index for index in pool if index not in excluded),
            key=lambda index: (stable_key(seed, role, data.request_ids[index]), index),
        )
        if len(ordered) < count:
            raise RuntimeError(f"C35 insufficient label-free pool for {role}")
        return ordered[:count]

    roles["escrow"] = take(strict_pool, "escrow", ROLE_COUNTS["escrow"], used)
    used.update(roles["escrow"])
    roles["structural_repeat"] = take(
        repeat_pool,
        "structural_repeat",
        ROLE_COUNTS["structural_repeat"],
        used,
    )
    used.update(roles["structural_repeat"])
    roles["structural_nohistory"] = take(
        nohistory_pool,
        "structural_nohistory",
        ROLE_COUNTS["structural_nohistory"],
        used,
    )
    if {name: len(values) for name, values in roles.items()} != ROLE_COUNTS:
        raise AssertionError("C35 role counts differ")
    flat = [value for values in roles.values() for value in values]
    if len(flat) != len(set(flat)):
        raise AssertionError("C35 roles overlap")

    outcome = set(flat)
    donor_reserve = [index for index in strict_pool if index not in outcome]
    edges = [int(value) for value in config["selection"]["donor_length_bins"]]
    quantiles = int(config["selection"]["donor_time_quantiles"])
    time_edges = np.quantile(
        np.asarray(data.timestamps[donor_reserve], dtype=np.float64),
        np.linspace(0, 1, quantiles + 1)[1:-1],
    )

    def bucket(index: int) -> tuple[int, int]:
        return (
            length_bin(data.history_count(index), edges),
            int(np.searchsorted(time_edges, float(data.timestamps[index]), side="right")),
        )

    grouped: dict[tuple[int, int], list[int]] = {}
    by_length: dict[int, list[int]] = {}
    for index in donor_reserve:
        grouped.setdefault(bucket(index), []).append(index)
        by_length.setdefault(bucket(index)[0], []).append(index)
    for key, values in grouped.items():
        values.sort(key=lambda index: (stable_key(seed, f"donor:{key}", data.request_ids[index]), index))
    for key, values in by_length.items():
        values.sort(
            key=lambda index: (stable_key(seed, f"donor_length:{key}", data.request_ids[index]), index)
        )
    donor_reserve.sort(
        key=lambda index: (stable_key(seed, "donor_fallback", data.request_ids[index]), index)
    )

    def donor_for(recipient: int) -> int:
        candidates = grouped.get(bucket(recipient), []) or by_length.get(bucket(recipient)[0], []) or donor_reserve
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
        raise RuntimeError("C35 donor unavailable")

    donors = {
        role: [donor_for(index) for index in roles[role]]
        for role in ("fit", "internal_A", "delayed_B")
    }
    if any(donor in outcome or donor in prior_targets for values in donors.values() for donor in values):
        raise AssertionError("C35 donor intersects a target role")
    for role, values in donors.items():
        for recipient, donor in zip(roles[role], values):
            if users[recipient] == users[donor] or not set(
                int(value) for value in data.candidate_indices(recipient)
            ).isdisjoint(int(value) for value in data.history_indices(donor)):
                raise AssertionError("C35 donor contract differs")

    result = {
        "candidate_id": "c35",
        "selection_id": "c35_candidate_relative_surplus_selection_v1",
        "status": "frozen_before_any_c35_A_feature_score_or_label",
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
            "c34_selection_sha256": paths["c34_selection_sha256"],
            "c34_g0_report_sha256": paths["c34_g0_report_sha256"],
            "c34_train_report_sha256": paths["c34_train_report_sha256"],
            "c33_selection_sha256": paths["c33_selection_sha256"],
            "c32_selection_sha256": paths["c32_selection_sha256"],
        },
        "checks": {
            "c34_fit_reused_with_labels_previously_opened": True,
            "c34_A_features_scores_opened_but_not_reused": True,
            "c34_A_labels_opened": False,
            "c34_delayed_B_promoted_to_c35_A_unopened": True,
            "c34_escrow_promoted_to_c35_B_unopened": True,
            "c35_internal_A_features_scores_labels_opened": False,
            "c35_delayed_B_features_scores_labels_opened": False,
            "roles_pairwise_disjoint": True,
            "strict_nonrepeat_fit_A_B_escrow": True,
            "fresh_c35_escrow_and_structural_roles": True,
            "donor_candidate_overlap_zero": True,
            "donor_user_overlap_zero": True,
            "selection_label_access": False,
            "c35_code_dev_test_qrels_metrics_read": False,
        },
        "donor_matching": {
            "history_length_edges": edges,
            "timestamp_quantiles": quantiles,
            "same_user_forbidden": True,
            "recipient_candidate_overlap_forbidden": True,
            "reserve_requests": len(donor_reserve),
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
