"""Freeze C29 outcome roles and distinct-user wrong-history donors without labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


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


def ordered_subset(
    data: PackedStructure, values: list[int], *, count: int, seed: int, role: str
) -> list[int]:
    ordered = sorted(
        (int(value) for value in values),
        key=lambda index: (stable_key(seed, role, data.request_ids[index]), index),
    )
    if len(ordered) < count:
        raise ValueError(f"C29 source role too small: {role} ({len(ordered)} < {count})")
    return ordered[:count]


def length_bin(length: int, edges: list[int]) -> int:
    return next((position for position, edge in enumerate(edges) if length <= edge), len(edges))


def load_label_free_users(
    path: str | Path, data: PackedStructure
) -> tuple[list[str], dict[str, Any]]:
    positions = {request_id: index for index, request_id in enumerate(data.request_ids)}
    users: list[str | None] = [None] * len(data.request_ids)
    seen = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            request_id = str(row["request_id"])
            index = positions.get(request_id)
            if index is None:
                continue
            if users[index] is not None:
                raise ValueError(f"duplicate C29 metadata request: {request_id}")
            if str(row.get("split")) != "train":
                raise ValueError("C29 packed request is not train")
            if int(row["time_index"]) != int(data.timestamps[index]):
                raise ValueError("C29 metadata timestamp differs")
            if int(row["candidate_count"]) != len(data.candidate_indices(index)):
                raise ValueError("C29 metadata candidate count differs")
            users[index] = str(row["user_id"])
            seen += 1
    if seen != len(data.request_ids) or any(value is None for value in users):
        raise ValueError(f"C29 metadata coverage differs: {seen}/{len(data.request_ids)}")
    typed = [str(value) for value in users]
    return typed, {
        "packed_requests": len(typed),
        "unique_users": len(set(typed)),
        "train_only": True,
        "timestamp_exact": True,
        "candidate_count_exact": True,
    }


def materialize(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths, seed = config["paths"], int(config["selection"]["seed"])
    for name, expected_name in (
        ("c28_selection", "c28_selection_sha256"),
        ("c28_g0_report", "c28_g0_report_sha256"),
        ("c28_train_report", "c28_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("label_free_request_manifest", "label_free_request_manifest_sha256"),
        ("schema_incident_report", "schema_incident_report_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C29 registered source changed: {name}")

    c28 = read_json(paths["c28_selection"])
    c28_g0 = read_json(paths["c28_g0_report"])
    c28_outcome = read_json(paths["c28_train_report"])
    if c28_outcome.get("internal_A_labels_opened") is not True:
        raise PermissionError("C29 inherited C28 internal-A label state differs")
    if c28_outcome.get("delayed_B_labels_opened") is not False:
        raise PermissionError("C29 source delayed-B labels are not closed")
    if c28_g0.get("escrow_features_or_labels_opened") is not False:
        raise PermissionError("C29 source escrow was materialized")

    data = PackedStructure(paths["packed_train_root"])
    users, metadata_audit = load_label_free_users(paths["label_free_request_metadata"], data)
    nonrepeat: list[int] = []
    repeat: list[int] = []
    nohistory: list[int] = []
    for index in range(len(data.request_ids)):
        if data.history_count(index) == 0:
            nohistory.append(index)
        elif data.repeat_candidate_count(index) == 0:
            nonrepeat.append(index)
        else:
            repeat.append(index)
    if (len(nonrepeat), len(repeat), len(nohistory)) != (29_277, 25_122, 42_540):
        raise ValueError("C29 structural pools changed")

    prior_roles = {
        role: [int(value) for value in row["indices"]] for role, row in c28["roles"].items()
    }
    prior_donors = {
        int(value)
        for row in c28["wrong_history_donors"].values()
        for value in row["indices"]
    }
    prior_footprint = {
        int(value) for values in prior_roles.values() for value in values
    } | prior_donors

    inherited_fit = prior_roles["fit"] + prior_roles["internal_A"]
    if len(inherited_fit) != int(config["selection"]["inherited_open_fit_requests"]):
        raise ValueError("C29 inherited fit size differs")
    if len(set(inherited_fit)) != len(inherited_fit):
        raise AssertionError("C29 inherited fit overlaps")
    internal_A = list(prior_roles["escrow"])

    new_nonrepeat = [index for index in nonrepeat if index not in prior_footprint]
    ordered_new = sorted(
        new_nonrepeat,
        key=lambda index: (
            stable_key(seed, "new_outcome_partition", data.request_ids[index]),
            index,
        ),
    )
    fit_add_count = ROLE_COUNTS["fit"] - len(inherited_fit)
    needed = fit_add_count + ROLE_COUNTS["delayed_B"] + ROLE_COUNTS["escrow"]
    if len(ordered_new) < needed:
        raise ValueError("C29 new strict-nonrepeat pool too small")
    cursor = 0
    fit_added = ordered_new[cursor : cursor + fit_add_count]
    cursor += fit_add_count
    delayed_B = ordered_new[cursor : cursor + ROLE_COUNTS["delayed_B"]]
    cursor += ROLE_COUNTS["delayed_B"]
    escrow = ordered_new[cursor : cursor + ROLE_COUNTS["escrow"]]
    fit = inherited_fit + fit_added

    structural_repeat = ordered_subset(
        data,
        [index for index in repeat if index not in prior_footprint],
        count=ROLE_COUNTS["structural_repeat"],
        seed=seed,
        role="structural_repeat",
    )
    structural_nohistory = ordered_subset(
        data,
        [index for index in nohistory if index not in prior_footprint],
        count=ROLE_COUNTS["structural_nohistory"],
        seed=seed,
        role="structural_nohistory",
    )
    roles = {
        "fit": fit,
        "internal_A": internal_A,
        "delayed_B": delayed_B,
        "escrow": escrow,
        "structural_repeat": structural_repeat,
        "structural_nohistory": structural_nohistory,
    }
    flat = [index for values in roles.values() for index in values]
    if len(flat) != len(set(flat)):
        raise AssertionError("C29 roles overlap")
    if {name: len(values) for name, values in roles.items()} != ROLE_COUNTS:
        raise AssertionError("C29 role counts differ")

    outcome_set = set(flat)
    reserve = [
        index
        for index in nonrepeat
        if index not in outcome_set and index not in prior_footprint
    ]
    edges = [int(value) for value in config["selection"]["donor_length_bins"]]
    quantiles = int(config["selection"]["donor_time_quantiles"])
    time_edges = np.quantile(
        np.asarray([data.timestamps[index] for index in reserve], dtype=np.float64),
        np.linspace(0.0, 1.0, quantiles + 1)[1:-1],
    )

    def bucket(index: int) -> tuple[int, int]:
        return (
            length_bin(data.history_count(index), edges),
            int(np.searchsorted(time_edges, float(data.timestamps[index]), side="right")),
        )

    grouped: dict[tuple[int, int], list[int]] = {}
    length_grouped: dict[int, list[int]] = {}
    for index in reserve:
        grouped.setdefault(bucket(index), []).append(index)
        length_grouped.setdefault(bucket(index)[0], []).append(index)
    for group, values in grouped.items():
        values.sort(
            key=lambda index: (stable_key(seed, f"donor:{group}", data.request_ids[index]), index)
        )
    for group, values in length_grouped.items():
        values.sort(
            key=lambda index: (
                stable_key(seed, f"donor_length:{group}", data.request_ids[index]),
                index,
            )
        )
    reserve.sort(
        key=lambda index: (stable_key(seed, "donor_fallback", data.request_ids[index]), index)
    )

    def donor_for(recipient: int) -> int:
        candidates = grouped.get(bucket(recipient), [])
        if not candidates:
            candidates = length_grouped.get(bucket(recipient)[0], [])
        if not candidates:
            candidates = reserve
        start = int.from_bytes(
            stable_key(seed, "donor_start", data.request_ids[recipient])[0][:8], "big"
        ) % len(candidates)
        recipient_candidates = set(int(value) for value in data.candidate_indices(recipient))
        for step in range(len(candidates)):
            donor = int(candidates[(start + step) % len(candidates)])
            if (
                donor != recipient
                and users[donor] != users[recipient]
                and recipient_candidates.isdisjoint(
                    int(value) for value in data.history_indices(donor)
                )
            ):
                return donor
        raise RuntimeError(f"C29 donor unavailable: {recipient}")

    donors = {
        role: [donor_for(index) for index in roles[role]]
        for role in ("fit", "internal_A", "delayed_B")
    }
    if any(donor in outcome_set or donor in prior_footprint for values in donors.values() for donor in values):
        raise AssertionError("C29 donor intersects frozen outcome/prior footprint")
    if any(
        users[recipient] == users[donor]
        for role, values in donors.items()
        for recipient, donor in zip(roles[role], values)
    ):
        raise AssertionError("C29 wrong donor shares recipient user")

    result = {
        "candidate_id": "c29",
        "selection_id": "c29_causally_authenticated_mediation_selection_v1",
        "status": "frozen_before_any_c29_feature_label_score_or_outcome",
        "seed": seed,
        "pool_counts": {
            "strict_nonrepeat_history_present": len(nonrepeat),
            "repeat_present": len(repeat),
            "nohistory": len(nohistory),
            "prior_footprint": len(prior_footprint),
            "new_strict_nonrepeat_available": len(new_nonrepeat),
            "donor_reserve": len(reserve),
        },
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
        "fit_composition": {
            "c28_fit": len(prior_roles["fit"]),
            "c28_internal_A": len(prior_roles["internal_A"]),
            "new_label_free_reserve": len(fit_added),
            "authentication_present_filter": False,
            "query_or_category_filter": False,
        },
        "donor_matching": {
            "history_length_edges": edges,
            "timestamp_quantiles": quantiles,
            "recipient_candidate_overlap_forbidden": True,
            "same_user_forbidden": True,
            "outcome_role_donors_forbidden": True,
            "prior_footprint_donors_forbidden": True,
            "reserve_requests": len(reserve),
        },
        "metadata_audit": metadata_audit,
        "sources": {
            "c28_selection_sha256": paths["c28_selection_sha256"],
            "c28_g0_report_sha256": paths["c28_g0_report_sha256"],
            "c28_train_report_sha256": paths["c28_train_report_sha256"],
            "label_free_request_metadata_sha256": paths["label_free_request_metadata_sha256"],
            "schema_incident_report_sha256": paths["schema_incident_report_sha256"],
        },
        "checks": {
            "c28_fit_labels_previously_opened": True,
            "c28_internal_A_labels_previously_opened": True,
            "c28_delayed_B_labels_opened": False,
            "c28_escrow_features_or_labels_opened": False,
            "c29_internal_A_features_labels_scores_opened": False,
            "c29_delayed_B_features_labels_scores_opened": False,
            "roles_pairwise_disjoint": True,
            "strict_nonrepeat_fit_A_B_escrow": True,
            "donors_outside_outcome_and_prior_roles": True,
            "donor_candidate_overlap_zero": True,
            "donor_user_overlap_zero": True,
            "selection_label_access": False,
            "c29_code_dev_test_qrels_metrics_read": False,
            "global_schema_inspection_incident_registered": True,
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
                "pool_counts": result["pool_counts"],
                "fit_composition": result["fit_composition"],
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
