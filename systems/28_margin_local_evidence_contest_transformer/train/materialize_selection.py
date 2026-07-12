"""Build outcome-isolated C28 roles and matched wrong-history donors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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
        raise ValueError(f"C28 source role too small: {role}")
    return ordered[:count]


def length_bin(length: int, edges: list[int]) -> int:
    return next((position for position, edge in enumerate(edges) if length <= edge), len(edges))


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths, seed = config["paths"], int(config["selection"]["seed"])
    for name, expected_name in (
        ("c26_selection", "c26_selection_sha256"),
        ("c27_selection", "c27_selection_sha256"),
        ("c27_g0_report", "c27_g0_report_sha256"),
        ("c27_train_report", "c27_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C28 registered source changed: {name}")
    c26 = read_json(paths["c26_selection"])
    c27 = read_json(paths["c27_selection"])
    c27_g0 = read_json(paths["c27_g0_report"])
    c27_outcome = read_json(paths["c27_train_report"])
    if c27_outcome.get("internal_A_labels_opened") is not False or c27_outcome.get(
        "delayed_B_labels_opened"
    ) is not False:
        raise PermissionError("C28 source A/B labels are not untouched")
    if c27_g0.get("escrow_features_or_labels_opened") is not False:
        raise PermissionError("C28 source escrow was materialized")
    data = PackedStructure(paths["packed_train_root"])

    fit = [int(value) for value in c27["roles"]["fit"]["indices"]]
    internal_A = [int(value) for value in c27["roles"]["escrow"]["indices"]]
    internal_set = set(internal_A)
    delayed_B = ordered_subset(
        data,
        [int(value) for value in c26["roles"]["escrow"]["indices"] if int(value) not in internal_set],
        count=ROLE_COUNTS["delayed_B"],
        seed=seed,
        role="delayed_B",
    )
    structural_repeat = ordered_subset(
        data,
        list(
            set(int(value) for value in c26["roles"]["structural_repeat"]["indices"])
            - set(int(value) for value in c27["roles"]["structural_repeat"]["indices"])
        ),
        count=ROLE_COUNTS["structural_repeat"],
        seed=seed,
        role="structural_repeat",
    )
    structural_nohistory = ordered_subset(
        data,
        list(
            set(int(value) for value in c26["roles"]["structural_nohistory"]["indices"])
            - set(int(value) for value in c27["roles"]["structural_nohistory"]["indices"])
        ),
        count=ROLE_COUNTS["structural_nohistory"],
        seed=seed,
        role="structural_nohistory",
    )
    c26_outcomes = {
        int(value) for row in c26["roles"].values() for value in row["indices"]
    }
    donor_pool = sorted(
        {
            int(value)
            for row in c26["wrong_history_donors"].values()
            for value in row["indices"]
        }
        - c26_outcomes
    )
    escrow = ordered_subset(
        data,
        donor_pool,
        count=ROLE_COUNTS["escrow"],
        seed=seed,
        role="escrow",
    )
    roles = {
        "fit": fit,
        "internal_A": internal_A,
        "delayed_B": delayed_B,
        "escrow": escrow,
        "structural_repeat": structural_repeat,
        "structural_nohistory": structural_nohistory,
    }
    outcome_set = {index for values in roles.values() for index in values}
    if len(outcome_set) != sum(len(values) for values in roles.values()):
        raise AssertionError("C28 roles overlap")
    reserve = [index for index in donor_pool if index not in outcome_set]
    edges = [int(value) for value in config["selection"]["donor_length_bins"]]
    quantiles = int(config["selection"]["donor_time_quantiles"])
    time_edges = np.quantile(
        np.asarray(data.timestamps[reserve], dtype=np.float64),
        np.linspace(0.0, 1.0, quantiles + 1)[1:-1],
    )

    def bucket(index: int) -> tuple[int, int]:
        length = int(data.history_offsets[index + 1] - data.history_offsets[index])
        return length_bin(length, edges), int(np.searchsorted(time_edges, data.timestamps[index]))

    grouped: dict[tuple[int, int], list[int]] = {}
    length_grouped: dict[int, list[int]] = {}
    for index in reserve:
        grouped.setdefault(bucket(index), []).append(index)
        length_grouped.setdefault(bucket(index)[0], []).append(index)
    for group, values in grouped.items():
        values.sort(key=lambda index: (stable_key(seed, f"donor:{group}", data.request_ids[index]), index))
    for group, values in length_grouped.items():
        values.sort(key=lambda index: (stable_key(seed, f"donor_length:{group}", data.request_ids[index]), index))
    reserve.sort(key=lambda index: (stable_key(seed, "donor_fallback", data.request_ids[index]), index))

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
            if donor != recipient and recipient_candidates.isdisjoint(
                int(value) for value in data.history_indices(donor)
            ):
                return donor
        raise RuntimeError(f"C28 donor unavailable: {recipient}")

    donors = {
        role: [donor_for(index) for index in roles[role]]
        for role in ("fit", "internal_A", "delayed_B")
    }
    if any(donor in outcome_set for values in donors.values() for donor in values):
        raise AssertionError("C28 donor intersects outcome roles")
    result = {
        "candidate_id": "c28",
        "selection_id": "c28_margin_local_contest_selection_v1",
        "status": "frozen_before_any_c28_label_or_outcome",
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
            "recipient_candidate_overlap_forbidden": True,
            "outcome_role_donors_forbidden": True,
            "reserve_requests": len(reserve),
        },
        "sources": {
            "c26_selection_sha256": paths["c26_selection_sha256"],
            "c27_selection_sha256": paths["c27_selection_sha256"],
            "c27_g0_report_sha256": paths["c27_g0_report_sha256"],
            "c27_train_report_sha256": paths["c27_train_report_sha256"],
        },
        "checks": {
            "fit_labels_previously_opened": True,
            "c27_escrow_features_or_labels_opened": False,
            "c28_internal_A_labels_opened": False,
            "c28_delayed_B_labels_opened": False,
            "roles_pairwise_disjoint": True,
            "strict_nonrepeat_fit_A_B_escrow": True,
            "donors_outside_outcome_roles": True,
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
                "role_counts": {name: len(row["indices"]) for name, row in result["roles"].items()},
                "donor_reserve": result["donor_matching"]["reserve_requests"],
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
