"""Freeze C43 from C37 fit and its completely unopened reserve roles."""

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


def load_user_ids(path: str | Path, data: PackedStructure) -> list[str]:
    positions = {request_id: index for index, request_id in enumerate(data.request_ids)}
    users: list[str | None] = [None] * len(data.request_ids)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = positions.get(str(row["request_id"]))
            if index is None:
                continue
            if users[index] is not None:
                raise ValueError("duplicate C43 label-free request metadata")
            if str(row.get("split")) != "train":
                raise ValueError("C43 user metadata contains non-train request")
            if int(row["time_index"]) != int(data.timestamps[index]):
                raise ValueError("C43 user metadata timestamp differs")
            users[index] = str(row["user_id"])
    if any(value is None for value in users):
        raise ValueError("C43 user metadata does not cover packed train")
    return [str(value) for value in users]


def target_indices(selection: dict) -> set[int]:
    return {
        int(value)
        for row in selection["roles"].values()
        for value in row["indices"]
    }


def length_bin(length: int, edges: list[int]) -> int:
    return next((position for position, edge in enumerate(edges) if length <= edge), len(edges))


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    integrity = config["integrity"]
    seed = int(config["selection"]["seed"])
    registered = (
        ("c40_model_source", "c40_model_source_sha256"),
        ("c40_report", "c40_report_sha256"),
        ("c41_report", "c41_report_sha256"),
        ("c42_report", "c42_report_sha256"),
        ("c37_config", "c37_config_sha256"),
        ("c37_selection", "c37_selection_sha256"),
        ("c37_g0_report", "c37_g0_report_sha256"),
        ("c37_train_report", "c37_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("label_free_request_manifest", "label_free_request_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
        ("raw_item_embeddings", "raw_item_embeddings_sha256"),
        ("calibration_checkpoint", "calibration_checkpoint_sha256"),
        ("internal_train_popularity", "internal_train_popularity_sha256"),
        ("train_candidate_labels", "train_candidate_labels_sha256"),
        ("candidate_manifest", "candidate_manifest_sha256"),
    )
    for name, expected in registered:
        if sha256_file(paths[name]) != integrity[expected]:
            raise RuntimeError(f"C43 registered source changed: {name}")
    if sha256_file(SYSTEM_ROOT / "model/metric_coupled.py") != integrity[
        "c40_model_source_sha256"
    ]:
        raise RuntimeError("C43 model is not exact C40 operator source")

    c42 = read_json(paths["c42_report"])
    required_c42 = (
        "over_base_ci",
        "over_base_all_seeds",
        "over_base_all_folds",
        "over_c38_unprojected_ci",
        "over_c38_unprojected_all_seeds",
        "over_c38_unprojected_all_folds",
        "true_over_wrong_ci",
        "true_over_wrong_all_seeds",
        "true_over_wrong_all_folds",
        "clicked_direction_ci",
    )
    if c42.get("status") != "failed_A1_terminal" or not all(
        c42["A1"]["checks"].get(name) is True for name in required_c42
    ):
        raise PermissionError("C42 cross-domain trigger differs")
    if not all(c42["A0"]["checks"].values()):
        raise PermissionError("C42 A0 trigger differs")

    c37_selection = read_json(paths["c37_selection"])
    c37_g0 = read_json(paths["c37_g0_report"])
    c37_report = read_json(paths["c37_train_report"])
    if c37_g0.get("delayed_B_features_labels_scores_opened") is not False:
        raise PermissionError("C37 delayed-B was materialized")
    if c37_g0.get("escrow_features_or_labels_opened") is not False:
        raise PermissionError("C37 escrow was materialized")
    if c37_report.get("delayed_B_features_labels_scores_opened") is not False:
        raise PermissionError("C37 delayed-B terminal boundary differs")
    if c37_report.get("escrow_dev_test_opened") is not False:
        raise PermissionError("C37 escrow terminal boundary differs")
    if c37_report.get("status") != "failed_A1_terminal":
        raise PermissionError("C37 terminal status differs")

    data = PackedStructure(paths["packed_train_root"])
    users = load_user_ids(paths["label_free_request_metadata"], data)
    c37_roles = {
        role: [int(value) for value in row["indices"]]
        for role, row in c37_selection["roles"].items()
    }
    fit = sorted(
        c37_roles["fit"],
        key=lambda index: (stable_key(seed, "fit", data.request_ids[index]), index),
    )[: ROLE_COUNTS["fit"]]
    internal_A = sorted(
        c37_roles["delayed_B"] + c37_roles["escrow"],
        key=lambda index: (stable_key(seed, "internal_A", data.request_ids[index]), index),
    )
    roles = {
        "fit": fit,
        "internal_A": internal_A,
        "structural_repeat": c37_roles["structural_repeat"],
        "structural_nohistory": c37_roles["structural_nohistory"],
    }
    if {name: len(values) for name, values in roles.items()} != ROLE_COUNTS:
        raise AssertionError("C43 role counts differ")
    flat = [index for values in roles.values() for index in values]
    if len(flat) != len(set(flat)):
        raise AssertionError("C43 roles overlap")
    if any(data.history_count(index) <= 0 or data.repeat_candidate_count(index) != 0 for index in fit + internal_A):
        raise AssertionError("C43 fit/A must be strict nonrepeat history requests")

    old_features = np.load(
        Path(paths["c37_selection"]).parent / "feature_request_indices.npy",
        mmap_mode="r",
    )
    old_feature_set = set(int(value) for value in old_features)
    if old_feature_set.intersection(internal_A):
        raise PermissionError("C43-A overlaps C37 materialized features")

    all_c37_targets = target_indices(c37_selection)
    outcome = set(flat)
    donor_reserve = [
        index
        for index in range(len(data.request_ids))
        if index not in all_c37_targets
        and index not in outcome
        and data.history_count(index) > 0
        and data.repeat_candidate_count(index) == 0
    ]
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
        values.sort(key=lambda index: (stable_key(seed, f"donor-length:{key}", data.request_ids[index]), index))
    donor_reserve.sort(key=lambda index: (stable_key(seed, "donor-fallback", data.request_ids[index]), index))

    def donor_for(recipient: int) -> int:
        candidates = grouped.get(bucket(recipient), []) or by_length.get(bucket(recipient)[0], []) or donor_reserve
        start = int.from_bytes(stable_key(seed, "donor-start", data.request_ids[recipient])[0][:8], "big") % len(candidates)
        recipient_candidates = set(int(value) for value in data.candidate_indices(recipient))
        for offset in range(len(candidates)):
            donor = int(candidates[(start + offset) % len(candidates)])
            if users[donor] != users[recipient] and recipient_candidates.isdisjoint(
                int(value) for value in data.history_indices(donor)
            ):
                return donor
        raise RuntimeError("C43 donor unavailable")

    donors = {
        role: [donor_for(index) for index in roles[role]]
        for role in ("fit", "internal_A")
    }
    exact_bucket = []
    exact_length = []
    for role, values in donors.items():
        for recipient, donor in zip(roles[role], values):
            exact_bucket.append(bucket(recipient) == bucket(donor))
            exact_length.append(bucket(recipient)[0] == bucket(donor)[0])
            if donor in all_c37_targets or donor in outcome:
                raise AssertionError("C43 donor intersects target role")
            if users[recipient] == users[donor]:
                raise AssertionError("C43 same-user donor")
            if not set(int(value) for value in data.candidate_indices(recipient)).isdisjoint(
                int(value) for value in data.history_indices(donor)
            ):
                raise AssertionError("C43 donor overlaps recipient candidates")

    result = {
        "candidate_id": "c43",
        "selection_id": "c43_cross_domain_metric_coupled_selection_v1",
        "status": "frozen_before_any_c43_feature_score_or_label",
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
            "c37_fit": "deterministic_6000_subset",
            "c37_internal_A": "exact_union_of_unopened_delayed_B_and_escrow",
            "c37_selection_sha256": integrity["c37_selection_sha256"],
            "c37_g0_report_sha256": integrity["c37_g0_report_sha256"],
            "c37_train_report_sha256": integrity["c37_train_report_sha256"],
            "c42_report_sha256": integrity["c42_report_sha256"],
        },
        "checks": {
            "c42_core_trigger_passed": True,
            "c37_delayed_B_and_escrow_unopened": True,
            "internal_A_exact_source_union": set(internal_A)
            == set(c37_roles["delayed_B"] + c37_roles["escrow"]),
            "internal_A_overlap_c37_materialized_features_zero": True,
            "fit_is_c37_fit_subset": set(fit).issubset(c37_roles["fit"]),
            "roles_pairwise_disjoint": True,
            "strict_nonrepeat_fit_A": True,
            "structural_roles_inherited": True,
            "donor_target_overlap_zero": True,
            "donor_user_overlap_zero": True,
            "donor_candidate_overlap_zero": True,
            "selection_label_access_closed": True,
            "dev_test_qrels_closed": True,
        },
        "donor_matching": {
            "history_length_edges": edges,
            "timestamp_quantiles": quantiles,
            "same_length_bin_fraction": float(np.mean(exact_length)),
            "same_length_and_time_bucket_fraction": float(np.mean(exact_bucket)),
            "same_user_forbidden": True,
            "recipient_candidate_overlap_forbidden": True,
            "reserve_requests": len(donor_reserve),
        },
    }
    if not all(value is True for value in result["checks"].values()):
        raise AssertionError("C43 frozen selection checks failed")
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
                "donor_matching": result["donor_matching"],
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
