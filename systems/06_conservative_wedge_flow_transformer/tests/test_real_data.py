from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from train.real_data import (
    FrozenRealFeatures,
    SelectedLabels,
    StructuralTrainData,
    assert_internal_a_opening_barrier,
    build_selection,
    collate_structural,
    open_selected_labels,
    selected_candidate_key_sha256,
    sha256_file,
    write_json,
)


def _structural_data(tmp_path: Path) -> StructuralTrainData:
    requests = 24
    request_ids = [f"r{index:02d}" for index in range(requests)]
    candidate_offsets = np.arange(requests + 1, dtype=np.int64) * 2
    candidate_indices = np.asarray(
        [value for index in range(requests) for value in (2 * index, 2 * index + 1)],
        dtype=np.int64,
    )
    candidate_item_ids = candidate_indices + 1000
    history_rows = []
    history_offsets = [0]
    # Last four post-cut rows are no-history; all other history items are
    # structurally disjoint from their candidates.
    for index in range(requests):
        if index < requests - 4:
            history_rows.append(10_000 + index)
        history_offsets.append(len(history_rows))
    return StructuralTrainData(
        root=tmp_path,
        request_ids=request_ids,
        query_indices=np.arange(requests, dtype=np.int64),
        timestamps=np.arange(requests, dtype=np.int64),
        candidate_offsets=candidate_offsets,
        candidate_embedding_indices=candidate_indices,
        candidate_item_ids=candidate_item_ids,
        history_offsets=np.asarray(history_offsets, dtype=np.int64),
        history_embedding_indices=np.asarray(history_rows, dtype=np.int64),
        history_event_weights=np.ones(len(history_rows), dtype=np.float32),
    )


def test_structural_selection_is_stable_disjoint_and_excludes_c05(tmp_path: Path) -> None:
    data = _structural_data(tmp_path)
    counts = {
        "fit": 3,
        "internal_A": 2,
        "internal_B": 2,
        "escrow": 1,
        "nohistory": 2,
    }
    c05 = {"r01", "r13"}
    first = build_selection(
        data, c05_request_ids=c05, seed=20260708, cut=12, counts=counts
    )
    second = build_selection(
        data, c05_request_ids=c05, seed=20260708, cut=12, counts=counts
    )
    assert first == second
    selected = [
        request_id
        for row in first["roles"].values()
        for request_id in row["request_ids"]
    ]
    assert len(selected) == len(set(selected))
    assert not (set(selected) & c05)
    assert all(index < 12 for index in first["roles"]["fit"]["indices"])
    assert all(
        data.history_offsets[index] == data.history_offsets[index + 1]
        for index in first["roles"]["nohistory"]["indices"]
    )
    assert first["labels_opened_before_selection"] is False


def test_collation_uses_all_candidates_and_uniform_history_prior(tmp_path: Path) -> None:
    data = _structural_data(tmp_path)
    batch = collate_structural(data, [0, 23], history_limit=20)
    assert batch["candidate_mask"].sum(axis=1).tolist() == [2, 2]
    assert batch["history_mask"].sum(axis=1).tolist() == [1, 0]
    assert batch["history_prior"][0, 0] == 1.0
    assert not batch["history_prior"][1].any()
    assert "candidate_labels" not in batch


def test_selected_labels_cannot_escape_allowed_role(tmp_path: Path) -> None:
    data = _structural_data(tmp_path)
    label_path = tmp_path / "candidate_labels.npy"
    np.save(label_path, np.arange(48, dtype=np.float32))
    with pytest.raises(PermissionError):
        open_selected_labels(
            data, [1], label_path=label_path, allowed_indices={0}
        )
    selected = open_selected_labels(
        data, [0, 2], label_path=label_path, allowed_indices={0, 2}
    )
    batch = collate_structural(data, [2, 0], history_limit=20)
    padded = selected.padded(batch)
    assert padded.tolist() == [[4.0, 5.0], [0.0, 1.0]]
    with pytest.raises(PermissionError):
        selected.padded(collate_structural(data, [1], history_limit=20))


def test_internal_a_barrier_requires_durable_pass_and_exact_candidates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "a0.json"
    report = {
        "gate": "G2_A0_label_free",
        "status": "passed",
        "candidate_key_sha256": "abc",
        "internal_A_labels_opened": False,
        "internal_B_or_escrow_opened": False,
        "checks": {"determinism": True, "finite": True},
    }
    write_json(path, report)
    assert assert_internal_a_opening_barrier(
        path, expected_candidate_key_sha256="abc"
    ) == sha256_file(path)
    with pytest.raises(PermissionError):
        assert_internal_a_opening_barrier(
            path, expected_candidate_key_sha256="different"
        )
    report["checks"]["finite"] = False
    write_json(path, report)
    with pytest.raises(PermissionError):
        assert_internal_a_opening_barrier(
            path, expected_candidate_key_sha256="abc"
        )


def test_candidate_key_hash_binds_order_and_full_candidate_identity(
    tmp_path: Path,
) -> None:
    data = _structural_data(tmp_path)
    original = selected_candidate_key_sha256(data, [0, 1])
    assert original != selected_candidate_key_sha256(data, [1, 0])
    changed_ids = data.candidate_item_ids.copy()
    changed_ids[0] += 1
    changed = StructuralTrainData(
        **{**data.__dict__, "candidate_item_ids": changed_ids}
    )
    assert original != selected_candidate_key_sha256(changed, [0, 1])


def test_selected_labels_validate_offsets() -> None:
    with pytest.raises(ValueError):
        SelectedLabels(
            np.asarray([1]), np.asarray([0, 1, 2]), np.asarray([1.0, 2.0])
        )


def test_frozen_features_use_sorted_selected_item_state_lookup(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    np.save(root / "feature_request_indices.npy", np.asarray([3, 4, 5]))
    np.save(root / "query_embeddings.npy", np.ones((3, 2), dtype=np.float32))
    np.save(root / "item_embedding_indices.npy", np.asarray([0, 7, 9]))
    np.save(
        root / "item_embeddings.npy",
        np.asarray([[0.0, 0.0], [7.0, 1.0], [9.0, 1.0]], dtype=np.float32),
    )
    np.save(root / "feature_candidate_offsets.npy", np.asarray([0, 1, 2, 3]))
    np.save(root / "base_scores.npy", np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    np.save(root / "fit_request_indices.npy", np.asarray([3]))
    np.save(root / "fit_label_offsets.npy", np.asarray([0, 1]))
    np.save(root / "fit_labels.npy", np.asarray([1.0], dtype=np.float32))
    config = {"paths": {"artifact_root": str(root)}}
    selection = {
        "roles": {
            "fit": {"indices": [3]},
            "internal_A": {"indices": [4]},
            "nohistory": {"indices": [5]},
        }
    }
    features = FrozenRealFeatures(config, selection)
    batch = {
        "request_indices": np.asarray([3]),
        "candidate_indices": np.asarray([[7]]),
        "candidate_mask": np.asarray([[True]]),
        "history_indices": np.asarray([[0]]),
        "history_mask": np.asarray([[False]]),
        "history_prior": np.asarray([[0.0]], dtype=np.float32),
    }
    tensors = features.tensors(batch, "cpu", labels=features.fit_labels)
    assert tensors["candidates"].tolist() == [[[7.0, 1.0]]]
    assert tensors["history"].tolist() == [[[0.0, 0.0]]]
    batch["candidate_indices"] = np.asarray([[8]])
    with pytest.raises(ValueError, match="unmaterialized"):
        features.tensors(batch, "cpu")
