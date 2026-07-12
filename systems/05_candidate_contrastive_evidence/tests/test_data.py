import json
from types import SimpleNamespace

import numpy as np
import pytest

from train.data import (
    assert_training_path_firewall,
    collate_g2a,
    freeze_selection,
)


def _save(root, name, values):
    np.save(root / name, np.asarray(values))


def test_selection_uses_only_structural_arrays_before_labels(tmp_path):
    train = tmp_path / "packed" / "train"
    train.mkdir(parents=True)
    with (train / "request_ids.jsonl").open("w", encoding="utf-8") as handle:
        for index in range(4):
            handle.write(json.dumps({"request_id": f"r{index}"}) + "\n")
    _save(train, "candidate_offsets.npy", [0, 2, 4, 6, 8])
    _save(train, "candidate_embedding_indices.npy", [1, 2, 1, 3, 4, 5, 4, 5])
    _save(train, "history_offsets.npy", [0, 1, 2, 2, 3])
    _save(train, "history_embedding_indices.npy", [5, 1, 9])
    # Deliberately do not create candidate_labels.npy or any other label array.
    config = {
        "seed": 20260708,
        "paths": {
            "packed_train_root": str(tmp_path / "packed"),
            "artifact_root": str(tmp_path / "artifacts"),
        },
        "integrity": {
            "packed_train_requests": 4,
            "packed_cut_request_index": 3,
        },
        "selection": {
            "fit_range": [0, 3],
            "internal_range": [3, 4],
            "fit_nonrepeat_requests": 1,
            "internal_nonrepeat_requests": 1,
        },
        "candidate_id": "c05",
    }
    result = freeze_selection(config)
    assert result["labels_opened_before_selection"] is False
    assert result["fit"]["indices"] == [0]
    assert result["internal"]["indices"] == [3]
    assert result["pool_counts"] == {
        "fit_nonrepeat_history_present": 1,
        "internal_nonrepeat_history_present": 1,
    }


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("leak", "data/qrels_dev.jsonl"),
        ("checkpoint", "models/d2_final_seed/model.pt"),
        ("internal_train_popularity", "artifacts/item_log_click_full_train.npy"),
    ],
)
def test_training_path_firewall_fails_closed(name, path):
    config = {
        "paths": {
            "calibration_checkpoint": "models/d2_calibrate/model.pt",
            "internal_train_popularity": "artifacts/item_log_click_internal_train.npy",
            name: path,
        }
    }
    if name == "checkpoint":
        config["paths"]["calibration_checkpoint"] = path
    if name == "internal_train_popularity":
        config["paths"]["internal_train_popularity"] = path
    with pytest.raises(ValueError):
        assert_training_path_firewall(config)


def test_full_candidate_collation_keeps_recent_history_and_event_identity():
    data = SimpleNamespace(
        candidate_offsets=np.asarray([0, 3], dtype=np.int64),
        candidate_embedding_indices=np.asarray([10, 11, 12], dtype=np.int64),
        candidate_item_ids=np.asarray([100, 101, 102], dtype=np.int64),
        candidate_labels=np.asarray([0, 1, 0], dtype=np.float32),
        history_offsets=np.asarray([0, 3], dtype=np.int64),
        history_embedding_indices=np.asarray([20, 21, 22], dtype=np.int64),
        history_event_weights=np.asarray([1.0, 1.5, 1.0], dtype=np.float32),
    )
    batch = collate_g2a(data, np.asarray([0]), history_limit=2)
    assert batch["candidate_mask"].tolist() == [[True, True, True]]
    assert batch["candidate_indices"].tolist() == [[10, 11, 12]]
    assert batch["history_indices"].tolist() == [[21, 22]]
    np.testing.assert_allclose(
        batch["history_event_weights"],
        [[1.5 / np.sqrt(2.0), 1.0]],
        rtol=1e-6,
    )
