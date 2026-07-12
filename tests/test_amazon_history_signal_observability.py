import json
import sys
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.amazon_history_signal_observability import (
    AmazonFrozenSemanticStore,
    AmazonObservabilityData,
    load_all_feature_labels,
    load_feature_users,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def fixture(tmp_path: Path) -> tuple[AmazonObservabilityData, Path, Path]:
    np.save(tmp_path / "feature_request_indices.npy", np.asarray([0, 1]))
    np.save(tmp_path / "candidate_offsets.npy", np.asarray([0, 2, 4]))
    np.save(tmp_path / "candidate_item_positions.npy", np.asarray([0, 1, 1, 2]))
    np.save(tmp_path / "true_history_offsets.npy", np.asarray([0, 1, 2]))
    np.save(tmp_path / "true_history_item_positions.npy", np.asarray([2, 1]))
    np.save(tmp_path / "wrong_history_offsets.npy", np.asarray([0, 1, 2]))
    np.save(tmp_path / "wrong_history_item_positions.npy", np.asarray([1, 0]))
    np.save(tmp_path / "query_embeddings.npy", np.ones((2, 4), dtype=np.float16))
    np.save(tmp_path / "item_embeddings.npy", np.eye(3, 4, dtype=np.float16))
    write_rows(
        tmp_path / "requests.jsonl",
        [
            {"position": 0, "record_index": 0, "request_id": "r0", "text": "q0"},
            {"position": 1, "record_index": 1, "request_id": "r1", "text": "q1"},
        ],
    )
    write_rows(
        tmp_path / "items.jsonl",
        [
            {"position": 0, "item_id": "i0", "text": "a"},
            {"position": 1, "item_id": "i1", "text": "b"},
            {"position": 2, "item_id": "i2", "text": "c"},
        ],
    )
    blind = tmp_path / "records_train_blind.jsonl"
    train = tmp_path / "records_train.jsonl"
    blind_rows = [
        {
            "request_id": "r0",
            "user_id": "u0",
            "candidates": [{"item_id": "i0"}, {"item_id": "i1"}],
        },
        {
            "request_id": "r1",
            "user_id": "u1",
            "candidates": [{"item_id": "i1"}, {"item_id": "i2"}],
        },
    ]
    train_rows = [
        {
            **row,
            "candidates": [
                {**candidate, "clicked": int(position == 0)}
                for position, candidate in enumerate(row["candidates"])
            ],
        }
        for row in blind_rows
    ]
    write_rows(blind, blind_rows)
    write_rows(train, train_rows)
    return AmazonObservabilityData(tmp_path), blind, train


def test_adapter_strict_nonrepeat_and_wrong_pseudo_rows(tmp_path: Path) -> None:
    data, _, _ = fixture(tmp_path)
    np.testing.assert_array_equal(data.strict_nonrepeat_indices(), np.asarray([0]))
    wrong = data.wrong_mapping([0, 1])
    np.testing.assert_array_equal(data.history(wrong[0])[0], np.asarray([1]))
    assert data.candidate_ids(0).tolist() == ["i0", "i1"]


def test_adapter_embeddings_users_and_staged_labels(tmp_path: Path) -> None:
    data, blind, train = fixture(tmp_path)
    features = AmazonFrozenSemanticStore(data)
    assert features.query([0, 1]).shape == (2, 4)
    assert features.item(np.asarray([[0, 1]])).shape == (1, 2, 4)
    assert load_feature_users(blind, data) == ["u0", "u1"]
    labels = load_all_feature_labels(train, data)
    np.testing.assert_array_equal(labels[0], np.asarray([1.0, 0.0]))
    assert data.labels_opened
