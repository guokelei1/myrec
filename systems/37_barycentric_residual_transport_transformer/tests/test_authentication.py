from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


SYSTEM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM))

from train.authentication import build_authentication, load_user_ids  # noqa: E402
from train.structure import PackedStructure  # noqa: E402


def packed(tmp_path: Path) -> PackedStructure:
    rows = [{"request_id": f"r{index}"} for index in range(4)]
    (tmp_path / "request_ids.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    np.save(tmp_path / "candidate_offsets.npy", np.asarray([0, 1, 2, 3, 4], np.int64))
    np.save(tmp_path / "candidate_embedding_indices.npy", np.asarray([90, 91, 92, 93]))
    np.save(tmp_path / "candidate_item_ids.npy", np.asarray([900, 901, 902, 903]))
    np.save(tmp_path / "history_offsets.npy", np.asarray([0, 1, 3, 4, 6], np.int64))
    np.save(
        tmp_path / "history_embedding_indices.npy",
        np.asarray([10, 10, 20, 30, 20, 30], np.int64),
    )
    np.save(tmp_path / "history_event_weights.npy", np.ones(6, np.float32))
    np.save(tmp_path / "timestamps.npy", np.asarray([1, 2, 2, 3], np.int64))
    return PackedStructure(tmp_path)


def test_strict_past_and_same_timestamp_score_before_update(tmp_path: Path) -> None:
    data = packed(tmp_path)
    rows = build_authentication(
        data=data,
        user_ids=["u", "u", "u", "u"],
        target_indices=[0, 1, 2, 3],
        donor_by_recipient={3: 2},
    )
    true = [
        rows.true_items[rows.true_offsets[index] : rows.true_offsets[index + 1]].tolist()
        for index in range(4)
    ]
    wrong = [
        rows.wrong_items[rows.wrong_offsets[index] : rows.wrong_offsets[index + 1]].tolist()
        for index in range(4)
    ]
    assert true == [[], [10], [], [20, 30]]
    assert wrong == [[], [], [], [30]]
    assert rows.profile_sizes.tolist() == [0, 1, 1, 3]
    assert rows.audit["same_timestamp_score_before_update"] is True


def test_label_free_user_metadata_is_train_and_time_bound(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    root.mkdir()
    data = packed(root)
    metadata = tmp_path / "requests.jsonl"
    metadata.write_text(
        "".join(
            json.dumps(
                {
                    "request_id": f"r{index}",
                    "user_id": f"u{index}",
                    "split": "train",
                    "time_index": timestamp,
                }
            )
            + "\n"
            for index, timestamp in enumerate([1, 2, 2, 3])
        ),
        encoding="utf-8",
    )
    assert load_user_ids(metadata, data) == ["u0", "u1", "u2", "u3"]
