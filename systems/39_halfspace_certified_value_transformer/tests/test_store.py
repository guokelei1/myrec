from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.selection import sha256_file  # noqa: E402
from train.store import FrozenTransferStore, open_role_labels  # noqa: E402


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_store_and_role_scoped_label_opener(tmp_path: Path) -> None:
    root = tmp_path / "features"
    root.mkdir()
    np.save(root / "feature_request_indices.npy", np.asarray([0, 1], dtype=np.int64))
    np.save(root / "candidate_offsets.npy", np.asarray([0, 2, 4], dtype=np.int64))
    np.save(root / "candidate_item_positions.npy", np.asarray([0, 1, 1, 2], dtype=np.int32))
    np.save(root / "true_history_offsets.npy", np.asarray([0, 1, 2], dtype=np.int64))
    np.save(root / "true_history_item_positions.npy", np.asarray([1, 0], dtype=np.int32))
    np.save(root / "wrong_history_offsets.npy", np.asarray([0, 1, 2], dtype=np.int64))
    np.save(root / "wrong_history_item_positions.npy", np.asarray([0, 2], dtype=np.int32))
    embeddings = np.eye(3, 4, dtype=np.float16)
    np.save(root / "item_embeddings.npy", embeddings)
    np.save(root / "query_embeddings.npy", np.ones((2, 4), dtype=np.float16))
    np.save(root / "base_scores.npy", np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    (root / "items.jsonl").write_text(
        "".join(
            json.dumps({"position": position, "item_id": f"i{position}", "text": "x"}) + "\n"
            for position in range(3)
        ),
        encoding="utf-8",
    )
    (root / "requests.jsonl").write_text(
        "".join(
            json.dumps({"position": position, "record_index": position, "request_id": f"r{position}", "text": "q"}) + "\n"
            for position in range(2)
        ),
        encoding="utf-8",
    )
    _write_json(root / "embedding_manifest.json", {"finite": True})
    selection_path = tmp_path / "selection.json"
    _write_json(
        selection_path,
        {
            "roles": {
                "fit": {"indices": [0]},
                "internal_A": {"indices": [1]},
                "delayed_B": {"indices": []},
                "escrow": {"indices": []},
            }
        },
    )
    train_path = tmp_path / "records_train.jsonl"
    train_rows = [
        {
            "request_id": "r0",
            "candidates": [
                {"item_id": "i0", "clicked": 1},
                {"item_id": "i1", "clicked": 0},
            ],
        },
        {
            "request_id": "r1",
            "candidates": [
                {"item_id": "i1", "clicked": 0},
                {"item_id": "i2", "clicked": 1},
            ],
        },
    ]
    train_path.write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
    config = {
        "paths": {"selection": str(selection_path), "feature_root": str(root)},
        "model": {"embedding_dim": 4},
    }
    store = FrozenTransferStore(config)
    assert store.candidate_ids(0) == ["i0", "i1"]
    assert store.has_repeat(0) is True
    assert store.has_repeat(1) is False
    labels = open_role_labels(
        records_train_path=train_path,
        records_train_sha256=sha256_file(train_path),
        selection_path=selection_path,
        selection_sha256=sha256_file(selection_path),
        store=store,
        role="internal_A",
    )
    assert labels.request_indices.tolist() == [1]
    assert labels.row(1, 2).tolist() == [0.0, 1.0]
