import json
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.token_history_observability import (
    TokenHistoryData,
    listwise_loss,
    sample_positions,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def fixture(tmp_path: Path) -> TokenHistoryData:
    np.save(tmp_path / "request_original_indices.npy", np.asarray([10]))
    np.save(tmp_path / "request_roles.npy", np.asarray([1], dtype=np.int8))
    np.save(tmp_path / "candidate_offsets.npy", np.asarray([0, 2]))
    np.save(tmp_path / "candidate_item_positions.npy", np.asarray([0, 1]))
    np.save(tmp_path / "history_offsets.npy", np.asarray([0, 2]))
    np.save(tmp_path / "history_item_positions.npy", np.asarray([2, 3]))
    np.save(tmp_path / "wrong_history_offsets.npy", np.asarray([0, 1]))
    np.save(tmp_path / "wrong_history_item_positions.npy", np.asarray([1]))
    np.save(tmp_path / "query_token_ids.npy", np.asarray([[10, 11, 0, 0]], dtype=np.int32))
    np.save(tmp_path / "query_attention_mask.npy", np.asarray([[1, 1, 0, 0]], dtype=bool))
    np.save(
        tmp_path / "item_token_ids.npy",
        np.asarray([[20, 0, 0], [21, 22, 0], [30, 0, 0], [31, 32, 0]], dtype=np.int32),
    )
    np.save(
        tmp_path / "item_attention_mask.npy",
        np.asarray([[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    )
    write_rows(
        tmp_path / "requests.jsonl",
        [{"position": 0, "request_id": "r", "user_id": "u"}],
    )
    write_rows(
        tmp_path / "items.jsonl",
        [{"position": index, "item_id": f"i{index}"} for index in range(4)],
    )
    (tmp_path / "token_manifest.json").write_text(
        json.dumps(
            {
                "special_tokens": {
                    "cls_token_id": 101,
                    "sep_token_id": 102,
                    "pad_token_id": 0,
                }
            }
        )
    )
    return TokenHistoryData(tmp_path)


def test_pack_true_null_and_shuffle(tmp_path: Path) -> None:
    data = fixture(tmp_path)
    true, true_mask = data.pack_candidate(
        0,
        0,
        scenario="true",
        query_tokens=4,
        candidate_tokens=3,
        history_item_tokens=3,
        max_history=2,
        max_length=20,
    )
    null, null_mask = data.pack_candidate(
        0,
        0,
        scenario="null",
        query_tokens=4,
        candidate_tokens=3,
        history_item_tokens=3,
        max_history=2,
        max_length=20,
    )
    shuffled, _ = data.pack_candidate(
        0,
        0,
        scenario="shuffle",
        query_tokens=4,
        candidate_tokens=3,
        history_item_tokens=3,
        max_history=2,
        max_length=20,
    )
    assert true[:6].tolist() == [101, 10, 11, 102, 20, 102]
    assert int(true_mask.sum()) > int(null_mask.sum())
    assert true.tolist() != shuffled.tolist()


def test_sampling_keeps_positive_and_loss_is_finite() -> None:
    labels = np.asarray([0, 0, 1, 0, 0], dtype=np.float32)
    selected = sample_positions(labels, 3, np.random.default_rng(3))
    assert 2 in selected
    scores = torch.tensor([[0.1, 0.2, 0.3]])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    assert torch.isfinite(listwise_loss(scores, target))


def test_sampling_keeps_multiple_positives() -> None:
    labels = np.asarray([1, 0, 1, 0, 0, 0], dtype=np.float32)
    selected = sample_positions(labels, 4, np.random.default_rng(7))
    assert {0, 2}.issubset(set(selected.tolist()))
