from __future__ import annotations

import hashlib

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord, TrainingGroup
from myrec.mechanism.recoverability_probe import (
    FEATURE_NAMES,
    RecoverabilityFeatureExtractor,
    _build_pairwise_dataset,
    _routing_query_map,
)


class _FakeStore:
    def __call__(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        value = np.asarray([byte + 1 for byte in digest[:8]], dtype=np.float32)
        return value / np.linalg.norm(value)


def _event(item_id: str, title: str, brand: str, category: str) -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": brand,
        "cat": ["root", category],
        "event": "click",
        "query": "old query",
        "ts": 1,
    }


def _candidate(item_id: str, title: str, brand: str, category: str) -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": brand,
        "cat": ["root", category],
    }


def test_features_do_not_serialize_raw_item_identity() -> None:
    extractor = RecoverabilityFeatureExtractor(_FakeStore())
    left = ModelRecord(
        request_id="r1",
        query="running shoes",
        history=(_event("history-a", "shoe one", "b", "shoe"),),
        candidates=(
            _candidate("candidate-a", "shoe two", "b", "shoe"),
            _candidate("candidate-b", "hat", "c", "hat"),
        ),
    )
    right = ModelRecord(
        request_id="r2",
        query=left.query,
        history=(_event("different-history-id", "shoe one", "b", "shoe"),),
        candidates=(
            _candidate("different-candidate-a", "shoe two", "b", "shoe"),
            _candidate("different-candidate-b", "hat", "c", "hat"),
        ),
    )
    left_values = list(extractor.candidate_features(left).values())
    right_values = list(extractor.candidate_features(right).values())
    assert len(left_values) == len(right_values) == 2
    assert left_values[0].shape == (len(FEATURE_NAMES),)
    np.testing.assert_allclose(left_values, right_values)


def test_pairwise_dataset_is_transfer_only_mirrored_and_deterministic() -> None:
    extractor = RecoverabilityFeatureExtractor(_FakeStore())
    transfer = TrainingGroup(
        record=ModelRecord(
            request_id="transfer",
            query="shoe",
            history=(_event("h", "old shoe", "b", "shoe"),),
            candidates=(
                _candidate("p", "new shoe", "b", "shoe"),
                _candidate("n", "new hat", "c", "hat"),
            ),
        ),
        candidates=(
            _candidate("p", "new shoe", "b", "shoe"),
            _candidate("n", "new hat", "c", "hat"),
        ),
        gains=(2.0, 0.0),
    )
    recurrence = TrainingGroup(
        record=ModelRecord(
            request_id="repeat",
            query="shoe",
            history=(_event("p", "new shoe", "b", "shoe"),),
            candidates=transfer.record.candidates,
        ),
        candidates=transfer.candidates,
        gains=transfer.gains,
    )
    first_x, first_y, stats = _build_pairwise_dataset(
        [transfer, recurrence],
        extractor,
        label_shuffle=False,
        seed=7,
        max_pairs_per_request=16,
    )
    second_x, second_y, _ = _build_pairwise_dataset(
        [transfer, recurrence],
        extractor,
        label_shuffle=False,
        seed=7,
        max_pairs_per_request=16,
    )
    assert stats["eligible_strict_transfer_groups"] == 1
    assert stats["skipped_nontransfer_groups"] == 1
    assert first_y.tolist() == [1, 0]
    np.testing.assert_allclose(first_x[0], -first_x[1])
    np.testing.assert_array_equal(first_x, second_x)
    np.testing.assert_array_equal(first_y, second_y)


def test_routing_query_shuffle_is_stable_and_avoids_same_query_when_possible() -> None:
    records = [
        ModelRecord(str(index), f"query-{index}", (), (_candidate(f"a{index}", "a", "b", "c"), _candidate(f"b{index}", "b", "b", "c")))
        for index in range(4)
    ]
    first = _routing_query_map(records, seed=11)
    second = _routing_query_map(records, seed=11)
    assert first == second
    assert set(first) == {record.request_id for record in records}
    assert all(first[record.request_id] != record.query for record in records)
