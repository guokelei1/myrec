from __future__ import annotations

import numpy as np

from myrec.mechanism.deep_dive_representation_analysis import (
    _target_candidate_ordinals,
)


class _Record:
    def __init__(self, request_id, candidate_ids):
        self.request_id = request_id
        self.candidates = [{"item_id": value} for value in candidate_ids]


def test_target_candidate_ordinal_is_first_maximum_in_frozen_order():
    records = [_Record("r", ["a", "b", "c"]), _Record("z", ["x", "y"])]
    values = _target_candidate_ordinals(
        records,
        {"r": {"a": 1.0, "b": 3.0, "c": 3.0}, "z": {}},
    )
    assert values == [1, 0]


def test_balanced_accuracy_fixture_is_hand_computed():
    from myrec.mechanism.deep_dive_representation_analysis import _balanced_accuracy

    target = np.asarray(["a", "a", "a", "b"])
    predicted = np.asarray(["a", "a", "b", "b"])
    # Recall(a)=2/3, Recall(b)=1, balanced=(2/3+1)/2=5/6.
    assert np.isclose(_balanced_accuracy(target, predicted), 5.0 / 6.0)
