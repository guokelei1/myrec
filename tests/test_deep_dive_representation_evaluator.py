from __future__ import annotations

import numpy as np

from myrec.mechanism.deep_dive_representation_evaluator import (
    BLOCK_REGIONS,
    _classification_summary,
    _geometry,
)


def test_block_regions_cover_every_transformer_block_output_once():
    states = [state for region in BLOCK_REGIONS.values() for state in region]
    assert states == list(range(1, 29))


def test_geometry_hand_computed_fixture():
    full = np.asarray([[3.0, 4.0]])
    null = np.asarray([[0.0, 5.0]])
    l2, cosine, ratio = _geometry(full, null)
    assert np.isclose(l2[0], np.sqrt(10.0) / np.sqrt(2.0))
    assert np.isclose(cosine[0], 1.0 - 0.8)
    assert np.isclose(ratio[0], 1.0)


def test_balanced_accuracy_is_macro_recall():
    target = np.asarray(["a", "a", "a", "b"])
    prediction = np.asarray(["a", "a", "b", "b"])
    summary = _classification_summary(target, prediction)
    assert np.isclose(summary["balanced_accuracy"], 5.0 / 6.0)
