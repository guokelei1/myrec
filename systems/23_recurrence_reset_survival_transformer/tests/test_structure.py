from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.structure import PackedStructure


def tiny() -> PackedStructure:
    return PackedStructure(
        root=Path("."),
        request_ids=["a", "b", "c"],
        query_indices=np.asarray([0, 1, 2]),
        timestamps=np.asarray([1, 2, 3]),
        candidate_offsets=np.asarray([0, 2, 4, 6]),
        candidate_embedding_indices=np.asarray([1, 2, 3, 4, 5, 6]),
        candidate_item_ids=np.asarray([10, 20, 30, 40, 50, 60]),
        history_offsets=np.asarray([0, 0, 2, 4]),
        history_embedding_indices=np.asarray([8, 9, 6, 7]),
        history_event_weights=np.asarray([1.0, 1.5, 1.0, 1.0]),
    )


def test_structural_strata_use_identity_only() -> None:
    data = tiny()
    data.validate()
    assert data.stratum(0) == "nohistory"
    assert data.stratum(1) == "nonrepeat"
    assert data.stratum(2) == "repeat"
