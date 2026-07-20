from __future__ import annotations

from myrec.mechanism.postblock_sweep_runtime import _stable_smoke_records
from myrec.mechanism.postblock_sweep_scoring import POSTBLOCK_CONDITIONS


def test_postblock_sweep_condition_order_is_frozen():
    assert POSTBLOCK_CONDITIONS == (
        "baseline_full",
        "baseline_null",
        "full_to_full_identity",
        "null_to_null_identity",
        "same_full_to_null",
        "cross_full_to_null",
    )


def test_postblock_smoke_selection_is_stable_and_not_prefix():
    class Record:
        def __init__(self, request_id):
            self.request_id = request_id

    records = [Record(f"r{ordinal}") for ordinal in range(20)]
    selected = _stable_smoke_records(records, 4)
    assert [row.request_id for row in selected] == [
        row.request_id for row in _stable_smoke_records(list(reversed(records)), 4)
    ]
    assert [row.request_id for row in selected] != ["r0", "r1", "r2", "r3"]
