from myrec.mechanism.selected_branch_runtime import (
    _extract_rms_errors,
    _request_shard_records,
    _stable_smoke_records,
    selected_branch_implementation_identity,
)


EXPECTED_SHARDED_SELECTED_BRANCH_DIGEST = (
    "3c98effc5e96cef7e9310ade19f515002e3b60bc717e9b4213a8b94a17c5a727"
)


def test_selected_branch_rms_error_extraction_is_keyed():
    value = {
        "node": {
            "yes": {
                "d_at_r_rms_max_abs_error": 0.25,
                "donor_rms_max": 99.0,
            }
        }
    }
    assert _extract_rms_errors(value) == [0.25]


def test_selected_branch_smoke_sample_is_order_independent():
    class Record:
        def __init__(self, request_id):
            self.request_id = request_id

    rows = [Record(f"r{index}") for index in range(20)]
    left = [row.request_id for row in _stable_smoke_records(rows, 5)]
    right = [row.request_id for row in _stable_smoke_records(rows[::-1], 5)]
    assert left == right


def test_selected_branch_request_shards_are_disjoint_complete_and_ordered():
    class Record:
        def __init__(self, request_id):
            self.request_id = request_id

    rows = [Record(f"r{index}") for index in range(9)]
    shards = [
        _request_shard_records(
            rows, request_shard_index=index, request_shard_count=2
        )
        for index in range(2)
    ]
    assert [[row.request_id for row in shard] for shard in shards] == [
        ["r0", "r2", "r4", "r6", "r8"],
        ["r1", "r3", "r5", "r7"],
    ]
    assert {row.request_id for shard in shards for row in shard} == {
        row.request_id for row in rows
    }


def test_selected_branch_sharded_implementation_is_locked_before_formal_run():
    assert (
        selected_branch_implementation_identity()["digest"]
        == EXPECTED_SHARDED_SELECTED_BRANCH_DIGEST
    )
