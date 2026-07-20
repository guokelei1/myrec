from __future__ import annotations

import pytest

from myrec.mechanism.attention_pattern_analysis import MODELS
from myrec.mechanism.attention_position_shift_analysis import (
    summarize_attention_position_shifts,
)


def _rows():
    result = {}
    for model_id in MODELS:
        path_names = ("prompt",) if model_id == MODELS[0] else ("yes", "no")
        rows = []
        for index in range(2):
            paths = {}
            for path_name in path_names:
                native = [14, 15] if model_id == MODELS[1] else [14]
                null_native = [9, 10] if model_id == MODELS[1] else [9]
                paths[path_name] = {
                    "full_positions": [3, 8, *native],
                    "null_positions": [3, 5, *null_native],
                    "full_sequence_length": 16 if model_id == MODELS[1] else 15,
                    "null_sequence_length": 11 if model_id == MODELS[1] else 10,
                }
            rows.append(
                {
                    "request_id": f"r{index}",
                    "candidate_item_id": f"i{index}",
                    "paths": paths,
                }
            )
        result[model_id] = rows
    return result


def test_position_shift_summary_preserves_paths_and_hand_values():
    result = summarize_attention_position_shifts(_rows(), expected_rows=2)
    assert len(result["path_cells"]) == 3
    assert result["invariants"]["all_query_position_shifts_zero"] is True
    assert result["invariants"]["downstream_position_shift_nonzero_fraction"] == 1.0
    assert (
        result["invariants"][
            "native_readout_shift_matches_sequence_length_shift_fraction"
        ]
        == 1.0
    )
    q2 = result["path_cells"][0]
    assert q2["positions"]["history_summary_end"]["full_minus_null_position"][
        "mean"
    ] == pytest.approx(3.0)
    assert q2["positions"]["native_readout_0"]["full_minus_null_position"][
        "mean"
    ] == pytest.approx(5.0)


def test_position_shift_rejects_qrels_or_model_identity_drift():
    with pytest.raises(ValueError, match="qrels/source-test blind"):
        summarize_attention_position_shifts(
            _rows(), expected_rows=2, qrels_read=True
        )
    rows = _rows()
    rows[MODELS[1]][0]["candidate_item_id"] = "wrong"
    with pytest.raises(ValueError, match="identity/order differs"):
        summarize_attention_position_shifts(rows, expected_rows=2)
