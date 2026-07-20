"""Qrels-blind semantic-position shift audit for full/null attention geometry."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.attention_pattern_analysis import MODELS


EXPECTED_PATHS = {
    MODELS[0]: ("prompt",),
    MODELS[1]: ("yes", "no"),
}


def summarize_attention_position_shifts(
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    expected_rows: int,
    qrels_read: bool = False,
    source_test_opened: bool = False,
) -> dict[str, Any]:
    """Summarize exact padded-sequence indices without inspecting labels/scores."""

    if qrels_read is not False or source_test_opened is not False:
        raise ValueError("position-shift analysis must remain qrels/source-test blind")
    if set(model_rows) != set(MODELS) or expected_rows <= 0:
        raise ValueError("position-shift model coverage differs")

    identities = None
    path_cells = []
    all_query_shifts: list[int] = []
    all_downstream_shifts: list[int] = []
    native_matches_sequence_shift: list[bool] = []
    for model_id in MODELS:
        rows = list(model_rows[model_id])
        if len(rows) != expected_rows:
            raise ValueError("position-shift row coverage differs")
        current_identities = [
            (str(row["request_id"]), str(row["candidate_item_id"])) for row in rows
        ]
        if len(set(current_identities)) != expected_rows:
            raise ValueError("position-shift row identity is duplicated")
        if identities is None:
            identities = current_identities
        elif current_identities != identities:
            raise ValueError("position-shift model row identity/order differs")
        path_names = EXPECTED_PATHS[model_id]
        for row in rows:
            if set(row.get("paths", {})) != set(path_names):
                raise ValueError("position-shift native path coverage differs")

        for path_name in path_names:
            labels = None
            shifts_by_position: dict[str, list[int]] = {}
            full_positions_by_label: dict[str, list[int]] = {}
            null_positions_by_label: dict[str, list[int]] = {}
            sequence_shifts = []
            for row in rows:
                path = row["paths"][path_name]
                full_positions = _integer_positions(path, "full_positions")
                null_positions = _integer_positions(path, "null_positions")
                if len(full_positions) != len(null_positions) or len(full_positions) < 3:
                    raise ValueError("position-shift semantic-position shape differs")
                current_labels = ("query_end", "history_summary_end") + tuple(
                    f"native_readout_{index}"
                    for index in range(len(full_positions) - 2)
                )
                if labels is None:
                    labels = current_labels
                    shifts_by_position = {label: [] for label in labels}
                    full_positions_by_label = {label: [] for label in labels}
                    null_positions_by_label = {label: [] for label in labels}
                elif current_labels != labels:
                    raise ValueError("position-shift semantic-position count differs")
                full_length = _positive_integer(path, "full_sequence_length")
                null_length = _positive_integer(path, "null_sequence_length")
                if (
                    any(not 0 <= value < full_length for value in full_positions)
                    or any(not 0 <= value < null_length for value in null_positions)
                    or any(left >= right for left, right in zip(full_positions, full_positions[1:]))
                    or any(left >= right for left, right in zip(null_positions, null_positions[1:]))
                ):
                    raise ValueError("position-shift semantic position is invalid")
                sequence_shift = full_length - null_length
                sequence_shifts.append(sequence_shift)
                for label, full_value, null_value in zip(
                    labels, full_positions, null_positions
                ):
                    shift = full_value - null_value
                    shifts_by_position[label].append(shift)
                    full_positions_by_label[label].append(full_value)
                    null_positions_by_label[label].append(null_value)
                    if label == "query_end":
                        all_query_shifts.append(shift)
                    else:
                        all_downstream_shifts.append(shift)
                native_matches_sequence_shift.append(
                    shifts_by_position[labels[-1]][-1] == sequence_shift
                )
            assert labels is not None
            path_cells.append(
                {
                    "method_id": model_id,
                    "path": path_name,
                    "rows": expected_rows,
                    "sequence_length_full_minus_null": _summary(sequence_shifts),
                    "positions": {
                        label: {
                            "full_position": _summary(full_positions_by_label[label]),
                            "null_position": _summary(null_positions_by_label[label]),
                            "full_minus_null_position": _summary(
                                shifts_by_position[label]
                            ),
                        }
                        for label in labels
                    },
                }
            )

    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d3_attention_position_shift",
        "status": "completed",
        "descriptive_only": True,
        "qrels_read": False,
        "source_test_opened": False,
        "models": list(MODELS),
        "rows_per_model": expected_rows,
        "path_cells": path_cells,
        "invariants": {
            "all_query_position_shifts_zero": all(
                value == 0 for value in all_query_shifts
            ),
            "query_position_shift_zero_fraction": _fraction(
                all_query_shifts, lambda value: value == 0
            ),
            "downstream_position_shift_nonzero_fraction": _fraction(
                all_downstream_shifts, lambda value: value != 0
            ),
            "native_readout_shift_matches_sequence_length_shift_fraction": float(
                np.mean(native_matches_sequence_shift)
            ),
        },
        "interpretation_boundary": (
            "Exact default-Qwen padded-sequence position shifts induced by replacing "
            "full history with the null marker. This identifies a RoPE comparison "
            "confound; it does not measure preference content or establish a causal "
            "position bottleneck."
        ),
    }


def _integer_positions(path: Mapping[str, Any], key: str) -> tuple[int, ...]:
    values = path.get(key)
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in values
    ):
        raise ValueError("position-shift indices must be integer lists")
    return tuple(values)


def _positive_integer(path: Mapping[str, Any], key: str) -> int:
    value = path.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("position-shift sequence length must be a positive integer")
    return value


def _summary(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("position-shift summary values differ")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": int(array.min()),
        "maximum": int(array.max()),
        "standard_deviation": float(array.std()),
        "zero_fraction": float(np.mean(array == 0)),
        "positive_fraction": float(np.mean(array > 0)),
        "negative_fraction": float(np.mean(array < 0)),
        "rows": int(array.size),
    }


def _fraction(values: Sequence[Any], predicate: Any) -> float:
    if not values:
        raise ValueError("position-shift invariant values are empty")
    return float(np.mean([bool(predicate(value)) for value in values]))
