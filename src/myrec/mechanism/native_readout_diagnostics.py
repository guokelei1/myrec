"""Qrels-blind descriptive diagnostics for completed Q2/Q3 native readouts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.native_readout_runtime import (
    Q2_METHOD_ID,
    Q2_NATIVE_READOUT_CONDITIONS,
)
from myrec.mechanism.q3_native_readout_runtime import (
    Q3_METHOD_ID,
    Q3_NATIVE_READOUT_CONDITIONS,
)


Q2_DIAGNOSTIC_KEYS = {
    "full_common_offset",
    "null_common_offset",
    "full_input_norm",
    "full_output_norm",
    "full_input_output_cosine",
}
Q3_DIAGNOSTIC_KEYS = {
    "full_terms",
    "null_terms",
    "full_prompt_contrast",
    "full_context_contrast",
    "full_yes_input_norm",
    "full_yes_output_norm",
    "full_no_input_norm",
    "full_no_output_norm",
}


def summarize_native_readout_diagnostics(
    q2_requests: Sequence[Mapping[str, Any]],
    q3_requests: Sequence[Mapping[str, Any]],
    *,
    expected_requests: int,
    expected_score_rows: int,
    qrels_read: bool = False,
    source_test_opened: bool = False,
) -> dict[str, Any]:
    """Summarize score-nullspace and final-norm geometry after both families close."""

    if qrels_read is not False or source_test_opened is not False:
        raise ValueError("native-readout diagnostics must remain qrels/source-test blind")
    q2 = _flatten_requests(
        q2_requests,
        method_id=Q2_METHOD_ID,
        expected_conditions=set(Q2_NATIVE_READOUT_CONDITIONS),
        expected_diagnostics=Q2_DIAGNOSTIC_KEYS,
        expected_requests=expected_requests,
        expected_score_rows=expected_score_rows,
    )
    q3 = _flatten_requests(
        q3_requests,
        method_id=Q3_METHOD_ID,
        expected_conditions=set(Q3_NATIVE_READOUT_CONDITIONS),
        expected_diagnostics=Q3_DIAGNOSTIC_KEYS,
        expected_requests=expected_requests,
        expected_score_rows=expected_score_rows,
    )
    if q2["identities"] != q3["identities"]:
        raise ValueError("native-readout Q2/Q3 candidate identity/order differs")

    q2_rows = q2["rows"]
    q2_delta_score = np.asarray(
        [row["conditions"]["baseline_full"] - row["conditions"]["baseline_null"] for row in q2_rows]
    )
    q2_delta_common = np.asarray(
        [
            row["readout_diagnostics"]["full_common_offset"]
            - row["readout_diagnostics"]["null_common_offset"]
            for row in q2_rows
        ]
    )
    q2_input_norm = np.asarray(
        [row["readout_diagnostics"]["full_input_norm"] for row in q2_rows]
    )
    q2_output_norm = np.asarray(
        [row["readout_diagnostics"]["full_output_norm"] for row in q2_rows]
    )
    q2_cosine = np.asarray(
        [row["readout_diagnostics"]["full_input_output_cosine"] for row in q2_rows]
    )

    q3_rows = q3["rows"]
    q3_delta_score = []
    q3_delta_prompt = []
    q3_delta_context = []
    q3_delta_prompt_common = []
    q3_delta_context_common = []
    q3_score_recomposition_errors = []
    q3_norms: dict[str, dict[str, list[float]]] = {
        state: {"input": [], "output": [], "ratio": []}
        for state in ("shared_prompt", "yes_context", "no_context")
    }
    shared_norm_deltas = []
    for row in q3_rows:
        diagnostics = row["readout_diagnostics"]
        full = _finite_vector(diagnostics["full_terms"], size=4)
        null = _finite_vector(diagnostics["null_terms"], size=4)
        full_prompt = 0.5 * (full[0] - full[2])
        null_prompt = 0.5 * (null[0] - null[2])
        full_context = 0.5 * (full[1] - full[3])
        null_context = 0.5 * (null[1] - null[3])
        delta_score = float(
            row["conditions"]["baseline_full"]
            - row["conditions"]["baseline_null"]
        )
        delta_prompt = full_prompt - null_prompt
        delta_context = full_context - null_context
        q3_delta_score.append(delta_score)
        q3_delta_prompt.append(delta_prompt)
        q3_delta_context.append(delta_context)
        q3_delta_prompt_common.append(
            0.5 * ((full[0] + full[2]) - (null[0] + null[2]))
        )
        q3_delta_context_common.append(
            0.5 * ((full[1] + full[3]) - (null[1] + null[3]))
        )
        q3_score_recomposition_errors.append(
            abs(delta_score - delta_prompt - delta_context)
        )
        yes_input = _finite_vector(diagnostics["full_yes_input_norm"], size=2)
        yes_output = _finite_vector(diagnostics["full_yes_output_norm"], size=2)
        no_input = _finite_vector(diagnostics["full_no_input_norm"], size=2)
        no_output = _finite_vector(diagnostics["full_no_output_norm"], size=2)
        shared_norm_deltas.extend(
            [abs(yes_input[0] - no_input[0]), abs(yes_output[0] - no_output[0])]
        )
        _append_norm(q3_norms["shared_prompt"], 0.5 * (yes_input[0] + no_input[0]), 0.5 * (yes_output[0] + no_output[0]))
        _append_norm(q3_norms["yes_context"], yes_input[1], yes_output[1])
        _append_norm(q3_norms["no_context"], no_input[1], no_output[1])

    request_groups = _request_groups(q2_requests, q3_requests)
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d6_native_readout_diagnostics",
        "status": "completed",
        "descriptive_only": True,
        "qrels_read": False,
        "source_test_opened": False,
        "request_count": expected_requests,
        "score_rows": expected_score_rows,
        "q2": {
            "full_minus_null_native_score": _summary(q2_delta_score),
            "full_minus_null_common_logit_offset": _summary(q2_delta_common),
            "score_common_offset_pearson": _pearson(q2_delta_score, q2_delta_common),
            "common_offset_rms_to_native_score_rms_ratio": _rms_ratio(
                q2_delta_common, q2_delta_score
            ),
            "request_level": _request_level_summary(
                q2_delta_score, q2_delta_common, request_groups
            ),
            "full_final_rmsnorm_geometry": {
                "input_norm": _summary(q2_input_norm),
                "output_norm": _summary(q2_output_norm),
                "output_to_input_norm_ratio": _summary(
                    q2_output_norm / np.maximum(q2_input_norm, 1.0e-12)
                ),
                "input_output_cosine": _summary(q2_cosine),
            },
            "algebra_boundary": (
                "The common logit offset is exactly removed by Q2's Yes-minus-No "
                "readout and is therefore a descriptive score-nullspace coordinate."
            ),
        },
        "q3": {
            "full_minus_null_native_score": _summary(q3_delta_score),
            "full_minus_null_prompt_contrast": _summary(q3_delta_prompt),
            "full_minus_null_context_contrast": _summary(q3_delta_context),
            "prompt_context_delta_pearson": _pearson(
                q3_delta_prompt, q3_delta_context
            ),
            "prompt_rms_to_score_rms_ratio": _rms_ratio(
                q3_delta_prompt, q3_delta_score
            ),
            "context_rms_to_score_rms_ratio": _rms_ratio(
                q3_delta_context, q3_delta_score
            ),
            "maximum_score_recomposition_error": float(
                max(q3_score_recomposition_errors)
            ),
            "full_minus_null_prompt_common_mode": _summary(
                q3_delta_prompt_common
            ),
            "full_minus_null_context_common_mode": _summary(
                q3_delta_context_common
            ),
            "full_final_rmsnorm_geometry": {
                state: {
                    key: _summary(values) for key, values in geometry.items()
                }
                for state, geometry in q3_norms.items()
            },
            "maximum_shared_prompt_yes_no_norm_delta": float(
                max(shared_norm_deltas)
            ),
            "algebra_boundary": (
                "Q3 score change is exactly prompt-contrast plus continuation-"
                "contrast change. Common modes cancel algebraically; term and norm "
                "summaries do not localize an earlier Transformer cause."
            ),
        },
        "interpretation_boundary": (
            "Complete all-candidate, qrels-blind native-readout diagnostics. They "
            "separate score-bearing and algebraically cancelled coordinates but do "
            "not measure ranking utility, establish mediation, or authorize a method."
        ),
    }


def _flatten_requests(
    requests: Sequence[Mapping[str, Any]],
    *,
    method_id: str,
    expected_conditions: set[str],
    expected_diagnostics: set[str],
    expected_requests: int,
    expected_score_rows: int,
) -> dict[str, Any]:
    if len(requests) != expected_requests:
        raise ValueError(f"{method_id} native-readout request coverage differs")
    rows = []
    identities = []
    seen_requests = set()
    for ordinal, request in enumerate(requests):
        request_id = str(request.get("request_id") or "")
        if request.get("ordinal") != ordinal or not request_id or request_id in seen_requests:
            raise ValueError(f"{method_id} native-readout request identity differs")
        seen_requests.add(request_id)
        candidate_rows = request.get("rows")
        if not isinstance(candidate_rows, list) or not candidate_rows:
            raise ValueError(f"{method_id} native-readout candidate rows differ")
        for candidate_ordinal, row in enumerate(candidate_rows):
            if (
                str(row.get("request_id") or "") != request_id
                or row.get("candidate_ordinal") != candidate_ordinal
                or not str(row.get("candidate_item_id") or "")
                or set(row.get("conditions", {})) != expected_conditions
                or set(row.get("readout_diagnostics", {})) != expected_diagnostics
            ):
                raise ValueError(f"{method_id} native-readout row schema differs")
            _assert_finite_tree(row["conditions"])
            _assert_finite_tree(row["readout_diagnostics"])
            identities.append(
                (request_id, str(row["candidate_item_id"]), candidate_ordinal)
            )
            rows.append(row)
    if len(rows) != expected_score_rows or len(set(identities)) != expected_score_rows:
        raise ValueError(f"{method_id} native-readout score coverage differs")
    return {"rows": rows, "identities": identities}


def _request_groups(q2_requests, q3_requests) -> list[int]:
    q2_counts = [len(request["rows"]) for request in q2_requests]
    q3_counts = [len(request["rows"]) for request in q3_requests]
    if q2_counts != q3_counts:
        raise ValueError("native-readout per-request candidate counts differ")
    return q2_counts


def _request_level_summary(
    score: np.ndarray, common: np.ndarray, counts: Sequence[int]
) -> dict[str, Any]:
    score_means, score_stds, common_means, common_stds = [], [], [], []
    start = 0
    for count in counts:
        end = start + count
        score_means.append(float(score[start:end].mean()))
        score_stds.append(float(score[start:end].std()))
        common_means.append(float(common[start:end].mean()))
        common_stds.append(float(common[start:end].std()))
        start = end
    if start != len(score):
        raise ValueError("native-readout request grouping differs")
    return {
        "native_score_candidate_mean": _summary(score_means),
        "native_score_within_request_std": _summary(score_stds),
        "common_offset_candidate_mean": _summary(common_means),
        "common_offset_within_request_std": _summary(common_stds),
    }


def _append_norm(target: dict[str, list[float]], input_norm: float, output_norm: float) -> None:
    if input_norm < 0 or output_norm < 0:
        raise ValueError("native-readout norm is negative")
    target["input"].append(float(input_norm))
    target["output"].append(float(output_norm))
    target["ratio"].append(float(output_norm / max(input_norm, 1.0e-12)))


def _finite_vector(value: Any, *, size: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError("native-readout diagnostic vector differs")
    return array


def _assert_finite_tree(value: Any) -> None:
    if isinstance(value, Mapping):
        for child in value.values():
            _assert_finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _assert_finite_tree(child)
    elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError("native-readout diagnostic contains a non-finite value")


def _summary(values: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("native-readout summary values differ")
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "standard_deviation": float(array.std()),
        "rms": float(np.sqrt(np.mean(array * array))),
        "minimum": float(array.min()),
        "maximum": float(array.max()),
        "rows": int(array.size),
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape or left_array.ndim != 1 or left_array.size < 2:
        raise ValueError("native-readout correlation arrays differ")
    if left_array.std() == 0 or right_array.std() == 0:
        return {"defined": False, "value": None, "rows": int(left_array.size)}
    return {
        "defined": True,
        "value": float(np.corrcoef(left_array, right_array)[0, 1]),
        "rows": int(left_array.size),
    }


def _rms_ratio(numerator: Sequence[float], denominator: Sequence[float]) -> float | None:
    numerator_array = np.asarray(numerator, dtype=np.float64)
    denominator_array = np.asarray(denominator, dtype=np.float64)
    denominator_rms = float(np.sqrt(np.mean(denominator_array * denominator_array)))
    if denominator_rms == 0:
        return None
    return float(np.sqrt(np.mean(numerator_array * numerator_array)) / denominator_rms)
