from __future__ import annotations

import pytest

from myrec.mechanism.native_readout_diagnostics import (
    summarize_native_readout_diagnostics,
)
from myrec.mechanism.native_readout_runtime import Q2_NATIVE_READOUT_CONDITIONS
from myrec.mechanism.q3_native_readout_runtime import Q3_NATIVE_READOUT_CONDITIONS


def _conditions(names, full, null):
    return {
        name: float(full if name == "baseline_full" else null if name == "baseline_null" else 0.0)
        for name in names
    }


def _requests():
    q2, q3 = [], []
    for request_index in range(2):
        q2_rows, q3_rows = [], []
        for candidate_index in range(2):
            delta_prompt = float(request_index + 1)
            delta_context = float(candidate_index + 1)
            full_terms = [4.0, 6.0, 2.0, 2.0]
            null_terms = [4.0 - 2 * delta_prompt, 6.0 - 2 * delta_context, 2.0, 2.0]
            full_score = 3.0
            null_score = full_score - delta_prompt - delta_context
            identity = f"i{request_index}{candidate_index}"
            q2_rows.append(
                {
                    "request_id": f"r{request_index}",
                    "candidate_item_id": identity,
                    "candidate_ordinal": candidate_index,
                    "conditions": _conditions(Q2_NATIVE_READOUT_CONDITIONS, 2.0, 1.0),
                    "readout_diagnostics": {
                        "full_common_offset": 5.0,
                        "null_common_offset": 3.0,
                        "full_input_norm": 4.0,
                        "full_output_norm": 2.0,
                        "full_input_output_cosine": 0.75,
                    },
                }
            )
            q3_rows.append(
                {
                    "request_id": f"r{request_index}",
                    "candidate_item_id": identity,
                    "candidate_ordinal": candidate_index,
                    "conditions": _conditions(Q3_NATIVE_READOUT_CONDITIONS, full_score, null_score),
                    "readout_diagnostics": {
                        "full_terms": full_terms,
                        "null_terms": null_terms,
                        "full_prompt_contrast": 1.0,
                        "full_context_contrast": 2.0,
                        "full_yes_input_norm": [4.0, 6.0],
                        "full_yes_output_norm": [2.0, 3.0],
                        "full_no_input_norm": [4.0, 8.0],
                        "full_no_output_norm": [2.0, 4.0],
                    },
                }
            )
        q2.append({"ordinal": request_index, "request_id": f"r{request_index}", "rows": q2_rows})
        q3.append({"ordinal": request_index, "request_id": f"r{request_index}", "rows": q3_rows})
    return q2, q3


def test_native_readout_diagnostics_hand_decompose_and_preserve_geometry():
    q2, q3 = _requests()
    result = summarize_native_readout_diagnostics(
        q2, q3, expected_requests=2, expected_score_rows=4
    )
    assert result["q2"]["full_minus_null_native_score"]["mean"] == pytest.approx(1.0)
    assert result["q2"]["full_minus_null_common_logit_offset"]["mean"] == pytest.approx(2.0)
    assert result["q2"]["full_final_rmsnorm_geometry"]["output_to_input_norm_ratio"]["mean"] == pytest.approx(0.5)
    assert result["q3"]["full_minus_null_prompt_contrast"]["mean"] == pytest.approx(1.5)
    assert result["q3"]["full_minus_null_context_contrast"]["mean"] == pytest.approx(1.5)
    assert result["q3"]["maximum_score_recomposition_error"] == pytest.approx(0.0)
    assert result["q3"]["full_final_rmsnorm_geometry"]["yes_context"]["ratio"]["mean"] == pytest.approx(0.5)


def test_native_readout_diagnostics_fail_closed_on_qrels_and_identity():
    q2, q3 = _requests()
    with pytest.raises(ValueError, match="qrels/source-test blind"):
        summarize_native_readout_diagnostics(
            q2,
            q3,
            expected_requests=2,
            expected_score_rows=4,
            qrels_read=True,
        )
    q3[0]["rows"][0]["candidate_item_id"] = "wrong"
    with pytest.raises(ValueError, match="candidate identity/order differs"):
        summarize_native_readout_diagnostics(
            q2, q3, expected_requests=2, expected_score_rows=4
        )
