from __future__ import annotations

import pytest

from myrec.mechanism.attention_pattern_analysis import BLOCKS, MODELS
from myrec.mechanism.attention_qk_geometry_analysis import (
    STAGES,
    summarize_attention_qk_geometry,
)


def _synthetic_source():
    results = {}
    for model_id in MODELS:
        results[model_id] = {}
        for block in BLOCKS:
            path_names = ("prompt",) if model_id == MODELS[0] else ("yes", "no")
            paths = {}
            for path_index, path_name in enumerate(path_names):
                qk = {}
                for tensor, size in (("q", 16), ("k", 8)):
                    qk[tensor] = {}
                    for stage in STAGES:
                        scale = float(path_index + 1)
                        qk[tensor][stage] = {
                            "full_norm": {
                                "p": {"mean": [2.0 * scale] * size}
                            },
                            "null_norm": {
                                "p": {"mean": [2.0 * scale] * size}
                            },
                            "full_null_delta_norm": {
                                "p": {"mean": [1.0 * scale] * size}
                            },
                            "full_null_cosine": {
                                "p": {"mean": [0.5] * size}
                            },
                        }
                paths[path_name] = {"qk_geometry": qk}
            results[model_id][str(block)] = {"paths": paths}
    return {
        "analysis_type": "transformer_deep_dive_d3_attention_head_observation",
        "status": "completed",
        "descriptive_only": True,
        "qrels_read": False,
        "source_test_opened": False,
        "blocks": list(BLOCKS),
        "results": results,
    }


def test_qk_geometry_covers_complete_fixed_grid_and_hand_values():
    result = summarize_attention_qk_geometry(_synthetic_source())
    assert len(result["cells"]) == 6
    assert len(result["path_cells"]) == 9
    rmsnorm = result["stage_transition_consistency"]["qk_rmsnorm_pre_to_post"]
    assert rmsnorm["overall"]["comparisons"] == 18
    assert rmsnorm["overall"]["relative_delta_unchanged"] == 18
    rope = result["stage_transition_consistency"]["rope_post_norm_to_post_rope"]
    assert rope["overall"]["cosine_unchanged"] == 18
    assert result["maximum_rope_norm_relative_l2_error"] == 0.0
    q2 = result["cells"][0]["tensors"]["q"]["stages"]["pre_norm"]["p"]
    assert q2["mean_relative_delta"] == pytest.approx(0.5)
    assert q2["mean_full_null_cosine"] == pytest.approx(0.5)
    q3 = next(row for row in result["cells"] if row["method_id"] == MODELS[1])
    # Averaging path scales 1 and 2 gives full norm 3 and delta norm 1.5.
    q3_q = q3["tensors"]["q"]["stages"]["pre_norm"]["p"]
    assert q3_q["mean_full_norm"] == pytest.approx(3.0)
    assert q3_q["mean_full_null_delta_norm"] == pytest.approx(1.5)
    q3_paths = [
        row
        for row in result["path_cells"]
        if row["method_id"] == MODELS[1] and row["block_zero_based"] == BLOCKS[0]
    ]
    assert [row["path"] for row in q3_paths] == ["yes", "no"]
    assert q3_paths[0]["tensors"]["q"]["stages"]["pre_norm"]["p"][
        "mean_full_norm"
    ] == pytest.approx(2.0)
    assert q3_paths[1]["tensors"]["q"]["stages"]["pre_norm"]["p"][
        "mean_full_norm"
    ] == pytest.approx(4.0)


def test_qk_geometry_rejects_qrels_access():
    source = _synthetic_source()
    source["qrels_read"] = True
    with pytest.raises(ValueError, match="source boundary differs"):
        summarize_attention_qk_geometry(source)
