from __future__ import annotations

import pytest

from myrec.mechanism.attention_pattern_analysis import (
    BLOCKS,
    MODELS,
    concentration_summary,
    summarize_attention_patterns,
)


def test_attention_concentration_is_hand_computed():
    result = concentration_summary([3.0, 1.0], top_k=2)
    assert result["shares"] == pytest.approx([0.75, 0.25])
    assert result["effective_count_simpson"] == pytest.approx(1.6)
    assert result["top_indices"] == [0, 1]
    assert result["top_share"] == pytest.approx(0.75)
    assert result["top_k_share"] == pytest.approx(1.0)


def test_attention_pattern_synthesis_covers_every_fixed_cell():
    def path(group_scale, head_scale):
        observations = {}
        for scope in ("history_summary", "native_readout"):
            observations[scope] = {}
            for span in ("query", "history", "candidate"):
                observations[scope][span] = {
                    metric: {
                        "gqa_group": {
                            "mean": [group_scale * (index + 1) for index in range(8)]
                        },
                        "query_head": {
                            "mean": [head_scale * (index + 1) for index in range(16)]
                        },
                    }
                    for metric in (
                        "attention_mass",
                        "o_proj_contribution_norm",
                    )
                }
        return {"observations": observations}

    results = {}
    for model_id in MODELS:
        results[model_id] = {}
        for block in BLOCKS:
            paths = (
                {"prompt": path(1.0, 1.0)}
                if model_id == MODELS[0]
                else {"yes": path(1.0, 1.0), "no": path(3.0, 3.0)}
            )
            results[model_id][str(block)] = {
                "paths": paths,
                "maximum_manual_attention_low_precision_ratio": 0.5,
                "maximum_score_identity_delta": 0.0,
            }
    metrics = {
        "analysis_type": "transformer_deep_dive_d3_attention_head_observation",
        "status": "completed",
        "descriptive_only": True,
        "qrels_read": False,
        "source_test_opened": False,
        "blocks": list(BLOCKS),
        "results": results,
    }
    result = summarize_attention_patterns(metrics)
    assert len(result["cells"]) == 6
    assert result["qrels_read"] is False
    assert all(
        row["axes"]["gqa_group"]["history_attention_mass"]["top_indices"][0]
        == 7
        for row in result["cells"]
    )
    q3 = next(row for row in result["cells"] if row["method_id"] == MODELS[1])
    assert q3["axes"]["gqa_group"]["history_attention_mass"]["values"][0] == 2.0


def test_attention_pattern_synthesis_rejects_qrels_or_incomplete_blocks():
    with pytest.raises(ValueError, match="boundary differs"):
        summarize_attention_patterns(
            {
                "analysis_type": "transformer_deep_dive_d3_attention_head_observation",
                "status": "completed",
                "descriptive_only": True,
                "qrels_read": True,
                "source_test_opened": False,
                "blocks": list(BLOCKS),
                "results": {},
            }
        )
