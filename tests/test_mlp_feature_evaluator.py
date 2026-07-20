from __future__ import annotations

import json

import pytest

from myrec.mechanism.attention_pattern_analysis import BLOCKS, MODELS
from myrec.mechanism.mlp_feature_evaluator import (
    DECOMPOSITION_SCALARS,
    DECOMPOSITION_TERM_METRICS,
    DECOMPOSITION_TERMS,
    STAGE_METRICS,
    evaluate_mlp_feature_bundles,
)
from myrec.mechanism.mlp_feature_formation import MLP_FEATURE_STAGES
from myrec.utils.hashing import sha256_file


def _summary(model_id, value):
    positions = 3 if model_id == MODELS[0] else 4
    return {
        "positions": [
            {
                "position_index": position,
                "groups": [
                    {
                        "group_id": group,
                        "dimensions": 192,
                        "stages": {
                            stage: {
                                metric: float(value + group)
                                for metric in STAGE_METRICS
                            }
                            for stage in MLP_FEATURE_STAGES
                        },
                        "product_delta_decomposition": {
                            **{
                                term: {
                                    metric: float(value + group)
                                    for metric in DECOMPOSITION_TERM_METRICS
                                }
                                for term in DECOMPOSITION_TERMS
                            },
                            **{
                                metric: float(value + group)
                                for metric in DECOMPOSITION_SCALARS
                            },
                        },
                    }
                    for group in range(16)
                ],
            }
            for position in range(positions)
        ],
        "groups": 16,
        "maximum_product_delta_recomposition_abs_error": 0.0,
        "maximum_actual_product_quantization_abs_error": 0.0,
        "maximum_product_identity_low_precision_ratio": 0.0,
        "interpretation_boundary": "synthetic",
    }


def _write_bundle(root, model_id, block, *, qrels_read=False):
    root.mkdir(parents=True)
    path_names = ("prompt",) if model_id == MODELS[0] else ("yes", "no")
    rows = []
    for ordinal in range(2):
        rows.append(
            {
                "ordinal": ordinal,
                "request_id": f"r{ordinal}",
                "candidate_item_id": f"i{ordinal}",
                "candidate_ordinal": ordinal,
                "selection_sha256": f"s{ordinal}",
                "maximum_score_identity_delta": 0.0,
                "paths": {
                    path_name: {
                        "capture_position_order": [
                            "query_end",
                            "history_summary_end",
                            "native_readout...",
                        ],
                        "summary": _summary(model_id, ordinal + 1),
                    }
                    for path_name in path_names
                },
            }
        )
    rows_path = root / "rows.jsonl"
    rows_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    metadata = {
        "status": "completed",
        "analysis_stage": "transformer_deep_dive_d4_mlp_feature_formation_extension",
        "method_id": model_id,
        "block_zero_based": block,
        "result_eligible": True,
        "confirmatory_family_member": False,
        "layer_or_group_selection_authorized": False,
        "qrels_read": qrels_read,
        "source_test_opened": False,
        "complete_finite_observation_coverage": True,
        "observation_rows": 2,
        "rows_path": str(rows_path),
        "rows_sha256": sha256_file(rows_path),
        "implementation_identity": {"digest": "fixed"},
        "run_contract": {"target_rows": 2, "implementation_digest": "fixed"},
        "maximum_score_identity_delta": 0.0,
        "maximum_product_recomposition_low_precision_ratio": 0.0,
        "maximum_delta_recomposition_abs_error": 0.0,
        "maximum_actual_product_quantization_abs_error": 0.0,
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, sort_keys=True), encoding="utf-8"
    )


def _bundles(tmp_path):
    bundles = {}
    for model_id in MODELS:
        bundles[model_id] = {}
        for block in BLOCKS:
            root = tmp_path / f"{model_id}_{block}"
            _write_bundle(root, model_id, block)
            bundles[model_id][block] = root
    return bundles


def test_mlp_feature_evaluator_requires_and_aggregates_complete_grid(tmp_path):
    result = evaluate_mlp_feature_bundles(_bundles(tmp_path), expected_rows=2)
    assert len(result["cells"]) == 6
    assert result["qrels_read"] is False
    q2 = result["cells"][0]
    mean = q2["paths"]["prompt"]["positions"]["query_end"]["groups"][0][
        "stages"
    ]["gate_pre"]["full_rms"]["mean"]
    assert mean == pytest.approx(1.5)


def test_mlp_feature_evaluator_rejects_qrels_or_missing_cell(tmp_path):
    bundles = _bundles(tmp_path)
    root = bundles[MODELS[0]][BLOCKS[0]]
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["qrels_read"] = True
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="completed bundle boundary differs"):
        evaluate_mlp_feature_bundles(bundles, expected_rows=2)
    del bundles[MODELS[1]][BLOCKS[-1]]
    with pytest.raises(ValueError, match="bundle coverage differs"):
        evaluate_mlp_feature_bundles(bundles, expected_rows=2)
