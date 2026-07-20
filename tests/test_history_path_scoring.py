from __future__ import annotations

from pathlib import Path

from myrec.mechanism.history_path_evaluator import FAMILY_SIZE
from myrec.mechanism.history_path_runtime import N9_MANIFEST_SHA256, _load_n9_manifest
from myrec.mechanism.history_path_scoring import N9_SCORE_CONDITIONS


def test_n9_manifest_is_hashed_and_condition_order_is_frozen():
    path = Path("experiments/motivation/transformer_n9_history_path_manifest_v1.yaml")
    manifest = _load_n9_manifest(path)
    assert manifest["_sha256"] == N9_MANIFEST_SHA256
    assert tuple(manifest["conditions"]) == N9_SCORE_CONDITIONS
    assert manifest["scope"]["blocks"] == [13, 20, 27]
    assert manifest["path_definition"]["position_ids_unchanged"] is True


def test_n9_registered_family_has_fixed_cross_model_block_endpoints():
    assert FAMILY_SIZE == 24
