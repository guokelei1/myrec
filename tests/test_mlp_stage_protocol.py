from __future__ import annotations

from pathlib import Path

from myrec.mechanism.mlp_stage_runtime import (
    N12_MANIFEST_PATH,
    N12_MANIFEST_SHA256,
    _load_n12_manifest,
    mlp_stage_runtime_implementation_identity,
)
from myrec.mechanism.mlp_stage_scoring import (
    ACTIVE_STAGE_CONDITIONS,
    MLP_STAGE_CONDITIONS,
)
from myrec.utils.hashing import sha256_file


def test_n12_manifest_is_frozen_and_diagnostic_only() -> None:
    assert Path(N12_MANIFEST_PATH).is_file()
    assert sha256_file(N12_MANIFEST_PATH) == N12_MANIFEST_SHA256
    manifest = _load_n12_manifest(N12_MANIFEST_PATH)
    assert manifest["frozen_inputs"]["request_count"] == 8000
    assert manifest["frozen_inputs"]["blocks"] == [13, 20, 27]
    assert manifest["claim_boundary"]["diagnostic_only"] is True
    assert manifest["claim_boundary"]["architecture_authorized"] is False


def test_n12_condition_registry_covers_gate_up_and_joint_controls() -> None:
    assert MLP_STAGE_CONDITIONS[:4] == (
        "baseline_full",
        "baseline_null",
        "full_gate_identity",
        "null_gate_identity",
    )
    assert "null_gate_from_full" in ACTIVE_STAGE_CONDITIONS
    assert "null_up_from_full" in ACTIVE_STAGE_CONDITIONS
    assert "null_joint_from_full" in ACTIVE_STAGE_CONDITIONS
    assert "full_gate_from_null" in ACTIVE_STAGE_CONDITIONS
    assert "full_joint_from_null" in ACTIVE_STAGE_CONDITIONS
    assert len(set(MLP_STAGE_CONDITIONS)) == len(MLP_STAGE_CONDITIONS)


def test_n12_implementation_digest_covers_stage_hook_and_cli() -> None:
    identity = mlp_stage_runtime_implementation_identity()
    paths = {item["path"] for item in identity["files"]}
    assert "src/myrec/mechanism/mlp_stage_interventions.py" in paths
    assert "src/myrec/mechanism/mlp_stage_scoring.py" in paths
    assert "scripts/score_deep_dive_mlp_stage.py" in paths
    assert identity["digest"]

