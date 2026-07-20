from pathlib import Path

import yaml

from myrec.mechanism.embedding_stage_runtime import (
    N14_MANIFEST_PATH,
    N14_MANIFEST_SHA256,
    _load_n14_manifest,
    embedding_stage_runtime_implementation_identity,
)
from myrec.mechanism.embedding_stage_scoring import EMBEDDING_STAGE_CONDITIONS


def test_n14_manifest_digest_and_conditions_are_frozen():
    manifest = _load_n14_manifest(N14_MANIFEST_PATH)
    assert manifest["_sha256"] == N14_MANIFEST_SHA256
    assert manifest["frozen_inputs"]["request_count"] == 8000
    assert tuple(manifest["conditions"]) == EMBEDDING_STAGE_CONDITIONS
    assert tuple(manifest["identity_conditions"]) == (
        "full_embedding_identity",
        "null_embedding_identity",
    )


def test_n14_claim_boundary_is_diagnostic_only():
    value = yaml.safe_load(Path(N14_MANIFEST_PATH).read_text(encoding="utf-8"))
    assert value["claim_boundary"]["diagnostic_only"] is True
    assert value["claim_boundary"]["architecture_authorized"] is False


def test_n14_digest_covers_embedding_operator_and_cli():
    identity = embedding_stage_runtime_implementation_identity()
    paths = {item["path"] for item in identity["files"]}
    assert "src/myrec/mechanism/embedding_stage_interventions.py" in paths
    assert "scripts/score_deep_dive_embedding_stage.py" in paths
    assert len(identity["digest"]) == 64
