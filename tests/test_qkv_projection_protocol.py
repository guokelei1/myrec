from pathlib import Path

import yaml

from myrec.mechanism.qkv_projection_runtime import (
    N13_MANIFEST_SHA256,
    N13_MANIFEST_PATH,
    _load_n13_manifest,
    qkv_projection_runtime_implementation_identity,
)
from myrec.mechanism.qkv_projection_scoring import QKV_PROJECTION_CONDITIONS


def test_n13_manifest_digest_and_conditions_are_frozen():
    manifest = _load_n13_manifest(N13_MANIFEST_PATH)
    assert manifest["_sha256"] == N13_MANIFEST_SHA256
    assert manifest["frozen_inputs"]["request_count"] == 8000
    assert tuple(manifest["conditions"]) == QKV_PROJECTION_CONDITIONS
    assert tuple(manifest["identity_conditions"]) == tuple(
        name for name in QKV_PROJECTION_CONDITIONS if name.endswith("_identity")
    )


def test_n13_manifest_file_is_valid_yaml_mapping():
    path = Path(N13_MANIFEST_PATH)
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    assert value["claim_boundary"]["diagnostic_only"] is True
    assert value["claim_boundary"]["architecture_authorized"] is False


def test_n13_implementation_digest_covers_cli_and_operator():
    identity = qkv_projection_runtime_implementation_identity()
    paths = {item["path"] for item in identity["files"]}
    assert "src/myrec/mechanism/qkv_projection_interventions.py" in paths
    assert "scripts/score_deep_dive_qkv_projection.py" in paths
    assert len(identity["digest"]) == 64
