from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from myrec.mechanism.supplemental_evidence_registry import (
    EXPECTED_SUPPLEMENT_IDS,
    REGISTRY_PATH,
    _audit_payload_contract,
    _validate_registry_schema,
    audit_supplemental_evidence_registry,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_supplement_registry_is_exhaustive_and_pending_only_future_outputs():
    result = audit_supplemental_evidence_registry(ROOT)
    assert result["status"] == "pending"
    assert result["entry_count"] == len(EXPECTED_SUPPLEMENT_IDS) == 21
    assert result["status_counts"] == {"completed": 17, "pending": 4}
    assert result["failures"] == []
    assert result["design_ranking_entry"] == (
        "component_functional_design_gate_synthesis"
    )
    assert result["effect_values_used_for_completion_or_selection"] is False
    assert result["qrels_files_opened_by_this_audit"] is False
    assert result["registry_manifest"]["manifest_id"] == (
        "motivation_transformer_supplemental_evidence_registry_manifest_v1"
    )
    assert all(
        row["command"]
        for row in result["entries"]
        if row["status"] == "completed"
    )


def test_only_functional_design_synthesis_can_change_design_ranking():
    registry = _registry()
    entry = next(
        row
        for row in registry["entries"]
        if row["evidence_id"] == "d3_qk_stage_geometry_v3"
    )
    entry["may_change_design_ranking"] = True
    with pytest.raises(ValueError, match="design-ranking authority drift"):
        _validate_registry_schema(registry)


def test_frozen_supplement_hash_drift_is_fail_closed(tmp_path):
    registry = _registry()
    registry["entries"][0]["frozen_sha256"] = "0" * 64
    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(
        yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
    )
    result = audit_supplemental_evidence_registry(
        ROOT, registry_path=registry_path, registry_manifest_path=None
    )
    row = next(
        value
        for value in result["entries"]
        if value["evidence_id"] == registry["entries"][0]["evidence_id"]
    )
    assert result["status"] == "failed"
    assert row["status"] == "failed_frozen_hash_drift"


def test_design_synthesis_payload_must_keep_claim_boundary():
    entry = next(
        row
        for row in _registry()["entries"]
        if row["evidence_id"] == "component_functional_design_gate_synthesis"
    )
    payload = {
        "analysis_type": entry["expected_analysis_type"],
        "status": "completed",
        "command": ["scripts/synthesize_deep_dive_component_design.py"],
        "qrels_read_by_this_synthesis": False,
        "source_test_opened": False,
        "claim_boundary": {
            "exact_layer_index_is_architecture_evidence": False,
            "operator_necessity_authorized": False,
            "single_model_support_may_change_global_architecture_ranking": False,
        },
    }
    _audit_payload_contract(payload, entry)
    drifted = deepcopy(payload)
    drifted["claim_boundary"]["operator_necessity_authorized"] = True
    with pytest.raises(ValueError, match="claim boundary differs"):
        _audit_payload_contract(drifted, entry)

    missing_command = deepcopy(payload)
    missing_command.pop("command")
    with pytest.raises(ValueError, match="lacks a reproducible command"):
        _audit_payload_contract(missing_command, entry)


def _registry() -> dict:
    return yaml.safe_load((ROOT / REGISTRY_PATH).read_text(encoding="utf-8"))
