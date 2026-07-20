import json
from types import SimpleNamespace

import pytest

from myrec.mechanism.component_necessity_runtime import (
    EXTENSION_MANIFEST_SHA256,
    _audit_parent_selected_branch,
    _load_extension_manifest,
    _stable_smoke_records,
    component_necessity_implementation_identity,
)
from myrec.utils.hashing import sha256_file


def _write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_component_necessity_manifest_and_implementation_are_frozen():
    manifest = _load_extension_manifest(
        "experiments/motivation/transformer_component_necessity_extension_manifest_v2.yaml"
    )
    assert manifest["_sha256"] == EXTENSION_MANIFEST_SHA256
    assert manifest["registered_design"]["family_policy"] == {
        "target_margin_units": 16,
        "ndcg_at_10_units": 16,
        "multiple_testing": "benjamini_hochberg_within_endpoint",
        "alpha": 0.05,
        "missing_or_gate_stopped_cell_p": 1.0,
    }
    identity = component_necessity_implementation_identity()
    assert len(identity["files"]) == 13
    assert len(identity["digest"]) == 64


def test_component_necessity_smoke_selection_is_order_invariant():
    records = [SimpleNamespace(request_id=f"r{index}") for index in range(20)]
    first = _stable_smoke_records(records, 7)
    second = _stable_smoke_records(list(reversed(records)), 7)
    assert [row.request_id for row in first] == [row.request_id for row in second]


def test_parent_selected_branch_must_be_completed_and_sha_bound(tmp_path):
    branch_contract = tmp_path / "contract.json"
    _write_json(branch_contract, {"frozen": True})
    parent = tmp_path / "parent"
    parent.mkdir()
    scores = parent / "scores.jsonl"
    scores.write_text("{}\n", encoding="utf-8")
    metadata = {
        "analysis_stage": "transformer_deep_dive_d2_selected_branch",
        "status": "completed",
        "result_eligible": True,
        "complete_finite_score_coverage": True,
        "identity_passed": True,
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "selected_block": 20,
        "branch_contract": {"sha256": sha256_file(branch_contract)},
        "qrels_read": False,
        "source_test_opened": False,
        "scores_sha256": sha256_file(scores),
    }
    _write_json(parent / "metadata.json", metadata)
    identity = _audit_parent_selected_branch(
        parent,
        method_id="q2_recranker_generalqwen",
        checkpoint_id="checkpoint",
        selected_block=20,
        branch_contract_path=branch_contract,
    )
    assert identity["scores_sha256"] == sha256_file(scores)

    metadata["result_eligible"] = False
    _write_json(parent / "metadata.json", metadata)
    with pytest.raises(ValueError, match="incomplete or unbound"):
        _audit_parent_selected_branch(
            parent,
            method_id="q2_recranker_generalqwen",
            checkpoint_id="checkpoint",
            selected_block=20,
            branch_contract_path=branch_contract,
        )
