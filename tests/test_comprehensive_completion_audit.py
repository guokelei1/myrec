from __future__ import annotations

import json
from pathlib import Path

import pytest

from myrec.mechanism import comprehensive_completion_audit as audit_module
from myrec.mechanism.comprehensive_completion_audit import (
    _audit_final_files,
    _validate_file_identity,
    audit_comprehensive_completion,
)
from myrec.mechanism.comprehensive_report_builder import (
    ANALYSIS_TYPE,
    REPORT_SECTION_CONTRACT,
)
from myrec.mechanism.deep_dive_evidence_topology import MODEL_IDS
from myrec.mechanism.deep_dive_report_builder import REPORT_ANALYSIS_TYPE
from myrec.mechanism.deep_dive_report_contract import COMPONENT_IDS
from myrec.utils.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def test_real_completion_audit_is_exhaustive_and_fail_closed_while_pending() -> None:
    result = audit_comprehensive_completion(ROOT)
    assert result["status"] in {"pending", "completed"}
    assert result["requirements_registered"] == 12
    assert len(result["requirements"]) == 12
    assert len({row["requirement_id"] for row in result["requirements"]}) == 12
    assert {
        "formal_19_closeout",
        "supplement_21_closeout",
        "d2_fixed_and_conditional_terminal",
        "component_18_coverage",
        "exact_interface_and_evidence_disposition",
        "producer_topology",
        "source_test_closed",
        "no_readiness_audit_failures",
        "final_human_interpretation_worksheet",
        "formal_deep_dive_report",
        "comprehensive_json_and_13_section_contract",
        "comprehensive_markdown_report",
    } == {row["requirement_id"] for row in result["requirements"]}
    assert result["scientific_effect_values_recomputed"] is False
    assert result["scientific_effect_values_used_for_completion"] is False
    assert result["qrels_files_opened"] is False
    assert result["source_test_opened"] is False
    interface_requirement = next(
        row
        for row in result["requirements"]
        if row["requirement_id"] == "exact_interface_and_evidence_disposition"
    )
    assert interface_requirement["status"] == "completed"
    assert interface_requirement["evidence"]["operator_causal_debts"] == 38
    assert interface_requirement["evidence"][
        "operator_causal_debt_class_counts"
    ] == {"inference_operator": 18, "training_mechanism": 20}
    assert interface_requirement["evidence"][
        "new_experiment_family_authorized"
    ] is False
    assert interface_requirement["evidence"][
        "operator_causal_debt_is_lower_bound"
    ] is True
    assert interface_requirement["evidence"][
        "operator_attribution_inferred_for_other_interfaces"
    ] is False
    system_layers = interface_requirement["evidence"]["system_layer_coverage"]
    assert sum(row["interface_count"] for row in system_layers.values()) == 63
    assert sum(
        row["operator_causal_debt_count"] for row in system_layers.values()
    ) == 38
    assert interface_requirement["evidence"][
        "registered_claim_ceiling_counts"
    ] == {"none": 3, "M": 0, "D": 35, "S": 21, "N": 1, "G": 3}
    assert sum(
        interface_requirement["evidence"][
            "completed_artifact_claim_ceiling_counts"
        ].values()
    ) == 63
    assert interface_requirement["evidence"][
        "claim_ceilings_are_artifact_availability_only"
    ] is True
    assert interface_requirement["evidence"][
        "actual_scientific_evidence_levels_inferred"
    ] is False
    assert interface_requirement["evidence"][
        "frozen_model_architecture_audit_status"
    ] == "completed"
    assert interface_requirement["evidence"]["config_backed_interfaces"] == 53
    assert interface_requirement["evidence"][
        "config_backed_interfaces_present"
    ] is True
    assert interface_requirement["evidence"][
        "dynamic_runtime_or_source_backed_interfaces"
    ] == 10
    assert interface_requirement["evidence"][
        "implementation_provenance_covered_interfaces"
    ] == 63
    assert interface_requirement["evidence"][
        "all_exact_interfaces_have_config_or_runtime_source_provenance"
    ] is True
    assert interface_requirement["evidence"]["forward_primitives"] == 40
    assert interface_requirement["evidence"]["forward_inference_interfaces"] == 43
    assert interface_requirement["evidence"]["forward_mapped_interfaces"] == 43
    assert interface_requirement["evidence"]["forward_missing_interfaces"] == []
    assert interface_requirement["evidence"]["forward_source_bindings"] == 11
    assert interface_requirement["evidence"]["forward_peft_version"] == "0.19.1"
    assert interface_requirement["evidence"][
        "forward_source_environment_is_frozen_checkpoint_environment"
    ] is True
    assert interface_requirement["evidence"]["inactive_architecture_paths"] == 9
    assert interface_requirement["evidence"][
        "forward_primitive_interface_coverage_complete"
    ] is True
    assert interface_requirement["evidence"][
        "all_inactive_architecture_paths_verified"
    ] is True
    assert interface_requirement["evidence"][
        "operator_attribution_inferred_from_forward_coverage"
    ] is False
    assert interface_requirement["evidence"]["training_primitives"] == 27
    assert interface_requirement["evidence"]["training_exact_interfaces"] == 20
    assert interface_requirement["evidence"]["training_mapped_interfaces"] == 20
    assert interface_requirement["evidence"]["training_missing_interfaces"] == []
    assert interface_requirement["evidence"]["training_source_bindings"] == 12
    assert interface_requirement["evidence"]["training_artifact_bindings"] == 3
    assert interface_requirement["evidence"][
        "training_primitive_interface_coverage_complete"
    ] is True
    assert interface_requirement["evidence"]["training_peft_version"] == "0.19.1"
    assert interface_requirement["evidence"][
        "training_coverage_is_multiseed_causal_attribution"
    ] is False
    assert interface_requirement["evidence"][
        "operator_attribution_inferred_from_training_coverage"
    ] is False
    assert interface_requirement["evidence"]["inactive_training_paths"] == 7
    assert interface_requirement["evidence"][
        "all_inactive_training_paths_verified"
    ] is True
    assert interface_requirement["evidence"]["runtime_identity_smokes"] == 4
    assert interface_requirement["evidence"]["runtime_hook_nodes"] == 18
    assert interface_requirement["evidence"][
        "runtime_identity_and_recomposition_validated"
    ] is True
    assert interface_requirement["evidence"]["superseded_runtime_lineage"] == 1
    assert interface_requirement["evidence"][
        "superseded_runtime_lineage_retained"
    ] is True
    assert interface_requirement["evidence"]["frozen_base_artifacts"] == 2
    assert interface_requirement["evidence"]["frozen_model_pathways"] == 4
    assert set(interface_requirement["evidence"]["model_pathways"]) == {
        "q0_qwen3_reranker_06b",
        "q1_instructrec_generalqwen",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    }
    if result["status"] == "completed":
        assert result["requirements_completed"] == 12
        assert result["pending_requirement_ids"] == []
    else:
        assert result["requirements_completed"] < 12
        assert result["pending_requirement_ids"]


def test_terminal_file_identity_rejects_byte_drift_and_root_escape(
    tmp_path: Path,
) -> None:
    report = tmp_path / "reports" / "evidence.json"
    report.parent.mkdir(parents=True)
    report.write_text('{"status":"completed"}\n', encoding="utf-8")
    identity = {
        "path": "reports/evidence.json",
        "sha256": sha256_file(report),
    }
    assert _validate_file_identity(
        tmp_path, identity, label="test evidence"
    ) == report.resolve()

    report.write_text('{"status":"altered"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 drift"):
        _validate_file_identity(tmp_path, identity, label="test evidence")

    outside = tmp_path.parent / "outside-evidence.json"
    outside.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes repository root"):
        _validate_file_identity(
            tmp_path,
            {
                "path": "../outside-evidence.json",
                "sha256": sha256_file(outside),
            },
            label="escaped evidence",
        )


def test_plausible_terminal_files_fail_closed_when_semantic_chain_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worksheet = {
        "schema_version": 1,
        "worksheet_status": "final",
    }
    formal = {
        "schema_version": 1,
        "analysis_type": REPORT_ANALYSIS_TYPE,
        "status": "completed",
    }
    comprehensive = {
        "schema_version": 1,
        "analysis_type": ANALYSIS_TYPE,
        "status": "completed",
        "report_section_contract": {
            "covered_sections": len(REPORT_SECTION_CONTRACT),
            "registered_sections": len(REPORT_SECTION_CONTRACT),
        },
        "frozen_model_architecture_audit": {
            "status": "completed",
            "failures": [],
        },
        "claim_invariants": {"source_test_opened": False},
        "component_matrix": {component_id: {} for component_id in COMPONENT_IDS},
        "component_model_matrix": {
            component_id: {model_id: {} for model_id in MODEL_IDS}
            for component_id in COMPONENT_IDS
        },
    }
    paths = audit_module.FINAL_PATHS
    for key, payload in (
        ("decision_worksheet", worksheet),
        ("formal_report", formal),
        ("comprehensive_json", comprehensive),
    ):
        path = tmp_path / paths[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    markdown_path = tmp_path / paths["comprehensive_markdown"]
    markdown_path.write_text("# Plausible but stale report\n", encoding="utf-8")

    monkeypatch.setattr(audit_module, "_audit_formal_report", lambda payload: None)

    def reject_chain(*args, **kwargs):
        raise ValueError("forged semantic chain")

    monkeypatch.setattr(audit_module, "_audit_terminal_report_chain", reject_chain)
    requirements = _audit_final_files(tmp_path)
    assert {row["status"] for row in requirements} == {"pending"}
    assert all(
        "forged semantic chain"
        in " ".join(row["evidence"].get("terminal_chain_validation_errors", []))
        for row in requirements
    )
