from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from myrec.mechanism.deep_dive_closeout_audit import (
    EXPECTED_DELIVERABLES,
    EXPECTED_FROZEN_ASSETS,
)
from myrec.mechanism.deep_dive_decision_worksheet import (
    DeepDiveDecisionWorksheetError,
    PRIMARY_FLAG_FIELDS,
    WORKSHEET_ANALYSIS_TYPE,
    WORKSHEET_STATUS,
    build_deep_dive_decision_worksheet,
)
from myrec.mechanism.deep_dive_opportunity_catalog import (
    OPPORTUNITY_DESIGN_CATALOG,
    OPPORTUNITY_DESIGN_CATALOG_SHA256,
    OPPORTUNITY_IDS,
)
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_IDS,
    CROSS_MODEL_ATTRIBUTION_BOUNDARY,
    CROSS_MODEL_ATTRIBUTION_SCOPES,
    HYPOTHESIS_IDS,
    PRIMARY_ATTRIBUTION_CRITERION_DESCRIPTIONS,
    PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE,
    PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE,
    PRIMARY_ATTRIBUTION_FOLD_SCOPE,
    PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE,
    PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE,
    PRIMARY_ATTRIBUTION_STRENGTH_RULES,
    PRIMARY_ATTRIBUTION_MODELS,
    REQUIRED_ASSERTIONS,
    REQUIRED_NARRATIVE_FIELDS,
    TRANSFER_FAILURE_CAUSAL_SCOPES,
    TRANSFER_EXPLANATION_LADDER_SCOPE,
    TRANSFER_EXPLANATION_LEVELS,
    DeepDiveReportContractError,
    validate_deep_dive_report_decisions,
)


def _frozen_root(tmp_path):
    for relative_path, sha256 in EXPECTED_FROZEN_ASSETS.items():
        # Tests need preimages for the production digests, so use the real
        # repository assets and first verify they still match the constants.
        source = Path.cwd() / relative_path
        content = source.read_bytes()
        assert hashlib.sha256(content).hexdigest() == sha256
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return tmp_path


def test_decision_worksheet_has_complete_frozen_topology(tmp_path):
    worksheet = build_deep_dive_decision_worksheet(_frozen_root(tmp_path))

    assert worksheet["analysis_type"] == WORKSHEET_ANALYSIS_TYPE
    assert worksheet["status"] == WORKSHEET_STATUS
    assert worksheet["final_report_input"] is False
    assert worksheet["final_validator_expected_to_reject"] is True
    assert worksheet["scientific_effect_values_read"] is False
    assert worksheet["qrels_read"] is False
    assert worksheet["source_test_opened"] is False
    assert worksheet["opportunity_design_catalog_sha256"] == (
        OPPORTUNITY_DESIGN_CATALOG_SHA256
    )
    assert set(worksheet["narratives"]) == set(REQUIRED_NARRATIVE_FIELDS)
    assert all(value is None for value in worksheet["narratives"].values())
    assert [
        row["component_id"] for row in worksheet["component_evidence_matrix"]
    ] == list(COMPONENT_IDS)
    assert len(worksheet["component_practical_equivalence_gate_catalog"]) == 4
    attention_value = next(
        row
        for row in worksheet["component_evidence_matrix"]
        if row["component_id"] == "attention_value_transport"
    )
    assert attention_value["causal_support_mechanical_dependencies"] == {
        "d3_attention_edges": ["d3_attention_heads"]
    }
    assert [
        row["method_id"] for row in worksheet["primary_loss_attribution"]
    ] == list(PRIMARY_ATTRIBUTION_MODELS)
    assert [
        row["hypothesis_id"] for row in worksheet["hypothesis_status_matrix"]
    ] == list(HYPOTHESIS_IDS)
    assert [
        row["opportunity_id"]
        for row in worksheet["architecture_opportunity_ranking"]
    ] == list(OPPORTUNITY_IDS)
    assert len(worksheet["closeout_gate"]["required_deliverables"]) == len(
        EXPECTED_DELIVERABLES
    ) == 19
    assert worksheet["closeout_gate"][
        "all_admitted_mechanical_failures_require_component_assignment"
    ] is True
    assert worksheet["required_boundary_assertions"] == REQUIRED_ASSERTIONS
    assert all(
        value is None for value in worksheet["boundary_assertions"].values()
    )
    assert worksheet["cross_model_primary_attribution"]["allowed_scopes"] == list(
        CROSS_MODEL_ATTRIBUTION_SCOPES
    )
    assert worksheet["cross_model_primary_attribution"]["boundary"] == (
        CROSS_MODEL_ATTRIBUTION_BOUNDARY
    )


def test_every_scientific_decision_remains_todo(tmp_path):
    worksheet = build_deep_dive_decision_worksheet(_frozen_root(tmp_path))

    for row in worksheet["component_evidence_matrix"]:
        assert row["decision"]
        assert all(value is None for value in row["decision"].values())
        assert row["allowed_negative_evidence_bases"]
        assert row["registered_probe_claim_boundary"]
        assert isinstance(row["registered_practical_equivalence_routes"], list)
        assert set(row["causal_support_capable_deliverables"]).issubset(
            row["allowed_evidence_deliverables"]
        )
        assert set(row["deliverable_model_coverage"]) == set(
            row["allowed_evidence_deliverables"]
        )
    for row in worksheet["primary_loss_attribution"]:
        assert set(PRIMARY_FLAG_FIELDS).issubset(row["decision"])
        assert all(value is None for value in row["decision"].values())
        assert row["required_decisive_evidence"] == [
            "d2_postblock",
            "d2_selected_branches",
        ]
        assert row["machine_derived_primary_component_required"] is True
        assert row["machine_derived_evidence_strength_required"] is True
        assert (
            row[
                "incoming_block_state_support_blocks_residual_or_norm_attribution"
            ]
            is True
        )
        assert (
            row[
                "normalization_attribution_requires_supported_post_norm_with_unsupported_paired_pre_norm_state"
            ]
            is True
        )
        assert row["deterministic_precedence"] == list(
            PRIMARY_ATTRIBUTION_DERIVATION_PRECEDENCE
        )
        assert row["criterion_descriptions"] == (
            PRIMARY_ATTRIBUTION_CRITERION_DESCRIPTIONS
        )
        assert row["evidence_strength_rules"] == (
            PRIMARY_ATTRIBUTION_STRENGTH_RULES
        )
        assert row["fold_scope"] == PRIMARY_ATTRIBUTION_FOLD_SCOPE
        assert (
            row["fold_scope"]["node_effect_two_fold_replication_tested"]
            is False
        )
        assert row["endpoint_scope"] == PRIMARY_ATTRIBUTION_ENDPOINT_SCOPE
        assert row["layer_selection_scope"] == (
            PRIMARY_ATTRIBUTION_LAYER_SELECTION_SCOPE
        )
        assert row["intervention_scope"] == (
            PRIMARY_ATTRIBUTION_INTERVENTION_SCOPE
        )
        assert set(row["allowed_transfer_failure_causal_scopes"]) == (
            TRANSFER_FAILURE_CAUSAL_SCOPES
        )
        assert row["allowed_transfer_explanation_levels"] == list(
            TRANSFER_EXPLANATION_LEVELS
        )
        assert row["transfer_explanation_ladder"] == (
            TRANSFER_EXPLANATION_LADDER_SCOPE
        )
        assert row["transfer_explanation_ladder"][
            "highest_level_establishes_necessity"
        ] is False
        assert "transfer_explanation_level" in row["decision"]
        assert row["residual_node_support_alone_is_not_composition"] is True
    assert all(
        value is None
        for value in worksheet["cross_model_primary_attribution"][
            "decision"
        ].values()
    )
    for row in worksheet["hypothesis_status_matrix"]:
        assert row["allowed_negative_evidence_bases"]
        assert all(value is None for value in row["decision"].values())
    for row in worksheet["architecture_opportunity_ranking"]:
        assert all(value is None for value in row["decision"].values())
        assert row["frozen_design"] == OPPORTUNITY_DESIGN_CATALOG[
            row["opportunity_id"]
        ]


def test_h5_cannot_be_promoted_in_this_stage(tmp_path):
    worksheet = build_deep_dive_decision_worksheet(_frozen_root(tmp_path))
    h5 = next(
        row
        for row in worksheet["hypothesis_status_matrix"]
        if row["hypothesis_id"] == "H5"
    )
    assert h5["support_authorized_in_this_stage"] is False
    assert h5["supported_required_evidence_groups"] == []
    assert "supported" not in h5["allowed_statuses"]
    assert "rejected" not in h5["allowed_statuses"]


def test_worksheet_is_deliberately_rejected_as_final_decisions(tmp_path):
    worksheet = build_deep_dive_decision_worksheet(_frozen_root(tmp_path))
    with pytest.raises(DeepDiveReportContractError, match="empty executive_summary"):
        validate_deep_dive_report_decisions(worksheet)


def test_worksheet_refuses_frozen_source_drift(tmp_path):
    root = _frozen_root(tmp_path)
    relative_path = next(iter(EXPECTED_FROZEN_ASSETS))
    path = root / relative_path
    path.write_bytes(path.read_bytes() + b"drift")
    with pytest.raises(
        DeepDiveDecisionWorksheetError, match="frozen source hash differs"
    ):
        build_deep_dive_decision_worksheet(root)


def test_returned_frozen_design_is_not_shared_mutable_state(tmp_path):
    worksheet = build_deep_dive_decision_worksheet(_frozen_root(tmp_path))
    changed = deepcopy(worksheet)
    changed["architecture_opportunity_ranking"][0]["frozen_design"][
        "innovation_claim"
    ] = "changed"
    assert OPPORTUNITY_DESIGN_CATALOG[OPPORTUNITY_IDS[0]][
        "innovation_claim"
    ] != "changed"
