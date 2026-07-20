from __future__ import annotations

import json
from pathlib import Path

import pytest

from myrec.mechanism.component_design_synthesis import (
    synthesize_component_design_gates,
)
from myrec.mechanism.component_necessity_evaluator import (
    DONOR_MODES,
    ENDPOINTS,
    METHODS,
)
from myrec.mechanism.component_necessity_runtime import (
    EXTENSION_MANIFEST_PATH,
)
from myrec.mechanism.component_necessity_scoring import NECESSITY_NODES
from myrec.mechanism.selected_branch_evaluator import (
    CONTRAST_GROUPS,
    SELECTED_BRANCH_FOLD_SCOPE,
    selected_branch_contrast_specs,
)
from myrec.utils.hashing import sha256_file


def test_component_design_gate_requires_same_specific_and_neutral_removal(tmp_path):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="attention_o_projection"
    )
    result = synthesize_component_design_gates(
        necessity_path, selected_path, tmp_path / "out", "run"
    )
    assert result["cross_model_functional_support"][
        "component_state_supported_nodes"
    ] == [
        "attention_o_projection"
    ]
    assert result["cross_model_functional_support"][
        "design_prioritized_nodes"
    ] == [
        "attention_o_projection"
    ]
    assert result["cross_model_functional_support"][
        "component_path_design_ranking_eligible"
    ]
    assert all(
        summary["interpretations"]
        == ["attention_output_state_is_a_necessary_mediator"]
        for summary in result["model_summaries"].values()
    )
    assert result["claim_boundary"]["exact_layer_index_is_architecture_evidence"] is False
    assert result["claim_boundary"]["operator_necessity_authorized"] is False
    assert result["claim_boundary"]["registered_behavior"] == (
        "harmful_full_history_target_margin_response"
    )
    assert result["claim_boundary"][
        "positive_neutral_removal_means_harm_reduction"
    ] is True
    assert result["claim_boundary"][
        "component_is_beneficial_for_transfer_authorized"
    ] is False
    assert result["claim_boundary"][
        "strengthen_or_preserve_component_authorized"
    ] is False
    assert all(
        value["shared_parent_bytes_verified"]
        for value in result["shared_parent_lineage"].values()
    )
    report = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert result["report"]["sha256"] == sha256_file(tmp_path / "out" / "report.md")
    assert "attention_branch_state_mediator" in report
    assert "Absolute layer indices" in report
    assert "selected_block" not in report
    assert "layer 20" not in report
    assert "registered harmful full-history response" in report
    assert "should be strengthened" in report
    table_lines = [line for line in report.splitlines() if line.startswith("|")]
    assert len(table_lines) == 2 + len(METHODS) * len(NECESSITY_NODES)
    assert {line.count("|") for line in table_lines} == {10}


def test_component_design_gate_fails_if_parent_bundle_bytes_do_not_match(tmp_path):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="mlp_down_projection"
    )
    necessity = _read_json(necessity_path)
    method_id = METHODS[0]
    root = Path(necessity["input_identities"][method_id]["path"])
    metadata_path = root / "metadata.json"
    metadata = _read_json(metadata_path)
    metadata["parent_selected_branch"] = {
        **metadata["parent_selected_branch"],
        "scores_sha256": "0" * 64,
    }
    _write_json(metadata_path, metadata)
    necessity["input_identities"][method_id]["metadata_sha256"] = sha256_file(
        metadata_path
    )
    pre_qrels_path = Path(necessity["pre_qrels_audit_path"])
    pre_qrels = _read_json(pre_qrels_path)
    pre_qrels["inputs"] = necessity["input_identities"]
    _write_json(pre_qrels_path, pre_qrels)
    necessity["pre_qrels_audit_sha256"] = sha256_file(pre_qrels_path)
    _write_json(necessity_path, necessity)
    with pytest.raises(ValueError, match="do not share parent bytes"):
        synthesize_component_design_gates(
            necessity_path, selected_path, tmp_path / "out", "run"
        )


def test_component_design_gate_does_not_accept_null_sensitivity_alone(tmp_path):
    necessity_path, selected_path = _write_inputs(tmp_path, supported_node=None)
    necessity = _read_json(necessity_path)
    for method_id in METHODS:
        result = necessity["results"][method_id]["attention_o_projection"]["null"][
            "target_margin"
        ]
        result.update(
            {
                "mean": 0.2,
                "ci95": [0.1, 0.3],
                "two_sided_p": 0.001,
                "bh_q": 0.008,
                "positive_removal_gate_passed": True,
            }
        )
        for row in necessity["family_rows"]:
            if (
                row["method_id"] == method_id
                and row["node"] == "attention_o_projection"
                and row["donor_mode"] == "null"
                and row["endpoint"] == "target_margin"
            ):
                row.update(
                    {
                        "mean": 0.2,
                        "ci95": [0.1, 0.3],
                        "two_sided_p": 0.001,
                        "bh_q": 0.008,
                        "positive_removal_gate_passed": True,
                    }
                )
    _write_json(necessity_path, necessity)
    result = synthesize_component_design_gates(
        necessity_path, selected_path, tmp_path / "out", "run"
    )
    assert result["cross_model_functional_support"][
        "component_state_supported_nodes"
    ] == []
    assert result["cross_model_functional_support"]["design_prioritized_nodes"] == []
    assert all(
        summary["interpretations"]
        == ["no_registered_component_state_gate_passed"]
        for summary in result["model_summaries"].values()
    )


def test_block_output_state_ceiling_cannot_become_residual_design_target(tmp_path):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="block_output_residual"
    )
    result = synthesize_component_design_gates(
        necessity_path, selected_path, tmp_path / "out", "run"
    )
    assert result["cross_model_functional_support"][
        "component_state_supported_nodes"
    ] == ["block_output_residual"]
    assert result["cross_model_functional_support"]["design_prioritized_nodes"] == []
    assert not result["cross_model_functional_support"][
        "component_path_design_ranking_eligible"
    ]
    assert all(
        summary["interpretations"]
        == ["residual_or_nonlinear_interaction_remains_unresolved"]
        for summary in result["model_summaries"].values()
    )
    assert result["claim_boundary"][
        "block_output_state_ceiling_authorizes_residual_operator_claim"
    ] is False


def test_component_state_support_is_not_design_priority_without_structural_controls(
    tmp_path,
):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="attention_o_projection"
    )
    selected = _read_json(selected_path)
    for row in selected["rows"]:
        if row["contrast_id"] == "random__attention_o_projection":
            row["registered_support"] = False
    _write_json(selected_path, selected)
    result = synthesize_component_design_gates(
        necessity_path, selected_path, tmp_path / "out", "run"
    )
    assert result["cross_model_functional_support"][
        "component_state_supported_nodes"
    ] == ["attention_o_projection"]
    assert result["cross_model_functional_support"]["design_prioritized_nodes"] == []
    assert not result["cross_model_functional_support"][
        "component_path_design_ranking_eligible"
    ]


def test_component_design_gate_rejects_forged_selected_support_flag(tmp_path):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="attention_o_projection"
    )
    selected = _read_json(selected_path)
    row = next(
        row
        for row in selected["rows"]
        if row["method_id"] == METHODS[0]
        and row["contrast_id"] == "same__attention_o_projection"
        and row["endpoint"] == "target_margin"
    )
    row["mean"] = 0.2
    _write_json(selected_path, selected)
    with pytest.raises(ValueError, match="not sign/BH/fold derived"):
        synthesize_component_design_gates(
            necessity_path, selected_path, tmp_path / "out", "run"
        )


def test_component_design_gate_rejects_forged_necessity_flag(tmp_path):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="attention_o_projection"
    )
    necessity = _read_json(necessity_path)
    result = necessity["results"][METHODS[0]]["attention_o_projection"][
        "neutral"
    ]["target_margin"]
    result["mean"] = -0.2
    _write_json(necessity_path, necessity)
    with pytest.raises(ValueError, match="nested/family inference differs"):
        synthesize_component_design_gates(
            necessity_path, selected_path, tmp_path / "out", "run"
        )


def test_component_design_gate_rejects_forged_necessity_bh_value(tmp_path):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="attention_o_projection"
    )
    necessity = _read_json(necessity_path)
    result = necessity["results"][METHODS[0]]["attention_o_projection"][
        "neutral"
    ]["target_margin"]
    result["bh_q"] = 0.02
    for row in necessity["family_rows"]:
        if (
            row["method_id"] == METHODS[0]
            and row["node"] == "attention_o_projection"
            and row["donor_mode"] == "neutral"
            and row["endpoint"] == "target_margin"
        ):
            row["bh_q"] = 0.02
    _write_json(necessity_path, necessity)
    with pytest.raises(ValueError, match="BH values differ"):
        synthesize_component_design_gates(
            necessity_path, selected_path, tmp_path / "out", "run"
        )


def test_component_design_gate_rejects_tampered_per_request_bytes(tmp_path):
    necessity_path, selected_path = _write_inputs(
        tmp_path, supported_node="attention_o_projection"
    )
    necessity = _read_json(necessity_path)
    Path(necessity["per_request_contrasts_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="per-request contrasts bytes differ"):
        synthesize_component_design_gates(
            necessity_path, selected_path, tmp_path / "out", "run"
        )


def _write_inputs(
    root: Path, *, supported_node: str | None
) -> tuple[Path, Path]:
    selected_input_metrics = {}
    necessity_input_identities = {}
    for method_id in METHODS:
        parent = root / f"parent_{method_id}"
        parent.mkdir()
        _write_json(parent / "metadata.json", {"method_id": method_id})
        (parent / "scores.jsonl").write_text("{}\n", encoding="utf-8")
        parent_identity = {
            "path": str(parent),
            "metadata_sha256": sha256_file(parent / "metadata.json"),
            "scores_sha256": sha256_file(parent / "scores.jsonl"),
        }

        selected_metrics_path = root / f"selected_{method_id}.json"
        _write_json(
            selected_metrics_path,
            {
                "analysis_type": "transformer_deep_dive_d2_selected_branch",
                "status": "completed",
                "method_id": method_id,
                "selected_block": 20,
                "implementation_digest": "selected-implementation",
                "input_bundle": parent_identity,
            },
        )
        selected_input_metrics[method_id] = {
            "path": str(selected_metrics_path),
            "sha256": sha256_file(selected_metrics_path),
            "implementation_digest": "selected-implementation",
        }

        necessity_root = root / f"necessity_{method_id}"
        necessity_root.mkdir()
        _write_json(
            necessity_root / "metadata.json",
            {
                "analysis_stage": "transformer_component_necessity_extension",
                "status": "completed",
                "method_id": method_id,
                "selected_block": 20,
                "parent_selected_branch": parent_identity,
            },
        )
        (necessity_root / "scores.jsonl").write_text("{}\n", encoding="utf-8")
        necessity_input_identities[method_id] = {
            "status": "completed_bundle",
            "path": str(necessity_root),
            "metadata_sha256": sha256_file(necessity_root / "metadata.json"),
            "scores_sha256": sha256_file(necessity_root / "scores.jsonl"),
        }

    specs = selected_branch_contrast_specs()
    selected_rows = []
    for method_id in METHODS:
        for contrast_id, spec in specs.items():
            for endpoint in ENDPOINTS:
                support = bool(
                    endpoint == "target_margin"
                    and supported_node is not None
                    and contrast_id
                    in {
                        f"same__{supported_node}",
                        f"same_minus_cross__{supported_node}",
                        f"same_minus_wrong__{supported_node}",
                        f"norm__{supported_node}",
                        f"direction__{supported_node}",
                        f"random__{supported_node}",
                    }
                )
                selected_rows.append(
                    {
                        "method_id": method_id,
                        "contrast_id": contrast_id,
                        "group": spec["group"],
                        "endpoint": endpoint,
                        "missing": False,
                        "registered_support": support,
                        "mean": -0.2 if support else 0.0,
                        "ci95": [-0.3, -0.1] if support else [-0.1, 0.1],
                        "two_sided_p": 0.001 if support else 1.0,
                        "bh_q": 0.01 if support else 1.0,
                        "bh_significant": support,
                        "expected_sign": (
                            "negative"
                            if spec["group"]
                            in {
                                "same",
                                "same_minus_cross",
                                "same_minus_wrong_history",
                                "direction_scale",
                            }
                            else None
                        ),
                        "expected_sign_met": (
                            support
                            if spec["group"]
                            in {
                                "same",
                                "same_minus_cross",
                                "same_minus_wrong_history",
                                "direction_scale",
                            }
                            else None
                        ),
                        "evidence_role": (
                            "registered_confirmatory_branch_localization"
                            if support
                            else "registered_confirmatory_branch_localization"
                        ),
                    }
                )
    selected_synthesis_path = root / "selected_synthesis.json"
    _write_json(
        selected_synthesis_path,
        {
            "analysis_type": "transformer_deep_dive_d2_selected_branch_synthesis",
            "status": "completed",
            "models": list(METHODS),
            "fold_scope": dict(SELECTED_BRANCH_FOLD_SCOPE),
            "rows": selected_rows,
            "families": {
                f"{group}__{endpoint}": {"planned_family_size": 2 * units}
                for group, units in CONTRAST_GROUPS.items()
                for endpoint in ENDPOINTS
            },
            "input_metrics": selected_input_metrics,
        },
    )

    results = {method_id: {} for method_id in METHODS}
    family_rows = []
    for method_id in METHODS:
        for node in NECESSITY_NODES:
            results[method_id][node] = {}
            for donor_mode in DONOR_MODES:
                results[method_id][node][donor_mode] = {}
                for endpoint in ENDPOINTS:
                    primary = bool(
                        supported_node == node
                        and donor_mode == "neutral"
                        and endpoint == "target_margin"
                    )
                    value = {
                        "status": "completed",
                        "mean": 0.2 if primary else 0.0,
                        "ci95": [0.1, 0.3] if primary else [-0.1, 0.1],
                        "two_sided_p": 0.001 if primary else 1.0,
                        "bh_q": 0.008 if primary else 1.0,
                        "positive_removal_gate_passed": primary,
                        "primary_position_preserving_gate_passed": primary,
                        "ndcg_practically_equivalent": False,
                    }
                    results[method_id][node][donor_mode][endpoint] = value
                    family_rows.append(
                        {
                            "method_id": method_id,
                            "node": node,
                            "donor_mode": donor_mode,
                            "endpoint": endpoint,
                            "status": value["status"],
                            **{
                                name: value[name]
                                for name in (
                                    "positive_removal_gate_passed",
                                    "primary_position_preserving_gate_passed",
                                    "ndcg_practically_equivalent",
                                )
                            },
                            "mean": value["mean"],
                            "ci95": value["ci95"],
                            "two_sided_p": value["two_sided_p"],
                            "bh_q": value["bh_q"],
                        }
                    )
    extension_manifest_sha = sha256_file(EXTENSION_MANIFEST_PATH)
    pre_qrels_path = root / "necessity_pre_qrels_audit.json"
    _write_json(
        pre_qrels_path,
        {
            "analysis_type": "component_necessity_pre_qrels_integrity",
            "status": "passed",
            "qrels_read": False,
            "checks": {
                "each_model_has_exactly_one_completed_bundle_or_gate_stop": True,
                "completed_bundles_have_fold1_complete_finite_coverage": True,
                "all_four_full_to_full_identities_at_most_1e-5": True,
                "frozen_baseline_recompute_within_path_local_bf16_bound": True,
                "position_preserving_content_neutral_rows_and_path_audits_bound": True,
                "parent_selected_branch_and_contract_sha_bound": True,
                "candidate_and_request_manifests_reconstructed": True,
                "extension_plan_and_manifest_hashes_bound": True,
            },
            "extension_manifest_sha256": extension_manifest_sha,
            "inputs": necessity_input_identities,
        },
    )
    per_request_path = root / "necessity_per_request_contrasts.npz"
    per_request_path.write_bytes(b"fixed-necessity-test-npz")
    necessity_path = root / "necessity_metrics.json"
    _write_json(
        necessity_path,
        {
            "analysis_type": "transformer_component_necessity_extension",
            "status": "completed",
            "methods": list(METHODS),
            "nodes": list(NECESSITY_NODES),
            "donor_modes": list(DONOR_MODES),
            "endpoints": list(ENDPOINTS),
            "normalized_query_fold": 1,
            "strict_transfer_requests": 100,
            "bootstrap": {
                "cluster": "normalized_query",
                "samples": 5000,
                "seed": 20260715,
            },
            "extension_manifest_sha256": extension_manifest_sha,
            "qrels_read": True,
            "qrels_fold_opened": 1,
            "other_fold_qrels_opened": False,
            "source_test_opened": False,
            "family_policy": {
                "separate_by_endpoint": True,
                "units_per_endpoint": 16,
                "method": "benjamini_hochberg",
                "missing_or_gate_stopped_p": 1.0,
            },
            "claim_boundary": {
                "primary_support_requires_position_preserving_neutral_removal": True,
                "null_removal_alone_authorizes_support": False,
                "design_ranking_requires_parent_sufficiency_and_specificity": True,
                "operator_necessity_authorized": False,
                "exclusive_origin_authorized": False,
                "cross_dataset_or_model_scale_generalization_authorized": False,
            },
            "family_rows": family_rows,
            "results": results,
            "input_identities": necessity_input_identities,
            "pre_qrels_audit_path": str(pre_qrels_path),
            "pre_qrels_audit_sha256": sha256_file(pre_qrels_path),
            "qrels_fold_sha256": "a" * 64,
            "qrels_split_manifest_sha256": "b" * 64,
            "qrels_source_sha256": "c" * 64,
            "per_request_contrasts_path": str(per_request_path),
            "per_request_contrasts_sha256": sha256_file(per_request_path),
        },
    )
    return necessity_path, selected_synthesis_path


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
