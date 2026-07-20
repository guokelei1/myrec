import json

import pytest

from myrec.mechanism.component_necessity_evaluator import (
    DONOR_MODES,
    ENDPOINTS,
    FAMILY_SIZE_PER_ENDPOINT,
    METHODS,
    _apply_bh_and_gates,
    _audit_gate_contract,
    _baseline_recompute_deltas,
    _family_row,
    _missing_inference,
)
from myrec.mechanism.component_necessity_scoring import NECESSITY_NODES
from myrec.mechanism.selected_branch_scoring import SELECTED_NODES


def test_component_necessity_family_keeps_all_gate_stopped_cells():
    results = {method: {} for method in METHODS}
    rows = []
    for method in METHODS:
        for node in NECESSITY_NODES:
            results[method][node] = {}
            for donor_mode in DONOR_MODES:
                results[method][node][donor_mode] = {}
                for endpoint in ENDPOINTS:
                    result = _missing_inference(
                        donor_mode=donor_mode, reason="registered_gate_stop"
                    )
                    results[method][node][donor_mode][endpoint] = result
                    rows.append(
                        _family_row(method, node, donor_mode, endpoint, result)
                    )
    _apply_bh_and_gates(rows, results)
    assert len(rows) == 2 * FAMILY_SIZE_PER_ENDPOINT == 32
    assert all(row["two_sided_p"] == 1.0 for row in rows)
    assert all(row["bh_q"] == 1.0 for row in rows)
    assert all(
        not results[method][node][donor_mode][endpoint][
            "positive_removal_gate_passed"
        ]
        for method in METHODS
        for node in NECESSITY_NODES
        for donor_mode in DONOR_MODES
        for endpoint in ENDPOINTS
    )


def test_component_necessity_positive_and_equivalence_gates_are_distinct():
    results = {method: {} for method in METHODS}
    rows = []
    for method in METHODS:
        for node in NECESSITY_NODES:
            results[method][node] = {}
            for donor_mode in DONOR_MODES:
                results[method][node][donor_mode] = {}
                for endpoint in ENDPOINTS:
                    if endpoint == "target_margin":
                        result = {
                            "status": "completed",
                            "mean": 0.2,
                            "ci95": [0.1, 0.3],
                            "two_sided_p": 0.001,
                        }
                    else:
                        result = {
                            "status": "completed",
                            "mean": 0.001,
                            "ci95": [-0.002, 0.004],
                            "two_sided_p": 0.5,
                        }
                    results[method][node][donor_mode][endpoint] = result
                    rows.append(
                        _family_row(method, node, donor_mode, endpoint, result)
                    )
    _apply_bh_and_gates(rows, results)
    for method in METHODS:
        for node in NECESSITY_NODES:
            for donor_mode in DONOR_MODES:
                assert results[method][node][donor_mode]["target_margin"][
                    "positive_removal_gate_passed"
                ]
                assert (
                    results[method][node][donor_mode]["target_margin"][
                        "primary_position_preserving_gate_passed"
                    ]
                    is (donor_mode == "neutral")
                )
                assert not results[method][node][donor_mode]["target_margin"][
                    "ndcg_practically_equivalent"
                ]
                assert not results[method][node][donor_mode]["ndcg@10"][
                    "positive_removal_gate_passed"
                ]
                assert results[method][node][donor_mode]["ndcg@10"][
                    "ndcg_practically_equivalent"
                ]


def test_confirmed_gate_contract_cannot_replace_required_bundle(tmp_path):
    path = tmp_path / "contract.json"
    base = {
        "contract_type": "transformer_deep_dive_d2_selected_branch_contract",
        "status": "completed",
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "checkpoint",
        "selected_nodes": list(SELECTED_NODES),
        "qrels_values_exposed_to_scorer": False,
        "source_test_opened": False,
        "branch_scoring_eligible": False,
        "fold1_negative_transition_reproduced": False,
        "evidence_role": "stopped_no_negative_fold0_transition",
    }
    path.write_text(json.dumps(base), encoding="utf-8")
    observed = _audit_gate_contract(
        path,
        method_id="q2_recranker_generalqwen",
        expected_checkpoint_id="checkpoint",
    )
    assert observed["branch_scoring_eligible"] is False

    base.update(
        {
            "branch_scoring_eligible": True,
            "fold1_negative_transition_reproduced": True,
            "evidence_role": "registered_confirmatory_branch_localization",
        }
    )
    path.write_text(json.dumps(base), encoding="utf-8")
    with pytest.raises(ValueError, match="requires a bundle"):
        _audit_gate_contract(
            path,
            method_id="q2_recranker_generalqwen",
            expected_checkpoint_id="checkpoint",
        )


def test_component_necessity_rederives_frozen_baseline_bf16_ratio():
    row = {
        "request_id": "r",
        "candidate_item_id": "a",
        "conditions": {
            "baseline_full": 0.125,
            "baseline_null": -0.1875,
        },
    }
    full_delta, null_delta, maximum_ratio = _baseline_recompute_deltas(
        row,
        {("r", "a"): 0.1},
        {("r", "a"): -0.2},
    )
    assert full_delta == pytest.approx(0.025)
    assert null_delta == pytest.approx(0.0125)
    assert maximum_ratio == pytest.approx(0.4)

    with pytest.raises(ValueError, match="absent from a frozen baseline"):
        _baseline_recompute_deltas(
            row,
            {},
            {("r", "a"): -0.2},
        )
