from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.eval.controlled_composition import (
    build_accounting_rows,
    build_endpoint_aligned_accounting_rows,
    build_response_curve,
    cluster_bootstrap_group_mean_contrast,
    spearman_correlation,
    summarize_accounting,
    summarize_endpoint_aligned_accounting,
    summarize_partition_contributions,
    summarize_recurrence_masking,
)


def test_group_mean_contrast_recomputes_two_hand_computed_means() -> None:
    rows = [
        {"request_id": "a", "surface": "repeat", "recovery": 1.0},
        {"request_id": "b", "surface": "repeat", "recovery": 0.5},
        {"request_id": "c", "surface": "no_overlap", "recovery": 0.2},
        {"request_id": "d", "surface": "no_overlap", "recovery": -0.2},
    ]
    result = cluster_bootstrap_group_mean_contrast(
        rows,
        {"a": "q1", "b": "q2", "c": "q1", "d": "q2"},
        group_field="surface",
        value_field="recovery",
        left_group="repeat",
        right_group="no_overlap",
        samples=20,
        seed=5,
    )
    assert result["left"]["mean"] == pytest.approx(0.75)
    assert result["right"]["mean"] == pytest.approx(0.0)
    assert result["point_estimate"] == pytest.approx(0.75)
    assert result["valid_bootstrap_samples"] == 20


def test_endpoint_aligned_accounting_and_conditional_counts() -> None:
    qc = {
        "p": {"ndcg@10": 0.5, "positive_eligible": True},
        "n": {"ndcg@10": 0.0, "positive_eligible": False},
    }
    response = {
        "p": {
            "null_ndcg@10": 0.25,
            "true_ndcg@10": 0.75,
            "positive_eligible": True,
            "direction_preferred_pairs": 2,
        },
        "n": {
            "null_ndcg@10": 0.0,
            "true_ndcg@10": 0.0,
            "positive_eligible": False,
            "direction_preferred_pairs": 0,
        },
    }
    rows = build_endpoint_aligned_accounting_rows(qc, response, qc)
    by_id = {row["request_id"]: row for row in rows}
    assert by_id["p"]["null_path_gap"] == pytest.approx(-0.25)
    assert by_id["p"]["same_checkpoint_recovery"] == pytest.approx(0.5)
    assert by_id["p"]["end_model_gap"] == pytest.approx(0.25)
    assert by_id["p"]["recovered_to_qc"] == 1
    summary = summarize_endpoint_aligned_accounting(
        rows,
        {"p": "q1", "n": "q2"},
        bootstrap_samples=20,
        seed=3,
    )
    assert summary["num_positive_eligible_requests"] == 1
    assert summary["num_direction_eligible_requests"] == 1


def test_arbitrary_partition_contribution_is_hand_computed() -> None:
    rows = [
        {"request_id": "a", "same_checkpoint_recovery": 0.8},
        {"request_id": "b", "same_checkpoint_recovery": -0.2},
        {"request_id": "c", "same_checkpoint_recovery": 0.0},
        {"request_id": "d", "same_checkpoint_recovery": 0.2},
    ]
    result = summarize_partition_contributions(
        all_rows=rows,
        partition_rows={"first": rows[:1], "rest": rows[1:]},
        metric="same_checkpoint_recovery",
    )
    assert result["all_mean"] == pytest.approx(0.2)
    assert result["surfaces"]["first"]["contribution"] == pytest.approx(0.2)
    assert result["surfaces"]["rest"]["contribution"] == pytest.approx(0.0)
    assert result["reconstructed_mean"] == pytest.approx(0.2)


def test_accounting_identity_and_request_states() -> None:
    qc = {
        "a": {"ndcg@10": 0.5},
        "b": {"ndcg@10": 0.5},
        "c": {"ndcg@10": 0.5},
        "d": {"ndcg@10": 0.5},
    }
    null = {
        "a": {"ndcg@10": 0.6},
        "b": {"ndcg@10": 0.3},
        "c": {"ndcg@10": 0.2},
        "d": {"ndcg@10": 0.5},
    }
    true = {
        "a": {"ndcg@10": 0.8},
        "b": {"ndcg@10": 0.6},
        "c": {"ndcg@10": 0.4},
        "d": {"ndcg@10": 0.4},
    }

    rows = build_accounting_rows(qc, null, true, qc)
    by_id = {row["request_id"]: row for row in rows}
    assert by_id["a"]["additive_gain"] == 1
    assert by_id["b"]["full_repair"] == 1
    assert by_id["c"]["repair_shortfall"] == 1
    assert by_id["d"]["nonpositive_history"] == 1
    for row in rows:
        assert row["net_value"] == pytest.approx(
            row["base_retention"] + row["history_utility"]
        )

    summary = summarize_accounting(
        rows,
        {request_id: request_id for request_id in qc},
        bootstrap_samples=20,
        seed=7,
    )
    assert sum(summary["means"][name] for name in (
        "additive_gain",
        "full_repair",
        "repair_shortfall",
        "nonpositive_history",
    )) == pytest.approx(1.0)
    assert summary["repayment_diagnostics"][
        "positive_history_on_damaged_base_rate"
    ] == pytest.approx(2 / 3)


def test_recurrence_masking_is_prevalence_weighted() -> None:
    rows = [
        {"request_id": "r", "history_utility": 0.8},
        {"request_id": "s1", "history_utility": 0.1},
        {"request_id": "s2", "history_utility": -0.1},
        {"request_id": "n", "history_utility": 0.0},
    ]
    result = summarize_recurrence_masking(
        all_rows=rows,
        repeat_rows=rows[:1],
        strict_nonrepeat_rows=rows[1:3],
        no_history_rows=rows[3:],
    )
    assert result["all_history_utility"] == pytest.approx(0.2)
    assert result["repeat"]["contribution"] == pytest.approx(0.2)
    assert result["strict_nonrepeat"]["contribution"] == pytest.approx(0.0)
    assert result["reconstructed_history_utility"] == pytest.approx(0.2)


def test_recurrence_masking_accepts_empty_repeat_surface() -> None:
    rows = [{"request_id": "s", "history_utility": 0.2}]
    result = summarize_recurrence_masking(
        all_rows=rows,
        repeat_rows=[],
        strict_nonrepeat_rows=rows,
        no_history_rows=[],
    )
    assert result["repeat"]["mean_history_utility"] is None
    assert result["repeat"]["contribution"] == 0.0
    assert result["strict_nonrepeat"]["contribution"] == pytest.approx(0.2)


def test_response_curve_uses_equal_count_bins_and_complete_pairs() -> None:
    response = {}
    intervention = {}
    clusters = {}
    for index in range(10):
        request_id = f"r{index}"
        response[request_id] = {
            "normalized_response_rms": float(index + 1),
            "pairwise_directional_accuracy": index / 10,
            "signed_delta_alignment": index / 10 - 0.5,
            "true_minus_null_ndcg@10": index / 100,
        }
        intervention[request_id] = {
            "actual_minus_random_ndcg@10": index / 200,
        }
        clusters[request_id] = f"q{index // 2}"

    result = build_response_curve(
        response,
        response,
        clusters,
        intervention_rows=intervention,
        bins=5,
        bootstrap_samples=20,
        seed=11,
    )
    assert [row["num_requests"] for row in result["bins"]] == [2] * 5
    assert result["bins"][0]["means"]["normalized_response_rms"] == 1.5
    assert result["bins"][-1]["means"]["normalized_response_rms"] == 9.5
    assert result["associations"]["true_minus_null_ndcg@10"]["spearman"] == pytest.approx(1.0)


def test_spearman_handles_ties_and_constant_outcome() -> None:
    assert spearman_correlation([(1, 1), (2, 2), (3, 3)]) == pytest.approx(1.0)
    assert spearman_correlation([(1, 3), (2, 2), (3, 1)]) == pytest.approx(-1.0)
    assert spearman_correlation([(1, 1), (2, 1), (3, 1)]) is None
    tied = spearman_correlation([(1, 1), (1, 1), (2, 2), (2, 3)])
    assert tied is not None and math.isfinite(tied)
