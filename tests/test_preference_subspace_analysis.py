import math

import numpy as np

from scripts.analyze_deep_dive_preference_subspace import (
    _aggregate_by_request,
    _build_cross_position_transport_rows,
    _probe_basis,
    _row_metrics,
    _subspace_similarity,
    _summarize,
)


def test_row_metrics_has_hand_computed_rank_one_projection() -> None:
    delta = np.asarray([[1.0, 0.0], [0.0, 2.0]])
    scale = np.ones(2)
    coefficient = np.asarray([[-1.0, 0.0], [1.0, 0.0]])

    result = _row_metrics(delta, scale, coefficient)

    assert result["subspace_rank"] == 1
    np.testing.assert_allclose(result["energy_fraction"], [1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        result["standardized_delta_rms"],
        [1.0 / math.sqrt(2.0), math.sqrt(2.0)],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result["discriminative_logit_delta_rms"], [1.0, 0.0], atol=1e-12
    )


def test_row_metrics_records_rank_zero_as_zero_projection() -> None:
    result = _row_metrics(
        np.asarray([[3.0, 4.0]]),
        np.ones(2),
        np.zeros((2, 2)),
    )

    assert result["subspace_rank"] == 0
    np.testing.assert_allclose(result["energy_fraction"], [0.0], atol=0.0)
    np.testing.assert_allclose(result["standardized_delta_rms"], [math.sqrt(12.5)])
    np.testing.assert_allclose(result["discriminative_logit_delta_rms"], [0.0])


def test_candidate_rows_are_request_weighted_before_summary() -> None:
    row_metrics = {
        "subspace_rank": 1,
        "nonzero": np.asarray([True, True, True]),
        "energy_fraction": np.asarray([0.25, 0.75, 0.5]),
        "standardized_delta_rms": np.asarray([1.0, 3.0, 4.0]),
        "discriminative_logit_delta_rms": np.asarray([2.0, 4.0, 6.0]),
    }
    rows = _aggregate_by_request(
        ["r1", "r1", "r2"], ["query-a", "query-a", "query-b"], row_metrics
    )

    assert rows == [
        {
            "request_id": "r1",
            "normalized_query": "query-a",
            "nonzero": True,
            "energy_fraction": 0.5,
            "standardized_delta_rms": 2.0,
            "discriminative_logit_delta_rms": 3.0,
        },
        {
            "request_id": "r2",
            "normalized_query": "query-b",
            "nonzero": True,
            "energy_fraction": 0.5,
            "standardized_delta_rms": 4.0,
            "discriminative_logit_delta_rms": 6.0,
        },
    ]
    summary = _summarize(
        "q2", "candidate_readout", "category", 13, "real_labels", "all", rows, 1, 2
    )
    assert summary["requests"] == 2
    assert summary["mean_energy_fraction"] == 0.5
    assert summary["isotropic_rank_over_hidden_baseline"] == 0.5
    assert summary["mean_fraction_over_isotropic_baseline"] == 1.0


def test_probe_transport_uses_raw_coordinate_principal_angle() -> None:
    horizontal = _probe_basis(
        np.asarray([2.0, 1.0]),
        np.asarray([[-2.0, 0.0], [2.0, 0.0]]),
    )
    diagonal = _probe_basis(
        np.ones(2),
        np.asarray([[-1.0, -1.0], [1.0, 1.0]]),
    )

    result = _subspace_similarity(horizontal, diagonal)

    assert result["left_rank"] == 1
    assert result["right_rank"] == 1
    assert math.isclose(result["mean_squared_canonical_cosine"], 0.5, abs_tol=1e-12)
    assert math.isclose(result["minimum_canonical_cosine"], 1.0 / math.sqrt(2.0))
    assert math.isclose(result["maximum_canonical_cosine"], 1.0 / math.sqrt(2.0))


def test_probe_transport_preserves_explicit_rank_zero_boundary() -> None:
    empty = _probe_basis(np.ones(2), np.zeros((2, 2)))
    horizontal = np.asarray([[1.0, 0.0]])

    assert empty.shape == (0, 2)
    assert _subspace_similarity(empty, horizontal) == {
        "left_rank": 0,
        "right_rank": 1,
        "mean_squared_canonical_cosine": None,
        "minimum_canonical_cosine": None,
        "maximum_canonical_cosine": None,
    }


def test_cross_position_transport_has_hand_computed_orthogonal_residual() -> None:
    history = np.tile(np.asarray([[[1.0, 0.0]]]), (1, 29, 1))
    candidate = np.tile(np.asarray([[[0.0, 1.0]]]), (1, 29, 1))
    rows = _build_cross_position_transport_rows(
        "q2",
        {
            "history_summary_end": history,
            "candidate_readout": candidate,
        },
        {"r1": "query"},
        [{"request_id": "r1"}],
        ["r1"],
    )

    all_fold_state_13 = next(
        row
        for row in rows
        if row["hidden_state_index"] == 13
        and row["normalized_query_fold"] == "all"
    )
    assert all_fold_state_13["valid_delta_pairs"] == 1
    assert all_fold_state_13["mean_cosine"] == 0.0
    assert all_fold_state_13["mean_cosine_squared"] == 0.0
    assert all_fold_state_13["mean_candidate_over_history_rms"] == 1.0
    assert all_fold_state_13["mean_signed_candidate_projection_scale"] == 0.0
    assert math.isclose(
        all_fold_state_13["mean_orthogonal_residual_rms"],
        1.0 / math.sqrt(2.0),
    )
