from __future__ import annotations

import math
from pathlib import Path

import pytest

from myrec.mechanism.matched_control_synthesis import (
    REGISTRATION_SHA256,
    STRICT_SURFACE,
    join_pair_rows,
    matched_control_synthesis_implementation_identity,
    summarize_strict_did,
    summarize_surfaces,
)
from myrec.mechanism.statistical_synthesis import normalized_query_fold
from myrec.utils.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _clusters_by_fold() -> dict[int, list[str]]:
    result = {0: [], 1: []}
    index = 0
    while any(len(result[fold]) < 2 for fold in (0, 1)):
        value = f"query{index}"
        fold = normalized_query_fold(value)
        if len(result[fold]) < 2:
            result[fold].append(value)
        index += 1
    return result


def _pair_row(
    request_id: str,
    cluster: str,
    *,
    ndcg: float,
    margin: float | None,
    surface: str = STRICT_SURFACE,
) -> dict[str, object]:
    return {
        "request_id": request_id,
        "normalized_query_cluster": cluster,
        "target_aware_surface": surface,
        "target_margin_change": margin,
        "treatment_minus_control_ndcg@10": ndcg,
    }


def test_hand_computed_join_fold_means_and_two_endpoint_family() -> None:
    clusters = _clusters_by_fold()
    original = []
    balanced = []
    # The four joined NDCG DIDs are .2, .4, .6, .8; margin DIDs are
    # .5, .1, .3, .7.  Overall means are therefore .5 and .4.
    ndcg_dids = (0.2, 0.4, 0.6, 0.8)
    margin_dids = (0.5, 0.1, 0.3, 0.7)
    ordered_clusters = [
        clusters[0][0],
        clusters[0][1],
        clusters[1][0],
        clusters[1][1],
    ]
    for index, (cluster, ndcg, margin) in enumerate(
        zip(ordered_clusters, ndcg_dids, margin_dids)
    ):
        request_id = f"r{index}"
        original.append(_pair_row(request_id, cluster, ndcg=0.0, margin=0.0))
        balanced.append(_pair_row(request_id, cluster, ndcg=ndcg, margin=margin))

    joined = join_pair_rows(original, balanced)
    assert [row["ndcg_did"] for row in joined] == pytest.approx(ndcg_dids)
    assert [row["margin_did"] for row in joined] == pytest.approx(margin_dids)
    result = summarize_strict_did(joined, samples=200, seed=20260715)
    endpoints = result["endpoints"]
    ndcg = endpoints["strict_transfer_ndcg_history_response_did"]
    margin = endpoints["strict_transfer_target_margin_change_did"]
    assert ndcg["mean"] == pytest.approx(0.5)
    assert margin["mean"] == pytest.approx(0.4)
    assert ndcg["folds"]["0"]["mean"] == pytest.approx(0.3)
    assert ndcg["folds"]["1"]["mean"] == pytest.approx(0.7)
    assert margin["folds"]["0"]["mean"] == pytest.approx(0.3)
    assert margin["folds"]["1"]["mean"] == pytest.approx(0.5)
    assert ndcg["direction_consistent_in_both_folds"] is True
    assert margin["direction_consistent_in_both_folds"] is True
    assert result["fdr"]["family_size"] == 2
    assert {
        row["hypothesis_id"] for row in result["fdr"]["results"]
    } == {
        "strict_transfer_ndcg_history_response_did",
        "strict_transfer_target_margin_change_did",
    }
    for row in result["fdr"]["results"]:
        assert 0.0 <= row["raw_p"] <= row["q_value"] <= 1.0


def test_surface_summaries_are_hand_computed_and_strict_only_is_inferential() -> None:
    clusters = _clusters_by_fold()
    original = [
        _pair_row("strict", clusters[0][0], ndcg=0.1, margin=0.2),
        _pair_row(
            "repeat",
            clusters[1][0],
            ndcg=0.4,
            margin=None,
            surface="target_repeat",
        ),
    ]
    balanced = [
        _pair_row("strict", clusters[0][0], ndcg=0.4, margin=0.7),
        _pair_row(
            "repeat",
            clusters[1][0],
            ndcg=0.2,
            margin=None,
            surface="target_repeat",
        ),
    ]
    summaries = summarize_surfaces(join_pair_rows(original, balanced))
    assert summaries[STRICT_SURFACE][
        "balanced_minus_original_history_response_ndcg@10"
    ] == pytest.approx(0.3)
    assert summaries[STRICT_SURFACE][
        "balanced_minus_original_target_margin_change"
    ] == pytest.approx(0.5)
    assert summaries["target_repeat"][
        "balanced_minus_original_history_response_ndcg@10"
    ] == pytest.approx(-0.2)
    assert summaries["all"][
        "balanced_minus_original_history_response_ndcg@10"
    ] == pytest.approx(0.05)
    assert summaries["all"]["num_margin_eligible_requests"] == 1


@pytest.mark.parametrize(
    "field,replacement,pattern",
    [
        ("normalized_query_cluster", "different", "cluster differs"),
        ("target_aware_surface", "target_repeat", "surface differs"),
    ],
)
def test_join_rejects_request_binding_drift(
    field: str, replacement: str, pattern: str
) -> None:
    cluster = _clusters_by_fold()[0][0]
    original = [_pair_row("r", cluster, ndcg=0.0, margin=0.0)]
    balanced = [_pair_row("r", cluster, ndcg=0.1, margin=0.1)]
    balanced[0][field] = replacement
    with pytest.raises(ValueError, match=pattern):
        join_pair_rows(original, balanced)


def test_join_rejects_request_set_margin_eligibility_and_nonfinite_values() -> None:
    cluster = _clusters_by_fold()[0][0]
    original = [_pair_row("r", cluster, ndcg=0.0, margin=None)]
    balanced = [_pair_row("r", cluster, ndcg=0.1, margin=0.1)]
    with pytest.raises(ValueError, match="margin eligibility"):
        join_pair_rows(original, balanced)
    with pytest.raises(ValueError, match="request sets differ"):
        join_pair_rows(original, balanced + [_pair_row("x", cluster, ndcg=0.0, margin=0.0)])
    balanced = [_pair_row("r", cluster, ndcg=math.inf, margin=None)]
    with pytest.raises(ValueError, match="finite"):
        join_pair_rows(original, balanced)


def test_registration_and_producer_identity_are_hash_bound() -> None:
    registration = (
        ROOT / "experiments/motivation/m3_matched_did_synthesis_registration.yaml"
    )
    assert sha256_file(registration) == REGISTRATION_SHA256
    identity = matched_control_synthesis_implementation_identity()
    paths = {row["path"] for row in identity["files"]}
    assert registration.relative_to(ROOT).as_posix() in paths
    assert "src/myrec/mechanism/statistical_synthesis.py" in paths
    assert len(identity["digest"]) == 64


def test_synthesis_source_has_no_shared_evaluator_or_qrels_file_dependency() -> None:
    source = (
        ROOT / "src/myrec/mechanism/matched_control_synthesis.py"
    ).read_text(encoding="utf-8")
    assert "from myrec.mechanism.evaluator" not in source
    assert "qrels_dev.jsonl" not in source
    assert "qrels_train.jsonl" not in source
