from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmp.build_mechanism_first_diagnosis import (  # noqa: E402
    CloseoutError,
    KNOWN_MECHANICAL_NON_RESULT_PATHS,
    validate_known_m3_matched_checkpoint_dir_failure,
    RESULT_NUMBER_RE,
    SYSTEM_COMPONENT_COVERAGE,
    _numeric_token_match,
    assert_no_outcome_selected_exclusion,
    assert_no_placeholders,
    assert_no_test_access,
    assert_no_unauthorized_architecture_claims,
    expected_artifact_ids,
    read_json,
    require_exact_once_coverage,
    require_no_orphan_ids,
    validate_static_registries,
    validate_known_m2_representation_query_hash_failure,
)
from tmp.m3_matched_control_supervisor import training_checkpoint_dir  # noqa: E402
from myrec.baselines.motivation_v12_ranker import CHECKPOINT_DIRNAME  # noqa: E402


@pytest.mark.parametrize(
    "value",
    [
        "results remain pending",
        "await final result",
        "coming soon",
        "???",
        "<fill this>",
        "N/A",
        "to follow",
        "unknown yet",
    ],
)
def test_placeholder_variants_are_rejected(value: str) -> None:
    with pytest.raises(CloseoutError):
        assert_no_placeholders({"value": value}, "test")


def test_only_exact_authorization_pending_sentinel_is_allowed() -> None:
    assert_no_placeholders(
        {
            "integrated_diagnosis": {
                "next_direction": "pending user authorization"
            }
        },
        "test",
    )


@pytest.mark.parametrize(
    "value",
    [
        "源测试集已经打开，并据此完成了结论。",
        "The held-out population was opened and used.",
        "The test corpus was read.",
    ],
)
def test_affirmative_test_access_prose_is_rejected(value: str) -> None:
    with pytest.raises(CloseoutError):
        assert_no_test_access({"claim": value}, "test")


@pytest.mark.parametrize(
    "value",
    [
        "The held-out evidence boundary remains closed.",
        "源测试集从未打开。",
    ],
)
def test_negative_test_boundary_attestations_are_allowed(value: str) -> None:
    assert_no_test_access({"claim": value}, "test")


@pytest.mark.parametrize(
    "value",
    [
        "We implemented and deployed the transfer architecture.",
        "已经实现并部署迁移架构。",
    ],
)
def test_implemented_architecture_claim_is_rejected(value: str) -> None:
    with pytest.raises(CloseoutError):
        assert_no_unauthorized_architecture_claims({"claim": value}, "test")


def test_design_only_architecture_attestation_is_allowed() -> None:
    assert_no_unauthorized_architecture_claims(
        {"claim": "No transfer architecture has been implemented."}, "test"
    )


def test_formal_aggregate_coverage_is_exact_once() -> None:
    require_exact_once_coverage(["a", "b"], {"a", "b"}, "test")
    with pytest.raises(CloseoutError):
        require_exact_once_coverage(["a", "a", "b"], {"a", "b"}, "test")


def test_matched_did_is_in_static_closeout_contract_exactly_once() -> None:
    artifact_ids = expected_artifact_ids()
    assert len(artifact_ids) == 18
    assert artifact_ids.count("m3.q2.matched_did_statistics") == 1
    counts, run_ids = validate_static_registries()
    assert len(run_ids) == 88
    assert counts["m3_matched_did_analyses"] == 1
    assert "20260717_kuaisearch_mech_m3_q2_matched_did_analysis" in run_ids


def test_contradictions_cannot_be_orphaned() -> None:
    require_no_orphan_ids({"c1"}, {"c1"}, "test")
    with pytest.raises(CloseoutError):
        require_no_orphan_ids({"c1", "c2"}, {"c1"}, "test")


@pytest.mark.parametrize(
    "payload",
    [
        '{"x": 1, "x": 2}',
        '{"x": NaN}',
        '{"x": Infinity}',
        '{"x": -Infinity}',
    ],
)
def test_closeout_json_parser_rejects_duplicate_or_nonfinite_values(
    tmp_path: Path, payload: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(CloseoutError):
        read_json(path)


def test_outcome_selected_mechanical_prose_is_rejected() -> None:
    with pytest.raises(CloseoutError):
        assert_no_outcome_selected_exclusion(
            "Completed valid result with favorable metrics; excluded because its "
            "outcome is inconvenient.",
            "test",
        )


def test_numeric_claim_token_must_match_a_producer_value() -> None:
    text = "The rate is 22.675% and the effect is +0.0031."
    tokens = list(RESULT_NUMBER_RE.finditer(text))
    sources = [("producer:$.rate", 0.22675), ("producer:$.effect", 0.003122)]
    assert _numeric_token_match(tokens[0], text, sources) is not None
    assert _numeric_token_match(tokens[1], text, sources) is not None

    fabricated = "The effect is +123456789.000000."
    token = next(RESULT_NUMBER_RE.finditer(fabricated))
    assert _numeric_token_match(token, fabricated, sources) is None


def test_first_diagnosis_cannot_claim_component_tested() -> None:
    assert SYSTEM_COMPONENT_COVERAGE == {"partial", "untested"}


def test_mechanical_failure_provenance_is_frozen_as_mechanical_only() -> None:
    assert len(KNOWN_MECHANICAL_NON_RESULT_PATHS) == 8
    assert (
        KNOWN_MECHANICAL_NON_RESULT_PATHS[
            "NR_M2_REPRESENTATION_RAW_QUERY_HASH_MECHANICAL_FAILURE"
        ]
        == {
            "tmp/m2_representation_failures/20260717_q2_pre_qrels_raw_query_hash_bug"
        }
    )
    assert (
        KNOWN_MECHANICAL_NON_RESULT_PATHS[
            "NR_M3_MATCHED_CHECKPOINT_DIR_CONTRACT_FAILURE"
        ]
        == {
            "tmp/m3_matched_control_failures/20260717_checkpoint_dir_contract_bug/failure.json"
        }
    )
    validate_known_m2_representation_query_hash_failure()
    validate_known_m3_matched_checkpoint_dir_failure()


def test_matched_supervisor_uses_shared_checkpoint_layout(tmp_path: Path) -> None:
    output_root = tmp_path / "matched-output"
    assert training_checkpoint_dir(output_root) == output_root / CHECKPOINT_DIRNAME
    assert CHECKPOINT_DIRNAME == "checkpoint_latest"
