from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.mechanism.evaluator import (
    MECHANISM_PROBE_MANIFEST_PATH,
    MECHANISM_PROBE_MANIFEST_SHA256,
    _evaluate_request,
    _resolved_condition_id,
    evaluate_mechanism_probe,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path, *, malformed_qrels: bool = False) -> dict[str, Path]:
    standardized = tmp_path / "standardized"
    runs = tmp_path / "runs"
    candidate_manifest = standardized / "candidate_manifest.json"
    request_manifest = standardized / "request_manifest.json"
    _write_json(
        candidate_manifest,
        {
            "dataset_version": "tiny-v1",
            "entries": [
                {
                    "split": "dev",
                    "request_id": "r1",
                    "candidate_item_ids": ["a", "b", "c"],
                }
            ],
        },
    )
    _write_json(
        request_manifest,
        {
            "dataset_version": "tiny-v1",
            "entries": [
                {
                    "split": "dev",
                    "request_id": "r1",
                    "candidate_item_ids_sha256": hashlib.sha256(
                        json.dumps(["a", "b", "c"], separators=(",", ":")).encode()
                    ).hexdigest(),
                    "query_sha256": hashlib.sha256(
                        " Red  Shoe ".encode()
                    ).hexdigest(),
                }
            ],
        },
    )
    _write_jsonl(
        standardized / "records_dev.jsonl",
        [
            {
                "request_id": "r1",
                "query": " Red  Shoe ",
                "history": [{"item_id": "a"}],
                "candidates": [
                    {"item_id": "a"},
                    {"item_id": "b"},
                    {"item_id": "c"},
                ],
            }
        ],
    )
    qrels = standardized / "qrels_dev.jsonl"
    if malformed_qrels:
        qrels.write_text("{not valid json\n", encoding="utf-8")
    else:
        _write_jsonl(
            qrels,
            [
                {
                    "request_id": "r1",
                    "clicked": ["a", "b"],
                    "purchased": ["a"],
                    "relevance": {"a": 2, "b": 1},
                }
            ],
        )

    base_signature = {"checkpoint_head": "rank", "max_length": 32}
    common = {
        "candidate_manifest_sha256": _sha(candidate_manifest),
        "checkpoint_id": "checkpoint-1",
        "dataset_id": "tiny",
        "dataset_version": "tiny-v1",
        "method_id": "q2",
        "qrels_read": False,
        "request_manifest_sha256": _sha(request_manifest),
        "split": "dev",
    }
    run_specs = {
        "treatment": {
            "condition_id": "relevant-only",
            "scores": {"a": 3.0, "b": 2.0, "c": 1.0},
            "metadata": {
                **common,
                "base_scoring_signature": base_signature,
                "scoring_signature": {"intervention_implementation": "treatment-v1"},
            },
        },
        "control": {
            "condition_id": "full",
            "scores": {"a": 1.0, "b": 2.0, "c": 3.0},
            "metadata": {
                **common,
                # Exercise the backward-compatible fallback independently.
                "scoring_signature": base_signature,
                "intervention_implementation": "control-v2",
            },
        },
    }
    for run_id, spec in run_specs.items():
        run_dir = runs / run_id
        _write_json(
            run_dir / "metadata.json",
            {**spec["metadata"], "condition_id": spec["condition_id"]},
        )
        _write_jsonl(
            run_dir / "scores.jsonl",
            [
                {
                    "request_id": "r1",
                    "candidate_item_id": item_id,
                    "score": score,
                }
                for item_id, score in spec["scores"].items()
            ],
        )
    return {
        "candidate_manifest": candidate_manifest,
        "ledger": tmp_path / "reports" / "dev_eval_log.jsonl",
        "runs": runs,
        "standardized": standardized,
    }


def _evaluate(paths: dict[str, Path], analysis_run_id: str = "analysis"):
    return evaluate_mechanism_probe(
        analysis_run_id=analysis_run_id,
        treatment_run_id="treatment",
        control_run_id="control",
        standardized_dir=paths["standardized"],
        candidate_manifest_path=paths["candidate_manifest"],
        runs_dir=paths["runs"],
        dev_eval_log_path=paths["ledger"],
    )


def _mark_mechanism_treatment_with_frozen_control(
    paths: dict[str, Path],
    declaration: dict | None,
) -> None:
    treatment_path = paths["runs"] / "treatment" / "metadata.json"
    treatment = json.loads(treatment_path.read_text())
    treatment["evidence_mode"] = "mechanism_diagnostic"
    treatment["intervention"] = {"condition_id": treatment["condition_id"]}
    if declaration is not None:
        treatment["mechanism_probe_manifest"] = declaration
    _write_json(treatment_path, treatment)

    control_path = paths["runs"] / "control" / "metadata.json"
    control = json.loads(control_path.read_text())
    control.pop("condition_id")
    control["evidence_mode"] = "first_round_pilot"
    control["history_condition"] = "true"
    _write_json(control_path, control)


def _valid_probe_manifest_declaration() -> dict:
    return {
        "expected_sha256": MECHANISM_PROBE_MANIFEST_SHA256,
        "path": MECHANISM_PROBE_MANIFEST_PATH,
        "sha256": MECHANISM_PROBE_MANIFEST_SHA256,
        "verified": True,
    }


def test_hand_computed_graded_ndcg_margin_surfaces_and_ledger(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    metrics = _evaluate(paths)

    ideal_dcg = 3.0 + 1.0 / math.log2(3)
    control_dcg = 1.0 / math.log2(3) + 3.0 / math.log2(4)
    expected_delta = 1.0 - control_dcg / ideal_dcg
    assert metrics["mean_treatment_ndcg@10"] == pytest.approx(1.0)
    assert metrics["mean_control_ndcg@10"] == pytest.approx(control_dcg / ideal_dcg)
    assert metrics["mean_treatment_minus_control_ndcg@10"] == pytest.approx(
        expected_delta
    )
    # treatment: 3(a)-2(b)=1; control: 1(a)-3(c)=-2.
    assert metrics["mean_target_margin_change"] == pytest.approx(3.0)
    assert metrics["query_cluster_ci95"]["target_margin_change"] == [3.0, 3.0]
    assert metrics["treatment_condition_id"] == "relevant-only"
    assert metrics["control_condition_id"] == "full"
    assert metrics["surfaces"]["observed_positive"]["num_requests"] == 1
    assert metrics["surfaces"]["target_repeat"]["num_requests"] == 1
    contributions = metrics["population_weighted_ndcg_contributions"]
    assert contributions["reconstructed_mean"] == pytest.approx(expected_delta)

    analysis_dir = paths["runs"] / "analysis"
    audit = json.loads((analysis_dir / "pre_qrels_audit.json").read_text())
    assert audit["qrels_read"] is False
    assert audit["status"] == "passed"
    row = json.loads((analysis_dir / "per_request.jsonl").read_text())
    assert row["target_item_id"] == "a"
    assert row["treatment_best_lower_gain_competitor_item_id"] == "b"
    assert row["control_best_lower_gain_competitor_item_id"] == "c"
    assert row["target_margin_change"] == pytest.approx(3.0)
    assert (analysis_dir / "metadata.json").is_file()
    assert (analysis_dir / "target_aware_surfaces" / "manifest.json").is_file()

    ledger_rows = paths["ledger"].read_text(encoding="utf-8").splitlines()
    assert len(ledger_rows) == 1
    ledger = json.loads(ledger_rows[0])
    assert ledger["run_id"] == "analysis"
    assert ledger["treatment_minus_control_ndcg@10"] == pytest.approx(expected_delta)
    assert ledger["target_margin_change"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("incomplete", "candidate coverage mismatch"),
        ("nonfinite", "non-finite score"),
        ("candidate_hash", "candidate manifest hash mismatch"),
        ("request_hash", "request manifest hash mismatch"),
        ("duplicate", "duplicate score"),
    ],
)
def test_bad_bundle_fails_before_malformed_qrels(
    tmp_path: Path, mutation: str, message: str
) -> None:
    paths = _fixture(tmp_path, malformed_qrels=True)
    scores_path = paths["runs"] / "treatment" / "scores.jsonl"
    metadata_path = paths["runs"] / "treatment" / "metadata.json"
    if mutation == "incomplete":
        rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
        _write_jsonl(scores_path, rows[:-1])
    elif mutation == "nonfinite":
        rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
        rows[0]["score"] = float("nan")
        _write_jsonl(scores_path, rows)
    elif mutation in {"candidate_hash", "request_hash"}:
        metadata = json.loads(metadata_path.read_text())
        metadata[
            "candidate_manifest_sha256"
            if mutation == "candidate_hash"
            else "request_manifest_sha256"
        ] = "0" * 64
        _write_json(metadata_path, metadata)
    else:
        rows = [json.loads(line) for line in scores_path.read_text().splitlines()]
        _write_jsonl(scores_path, [*rows, rows[0]])

    with pytest.raises(ValueError, match=message):
        _evaluate(paths, analysis_run_id=f"analysis-{mutation}")


def test_pre_qrels_audit_is_written_before_qrels_parse(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, malformed_qrels=True)
    with pytest.raises(ValueError, match="invalid JSONL row"):
        _evaluate(paths)
    audit = paths["runs"] / "analysis" / "pre_qrels_audit.json"
    assert json.loads(audit.read_text())["qrels_read"] is False
    assert not (paths["runs"] / "analysis" / "metrics.json").exists()


def test_no_observed_positive_has_no_target_or_margin(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    _write_jsonl(
        paths["standardized"] / "qrels_dev.jsonl",
        [{"request_id": "r1", "clicked": [], "purchased": [], "relevance": {}}],
    )
    metrics = _evaluate(paths)
    row = json.loads(
        (paths["runs"] / "analysis" / "per_request.jsonl").read_text()
    )
    assert row["target_item_id"] is None
    assert row["target_candidate_position"] is None
    assert row["target_margin_change"] is None
    assert metrics["mean_target_margin_change"] is None
    assert metrics["surfaces"]["no_observed_positive"]["num_requests"] == 1


def test_base_scoring_signature_is_a_strict_pair_invariant(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, malformed_qrels=True)
    metadata_path = paths["runs"] / "control" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["scoring_signature"]["max_length"] = 64
    _write_json(metadata_path, metadata)
    with pytest.raises(ValueError, match="base_scoring_signature mismatch"):
        _evaluate(paths)


def test_highest_gain_target_and_score_ties_use_candidate_order() -> None:
    row = _evaluate_request(
        request_id="r",
        item_ids=["first", "second", "competitor-a", "competitor-b"],
        gains={"first": 2.0, "second": 2.0},
        treatment_scores={
            "first": 2.0,
            "second": 9.0,
            "competitor-a": 1.0,
            "competitor-b": 1.0,
        },
        control_scores={
            "first": 1.0,
            "second": 9.0,
            "competitor-a": 0.0,
            "competitor-b": 0.0,
        },
    )
    assert row["target_item_id"] == "first"
    assert row["target_candidate_position"] == 0
    assert row["treatment_best_lower_gain_competitor_item_id"] == "competitor-a"
    assert row["control_best_lower_gain_competitor_item_id"] == "competitor-a"


@pytest.mark.parametrize(
    ("history_condition", "expected"),
    [
        ("true", "frozen_full"),
        ("null", "frozen_null"),
        ("wrong", "frozen_wrong_user"),
        ("relevant_6", None),
    ],
)
def test_only_frozen_history_conditions_receive_compatibility_ids(
    history_condition: str, expected: str | None
) -> None:
    assert _resolved_condition_id({"history_condition": history_condition}) == expected


def test_frozen_control_condition_fallback_retains_raw_history_condition(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    metadata_path = paths["runs"] / "control" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.pop("condition_id")
    metadata["history_condition"] = "true"
    _write_json(metadata_path, metadata)

    metrics = _evaluate(paths)
    assert metrics["control_condition_id"] == "frozen_full"
    assert metrics["control_raw_history_condition"] == "true"
    output_metadata = json.loads(
        (paths["runs"] / "analysis" / "metadata.json").read_text()
    )
    assert output_metadata["conditions"]["control"] == {
        "condition_id": "frozen_full",
        "raw_condition_id": None,
        "raw_history_condition": "true",
        "run_id": "control",
    }


def test_mechanism_probe_manifest_verified_with_explicit_frozen_exemption(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)
    _mark_mechanism_treatment_with_frozen_control(
        paths,
        _valid_probe_manifest_declaration(),
    )
    metrics = _evaluate(paths)
    admission = metrics["mechanism_probe_manifest_admission"]
    assert admission["pair_contains_mechanism_intervention"] is True
    assert admission["runs"]["treatment"]["status"] == "verified"
    assert admission["runs"]["treatment"]["legacy_exemption"] is False
    assert admission["runs"]["control"]["status"] == "legacy_exemption"
    assert admission["runs"]["control"]["legacy_exemption"] is True
    audit = json.loads(
        (paths["runs"] / "analysis" / "pre_qrels_audit.json").read_text()
    )
    assert audit["mechanism_probe_manifest_admission"] == admission


@pytest.mark.parametrize(
    ("tampered_field", "tampered_value"),
    [
        ("path", "experiments/motivation/not_the_frozen_manifest.yaml"),
        ("sha256", "0" * 64),
        ("expected_sha256", "0" * 64),
        ("verified", False),
        ("verified", 1),
    ],
)
def test_mechanism_probe_manifest_tamper_fails_before_malformed_qrels(
    tmp_path: Path,
    tampered_field: str,
    tampered_value,
) -> None:
    paths = _fixture(tmp_path, malformed_qrels=True)
    declaration = _valid_probe_manifest_declaration()
    declaration[tampered_field] = tampered_value
    _mark_mechanism_treatment_with_frozen_control(paths, declaration)
    with pytest.raises(
        ValueError,
        match=rf"mechanism_probe_manifest {tampered_field} mismatch",
    ):
        _evaluate(paths)


def test_new_mechanism_run_without_probe_manifest_fails_closed(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path, malformed_qrels=True)
    _mark_mechanism_treatment_with_frozen_control(paths, None)
    with pytest.raises(ValueError, match="missing mechanism_probe_manifest"):
        _evaluate(paths)


@pytest.mark.parametrize("split", ["confirmation", "test"])
def test_confirmation_and_test_are_locked_before_qrels(
    tmp_path: Path, split: str
) -> None:
    paths = _fixture(tmp_path, malformed_qrels=True)
    with pytest.raises(ValueError, match="confirmation and test are locked"):
        evaluate_mechanism_probe(
            analysis_run_id="analysis",
            treatment_run_id="treatment",
            control_run_id="control",
            standardized_dir=paths["standardized"],
            candidate_manifest_path=paths["candidate_manifest"],
            split=split,
            runs_dir=paths["runs"],
            dev_eval_log_path=paths["ledger"],
        )
