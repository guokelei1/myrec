from __future__ import annotations

import json
import math
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.mechanism import scorer
from myrec.utils.hashing import sha256_file


METHOD_ID = "q0_qwen3_reranker_06b"
CHECKPOINT_ID = f"{METHOD_ID}@fixture"
REFERENCE_RUN_ID = "20260717_kuaisearch_q0_fixture_internal_dev_full_score"
FROZEN_IDENTITY = {
    "digest": "f" * 64,
    "files": [{"path": "frozen.py", "sha256": "e" * 64}],
}
CHECKPOINT_FILES = [
    {"name": "model.safetensors", "sha256": "d" * 64, "size_bytes": 7}
]


class _Model:
    def eval(self):
        return self


def _fake_score(model, tokenizer, record, history, config, *, device, batch_size):
    del model, tokenizer, config, device, batch_size
    offset = 0.25 * len(history)
    return {
        str(candidate["item_id"]): float(index) + offset
        for index, candidate in enumerate(record.candidates)
    }, False


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _score_rows(records, histories):
    rows = []
    for record, history in zip(records, histories):
        offset = 0.25 * len(history)
        for index, candidate in enumerate(record["candidates"]):
            rows.append(
                {
                    "candidate_item_id": str(candidate["item_id"]),
                    "method_id": METHOD_ID,
                    "request_id": record["request_id"],
                    "score": float(index) + offset,
                }
            )
    return rows


def _make_fixture(tmp_path: Path, *, condition_id: str = "probe"):
    standardized = tmp_path / "standardized"
    standardized.mkdir(parents=True)
    histories = [
        [
            {
                "item_id": "h1",
                "title": "prior",
                "brand": "b",
                "cat": ["c"],
                "event": "click",
                "query": "old",
                "ts": 5,
            }
        ],
        [],
    ]
    records = [
        {
            "request_id": "r1",
            "user_id": "u1",
            "session_id": "s1",
            "ts": 10,
            "query": "query one",
            "history": histories[0],
            "candidates": [
                {"item_id": "a", "title": "A", "brand": "x", "cat": ["c"]},
                {"item_id": "b", "title": "B", "brand": "y", "cat": ["c"]},
            ],
            "masks": {},
        },
        {
            "request_id": "r2",
            "user_id": "u2",
            "session_id": "s2",
            "ts": 20,
            "query": "query two",
            "history": histories[1],
            "candidates": [
                {"item_id": "c", "title": "C", "brand": "x", "cat": ["d"]},
                {"item_id": "d", "title": "D", "brand": "y", "cat": ["d"]},
            ],
            "masks": {},
        },
    ]
    records_path = standardized / "records_dev.jsonl"
    _write_jsonl(records_path, records)
    dataset_manifest = {
        "schema_version": 1,
        "dataset_id": "kuaisearch",
        "dataset_version": "full_confirm_preceding40k_v11",
    }
    _write_json(standardized / "manifest.json", dataset_manifest)
    _write_json(standardized / "candidate_manifest.json", {"fixture": "candidate"})
    _write_json(standardized / "request_manifest.json", {"fixture": "request"})
    # This file is deliberately malformed.  A guarded Path.open in one test
    # proves that the scorer does not merely tolerate it; it never opens it.
    (standardized / "qrels_dev.jsonl").write_text("{malformed\n", encoding="utf-8")

    population = {
        "dataset_version": dataset_manifest["dataset_version"],
        "internal_dev_requests": len(records),
        "manifest_sha256": sha256_file(standardized / "manifest.json"),
        "candidate_manifest_sha256": sha256_file(
            standardized / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(standardized / "request_manifest.json"),
        "records_dev_sha256": sha256_file(records_path),
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = {
        "_config_sha256": sha256_file(config_path),
        "_protocol": {"data": {"development_population": population}},
        "method_id": METHOD_ID,
        "model": {},
        "protocol": {"sha256": "a" * 64},
        "scoring": {"batch_size": 2},
        "training": {},
    }
    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / "checkpoint_latest" / "model").mkdir(parents=True)
    _write_json(
        checkpoint_root / "training_metadata.json",
        {"checkpoint_id": CHECKPOINT_ID},
    )

    runs_dir = tmp_path / "runs"
    reference_dir = runs_dir / REFERENCE_RUN_ID
    reference_dir.mkdir(parents=True)
    reference_rows = _score_rows(records, histories)
    _write_jsonl(reference_dir / "scores.jsonl", reference_rows)
    signature = {
        "config_sha256": config["_config_sha256"],
        "implementation_digest": FROZEN_IDENTITY["digest"],
        "method_id": METHOD_ID,
        "protocol_sha256": config["protocol"]["sha256"],
        "prompt_contract": "fixture-frozen",
    }
    _write_json(
        reference_dir / "metadata.json",
        {
            "candidate_manifest_sha256": population["candidate_manifest_sha256"],
            "checkpoint_id": CHECKPOINT_ID,
            "checkpoint_weight_files": CHECKPOINT_FILES,
            "config_sha256": config["_config_sha256"],
            "dataset_id": "kuaisearch",
            "dataset_version": population["dataset_version"],
            "evidence_mode": "first_round_pilot",
            "history_condition": "true",
            "implementation_identity": FROZEN_IDENTITY,
            "method_id": METHOD_ID,
            "qrels_read": False,
            "request_count": len(records),
            "request_manifest_sha256": population["request_manifest_sha256"],
            "run_id": REFERENCE_RUN_ID,
            "score_rows": len(reference_rows),
            "scores_sha256": sha256_file(reference_dir / "scores.jsonl"),
            "scoring_signature": signature,
            "split": "dev",
        },
    )

    assignment_histories = histories if condition_id == "full" else [[], []]
    assignment_path = tmp_path / f"{condition_id}.jsonl"
    _write_jsonl(
        assignment_path,
        [
            {
                "request_id": record["request_id"],
                "condition_id": condition_id,
                "history": history,
            }
            for record, history in zip(records, assignment_histories)
        ],
    )
    assignment_manifest_path = tmp_path / "assignment_manifest.json"
    assignment_manifest = {
        "schema_version": 1,
        "probe_id": "m1_history_interventions_v1",
        "population_role": "train_only_internal_dev",
        "source_records_sha256": population["records_dev_sha256"],
        "qrels_read": False,
        "model_scores_read": False,
        "forbidden_field_count": 0,
        "candidate_leakage_count": 0,
        "causality_violation_count": 0,
        "query_candidate_immutability": {
            "assignment_payload_fields": ["request_id", "condition_id", "history"],
            "query_changed_rows": 0,
            "candidate_changed_rows": 0,
        },
        "conditions": {
            condition_id: {
                "path": str(assignment_path),
                "sha256": sha256_file(assignment_path),
                "request_count": len(records),
            }
        },
    }
    _write_json(assignment_manifest_path, assignment_manifest)
    return {
        "assignment": assignment_path,
        "assignment_manifest": assignment_manifest_path,
        "checkpoint_root": checkpoint_root,
        "condition_id": condition_id,
        "config": config,
        "config_path": config_path,
        "records": records,
        "runs_dir": runs_dir,
        "standardized": standardized,
    }


@contextmanager
def _mock_frozen_runtime(fixture, *, score_fn=_fake_score):
    with patch.object(
        scorer, "load_v12_ranker_config", return_value=fixture["config"]
    ), patch.object(
        scorer, "_validate_scoring_checkpoint_provenance", return_value=None
    ), patch.object(
        scorer,
        "_checkpoint_identity",
        return_value=(CHECKPOINT_ID, CHECKPOINT_FILES),
    ), patch.object(
        scorer,
        "_frozen_scorer_implementation_identity",
        return_value=FROZEN_IDENTITY,
    ), patch.object(
        scorer, "_load_model_and_tokenizer", return_value=(object(), _Model())
    ), patch.object(
        scorer, "_score_yes_no_request", side_effect=score_fn
    ), patch.object(
        scorer,
        "_runtime_metadata",
        return_value={
            "package_versions": {"torch": "fixture", "transformers": "fixture"},
            "python_executable": sys.executable,
            "python_version": "fixture",
        },
    ):
        yield


def _run(fixture, run_id, **kwargs):
    return scorer.write_mechanism_intervention_scores(
        fixture["standardized"],
        fixture["config_path"],
        fixture["checkpoint_root"],
        fixture["assignment"],
        fixture["assignment_manifest"],
        fixture["condition_id"],
        REFERENCE_RUN_ID,
        "dev",
        run_id,
        device="cpu",
        runs_dir=fixture["runs_dir"],
        command=["fixture"],
        **kwargs,
    )


def test_complete_finite_scores_and_malformed_qrels_is_never_opened(tmp_path):
    fixture = _make_fixture(tmp_path)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if path.name.startswith("qrels_"):
            raise AssertionError(f"scorer attempted to open qrels: {path}")
        return original_open(path, *args, **kwargs)

    run_id = "20260717_kuaisearch_mech_fixture_complete"
    with _mock_frozen_runtime(fixture), patch.object(Path, "open", guarded_open):
        metadata = _run(fixture, run_id)
    rows = list(
        scorer.iter_jsonl(fixture["runs_dir"] / run_id / "scores.jsonl")
    )
    assert metadata["status"] == "completed"
    assert metadata["qrels_read"] is False
    assert metadata["coverage_complete"] is True
    assert metadata["result_eligible"] is True
    assert len(rows) == 4
    assert all(math.isfinite(float(row["score"])) for row in rows)
    assert metadata["base_scoring_signature"] == metadata["scoring_signature"]
    reference_metadata = json.loads(
        (
            fixture["runs_dir"] / REFERENCE_RUN_ID / "metadata.json"
        ).read_text(encoding="utf-8")
    )
    assert metadata["base_scoring_signature"] == reference_metadata[
        "scoring_signature"
    ]
    assert "mechanism_probe_manifest_sha256" not in metadata[
        "base_scoring_signature"
    ]
    assert metadata["mechanism_probe_manifest"] == {
        "expected_sha256": scorer.MECHANISM_PROBE_MANIFEST_SHA256,
        "path": "experiments/motivation/probe_manifest.yaml",
        "sha256": scorer.MECHANISM_PROBE_MANIFEST_SHA256,
        "verified": True,
    }
    assert metadata["intervention"]["candidate_or_query_modified"] is False
    assert metadata["mechanism_scorer_sha256"] == sha256_file(
        Path(scorer.__file__)
    )


@pytest.mark.parametrize("failure", ["coverage", "finite"])
def test_candidate_coverage_and_finite_scores_are_mandatory(tmp_path, failure):
    fixture = _make_fixture(tmp_path)

    def invalid_score(model, tokenizer, record, history, config, *, device, batch_size):
        del model, tokenizer, history, config, device, batch_size
        first = str(record.candidates[0]["item_id"])
        if failure == "coverage":
            return {first: 1.0}, False
        return {
            str(candidate["item_id"]): (float("nan") if index == 0 else 1.0)
            for index, candidate in enumerate(record.candidates)
        }, False

    with _mock_frozen_runtime(fixture, score_fn=invalid_score):
        with pytest.raises((ValueError, FloatingPointError)):
            _run(fixture, f"20260717_kuaisearch_mech_fixture_{failure}")


def test_request_atomic_resume_records_lineage_and_validates_partial(tmp_path):
    fixture = _make_fixture(tmp_path)
    run_id = "20260717_kuaisearch_mech_fixture_resume"
    ticks = iter((0.0, 0.0, 2.0, 2.0))
    with _mock_frozen_runtime(fixture), patch.object(
        scorer, "_monotonic", side_effect=lambda: next(ticks)
    ):
        first = _run(fixture, run_id, max_wall_seconds=1.0)
    assert first["status"] == "wall_time_exhausted"
    assert first["completed_requests"] == 1
    assert not (fixture["runs_dir"] / run_id / "scores.jsonl").exists()

    with _mock_frozen_runtime(fixture):
        completed = _run(fixture, run_id, resume=True)
    assert completed["status"] == "completed"
    assert completed["request_count"] == 2
    assert len(completed["resume_lineage"]) == 1
    assert completed["resume_lineage"][0]["completed_requests"] == 1
    progress = json.loads(
        (fixture["runs_dir"] / run_id / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["resume_count"] == 1
    assert progress["status"] == "completed"


def test_resume_rejects_assignment_and_manifest_hash_drift(tmp_path):
    fixture = _make_fixture(tmp_path)
    run_id = "20260717_kuaisearch_mech_fixture_assignment_drift"
    ticks = iter((0.0, 0.0, 2.0, 2.0))
    with _mock_frozen_runtime(fixture), patch.object(
        scorer, "_monotonic", side_effect=lambda: next(ticks)
    ):
        _run(fixture, run_id, max_wall_seconds=1.0)
    fixture["assignment"].write_text(
        fixture["assignment"].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with _mock_frozen_runtime(fixture), pytest.raises(ValueError, match="manifest hash"):
        _run(fixture, run_id, resume=True)

    fixture = _make_fixture(tmp_path / "manifest_case")
    run_id = "20260717_kuaisearch_mech_fixture_manifest_drift"
    ticks = iter((0.0, 0.0, 2.0, 2.0))
    with _mock_frozen_runtime(fixture), patch.object(
        scorer, "_monotonic", side_effect=lambda: next(ticks)
    ):
        _run(fixture, run_id, max_wall_seconds=1.0)
    manifest = json.loads(fixture["assignment_manifest"].read_text(encoding="utf-8"))
    manifest["note"] = "post-start drift"
    _write_json(fixture["assignment_manifest"], manifest)
    with _mock_frozen_runtime(fixture), pytest.raises(ValueError, match="run contract"):
        _run(fixture, run_id, resume=True)


def test_resume_rejects_tampered_completed_request_block(tmp_path):
    fixture = _make_fixture(tmp_path)
    run_id = "20260717_kuaisearch_mech_fixture_partial_tamper"
    ticks = iter((0.0, 0.0, 2.0, 2.0))
    with _mock_frozen_runtime(fixture), patch.object(
        scorer, "_monotonic", side_effect=lambda: next(ticks)
    ):
        _run(fixture, run_id, max_wall_seconds=1.0)
    partial_path = fixture["runs_dir"] / run_id / "scores.partial.jsonl"
    block = json.loads(partial_path.read_text(encoding="utf-8"))
    block["rows"][0]["score"] += 7.0
    partial_path.write_text(
        json.dumps(block, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    with _mock_frozen_runtime(fixture), pytest.raises(ValueError, match="hash mismatch"):
        _run(fixture, run_id, resume=True)


def test_full_condition_is_a_byte_exact_frozen_scorer_canary(tmp_path):
    fixture = _make_fixture(tmp_path, condition_id="full")
    run_id = "20260717_kuaisearch_mech_fixture_full_canary"
    with _mock_frozen_runtime(fixture):
        metadata = _run(fixture, run_id)
    assert metadata["full_canary_passed"] is True
    assert metadata["scores_sha256"] == metadata["reference_scores_sha256"]
    assert metadata["result_eligible"] is True


def test_capped_scoring_is_permanently_marked_smoke_non_result(tmp_path):
    fixture = _make_fixture(tmp_path)
    run_id = "20260717_kuaisearch_mech_fixture_smoke"
    with _mock_frozen_runtime(fixture):
        metadata = _run(fixture, run_id, max_score_requests=1)
    assert metadata["status"] == "completed"
    assert metadata["evidence_mode"] == "smoke_non_result"
    assert metadata["coverage_complete"] is False
    assert metadata["result_eligible"] is False
    assert metadata["non_result_reason"] == "max_score_requests_cap"


@pytest.mark.parametrize("split", ["confirmation", "test", "train"])
def test_non_dev_splits_are_rejected_before_any_data_access(tmp_path, split):
    with pytest.raises(ValueError, match="split=dev"):
        scorer.write_mechanism_intervention_scores(
            tmp_path,
            tmp_path / "config",
            tmp_path / "checkpoint",
            tmp_path / "assignment",
            tmp_path / "manifest",
            "probe",
            REFERENCE_RUN_ID,
            split,
            "20260717_kuaisearch_mech_fixture_rejected_split",
            device="cpu",
            runs_dir=tmp_path / "runs",
        )


def test_probe_manifest_path_and_hash_are_fail_closed(tmp_path):
    copied = tmp_path / "probe_manifest.yaml"
    source = Path(__file__).resolve().parents[1] / scorer.MECHANISM_PROBE_MANIFEST_PATH
    copied.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="path mismatch"):
        scorer._load_mechanism_probe_manifest(copied)

    real_sha256_file = scorer.sha256_file

    def drifted_hash(path, *args, **kwargs):
        if Path(path).resolve() == source.resolve():
            return "0" * 64
        return real_sha256_file(path, *args, **kwargs)

    with patch.object(scorer, "sha256_file", side_effect=drifted_hash), pytest.raises(
        ValueError, match="hash mismatch"
    ):
        scorer._load_mechanism_probe_manifest(source)
