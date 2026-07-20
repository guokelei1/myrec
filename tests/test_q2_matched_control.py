from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.motivation_v12_contracts import ModelRecord, TrainingGroup
from myrec.mechanism import gradient_diagnostic
from myrec.mechanism import matched_control_evaluator as matched_evaluator
from myrec.mechanism.matched_control_evaluator import (
    _validate_score_metadata,
    validate_cross_checkpoint_fixed_recipe,
)
from myrec.mechanism.matched_control_scorer import (
    _canonical_json as scorer_canonical_json,
    _prepare_score_run,
    _validated_request_rows,
    validate_matched_training_checkpoint_metadata,
)
from myrec.mechanism.matched_training_control import (
    BALANCED_QUOTAS,
    GRADIENT_ACCUMULATION_STEPS,
    METHOD_ID,
    OPTIMIZER_STEPS,
    ROLE,
    TRAIN_GROUPS,
    _canonical_sha256,
    _assert_run_output_separation,
    _read_json,
    _run_mock_schedule,
    _truncate_learning_curve,
    classify_train_surface,
    run_q2_matched_training_control,
    select_matched_training_groups,
)
from myrec.utils.hashing import sha256_file


ROOT = Path(__file__).resolve().parents[1]


def _materialize_fake_shared_pair(
    kwargs: dict[str, Any],
    *,
    balanced_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    analysis_run_id = str(kwargs["analysis_run_id"])
    balanced = "surface_balanced" in analysis_run_id
    response = 0.3 if balanced else 0.1
    margin = 1.5 if balanced else 1.0
    metrics = {
        "analysis_run_id": analysis_run_id,
        "analysis_type": "motivation_mechanism_paired_probe",
        "candidate_manifest_sha256": "b" * 64,
        "control_run_id": str(kwargs["control_run_id"]),
        "dataset_id": "KuaiSearch",
        "dataset_version": "mechanism-test",
        "label_mode": "graded",
        "mean_target_margin_change": margin,
        "mean_treatment_minus_control_ndcg@10": response,
        "method_id": METHOD_ID,
        "num_requests": 8000,
        "qrels_sha256": "a" * 64,
        "request_manifest_sha256": "c" * 64,
        "split": "dev",
        "surfaces": {
            "all": {
                "mean_target_margin_change": margin,
                "mean_treatment_minus_control_ndcg@10": response,
                "num_requests": 8000,
            }
        },
        "treatment_run_id": str(kwargs["treatment_run_id"]),
    }
    if balanced and balanced_overrides:
        metrics.update(balanced_overrides)
    pair_dir = Path(kwargs["runs_dir"]) / analysis_run_id
    pair_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = pair_dir / "metrics.json"
    metadata_path = pair_dir / "metadata.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metadata_path.write_text(
        json.dumps(
            {
                "analysis_run_id": analysis_run_id,
                "analysis_type": "motivation_mechanism_paired_probe",
                "invariants": {"method_id": METHOD_ID},
                "label_mode": "graded",
                "metrics_path": str(metrics_path),
                "qrels_read": True,
                "qrels_sha256": metrics["qrels_sha256"],
                "split": "dev",
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return metrics


def _load_matched_evaluator_cli() -> ModuleType:
    path = ROOT / "scripts/evaluate_q2_matched_control.py"
    spec = importlib.util.spec_from_file_location(
        "test_evaluate_q2_matched_control_command_lineage", path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"could not load CLI: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _group(request_id: str, surface: str) -> tuple[TrainingGroup, dict[str, float]]:
    positive_id = f"{request_id}-positive"
    negative_id = f"{request_id}-negative"
    overlap_id = f"{request_id}-overlap"
    candidates = (
        {"item_id": positive_id, "title": "positive"},
        {"item_id": negative_id, "title": "negative"},
    )
    if surface == "recurrence":
        history = ({"item_id": positive_id, "title": "seen positive"},)
    elif surface == "other_overlap":
        history = ({"item_id": negative_id, "title": "seen negative"},)
    elif surface == "strict_transfer":
        history = ({"item_id": overlap_id, "title": "different item"},)
    elif surface == "none":
        history = ()
    else:
        raise AssertionError(surface)
    record = ModelRecord(
        request_id=request_id,
        query="query",
        history=history,
        candidates=candidates,
    )
    return (
        TrainingGroup(record=record, candidates=candidates, gains=(2.0, 0.0)),
        {positive_id: 2.0},
    )


def _surface_fixture() -> tuple[list[TrainingGroup], dict[str, dict[str, float]]]:
    groups = []
    gains = {}
    for surface in ("recurrence", "strict_transfer", "other_overlap"):
        for index in range(5):
            group, group_gains = _group(f"{surface}-{index}", surface)
            groups.append(group)
            gains[group.record.request_id] = group_gains
    group, group_gains = _group("excluded", "none")
    groups.append(group)
    gains[group.record.request_id] = group_gains
    return groups, gains


def _formal_metadata(condition: str) -> dict:
    config = _admission_config()
    recipe = {
        "backbone_initialization": {
            "artifact_manifest_sha256": "base-artifacts",
            "base_model_path": "models/qwen",
            "weights_sha256": "base-weights",
            "tokenizer_sha256": "tokenizer",
            "seed": 20260714,
        },
        "anchor": METHOD_ID,
        "batch_requests": 1,
        "config_sha256": "config-sha",
        "gradient_accumulation_steps": 16,
        "group_exposures": 4096,
        "history_budget": 6,
        "implementation_digest": "implementation-digest",
        "input_fields_used": [
            "query",
            "history.title",
            "history.brand",
            "history.cat",
            "history.event",
            "history.query",
            "candidates.title",
            "candidates.brand",
            "candidates.cat",
        ],
        "max_grad_norm": 1.0,
        "max_length": 2048,
        "objective": {
            "name": "ranknet_listnet",
            "pairwise_loss_weight": 0.5,
            "listwise_loss_weight": 0.5,
        },
        "optimizer": {
            "class": "AdamW",
            "learning_rate": 1e-5,
            "weight_decay": 0.01,
        },
        "optimizer_steps": 256,
        "probe_manifest_sha256": "probe-sha",
        "protocol_sha256": "protocol-sha",
        "role": ROLE,
        "runtime_contract": {"torch": "x"},
        "scheduler": {"total_steps": 256},
        "surface_classifier": {"function": "shared"},
    }
    selection = "original-selection" if condition == "original_mixture" else "balanced-selection"
    counts = (
        {"recurrence": 697, "strict_transfer": 2658, "other_overlap": 741}
        if condition == "original_mixture"
        else dict(BALANCED_QUOTAS)
    )
    run_contract = {"condition": condition, "config_sha256": config["_config_sha256"]}
    return {
        "checkpoint_id": f"checkpoint-{condition}",
        "base_initialization_seed": 20260714,
        "condition": condition,
        "config_sha256": "config-sha",
        "evidence_mode": "mechanism_diagnostic",
        "formal_checkpoint": True,
        "matched_recipe": recipe,
        "matched_recipe_sha256": _canonical_sha256(recipe),
        "matched_training_control_implementation_identity": {
            "digest": "implementation-digest"
        },
        "method_id": METHOD_ID,
        "progress": {
            "group_exposures": 4096,
            "micro_steps": 4096,
            "optimizer_steps": 256,
        },
        "probe_manifest": {
            "sha256": recipe.get("probe_manifest_sha256"),
            "verified": True,
        },
        "qrels_access": {"qrels_dev_read": False},
        "result_eligible": True,
        "role": ROLE,
        "run_contract": run_contract,
        "run_contract_sha256": _canonical_sha256(run_contract),
        "sampling": {
            "condition": condition,
            "selection_sha256": selection,
            "selected_surface_counts": counts,
        },
        "selection_sha256": selection,
        "status": "completed",
    }


def _admission_config() -> dict:
    return {
        "_config_sha256": "config-sha",
        "method": {
            "listwise_loss_weight": 0.5,
            "pairwise_loss_weight": 0.5,
        },
        "model": {
            "base_artifact_manifest_sha256": "base-artifacts",
            "base_model_path": "models/qwen",
            "base_weights_sha256": "base-weights",
            "tokenizer_sha256": "tokenizer",
        },
        "protocol": {"sha256": "protocol-sha"},
        "training": {
            "history_budget": 6,
            "learning_rate": 1e-5,
            "max_grad_norm": 1.0,
            "max_length": 2048,
            "weight_decay": 0.01,
        },
    }


def test_surface_classifier_is_exact_shared_function_and_hand_boundaries() -> None:
    assert classify_train_surface is gradient_diagnostic.classify_train_surface
    for expected in ("recurrence", "strict_transfer", "other_overlap"):
        group, gains = _group(expected, expected)
        assert classify_train_surface(group, gains) == expected
    group, gains = _group("none", "none")
    assert classify_train_surface(group, gains) is None


def test_balanced_selection_has_fixed_quota_no_duplicates_and_stable_order() -> None:
    groups, gains = _surface_fixture()
    quotas = {surface: 2 for surface in gradient_diagnostic.SURFACES}
    selected, manifest = select_matched_training_groups(
        groups,
        gains,
        "surface_balanced",
        train_groups=6,
        balanced_quotas=quotas,
    )
    reversed_selected, reversed_manifest = select_matched_training_groups(
        list(reversed(groups)),
        gains,
        "surface_balanced",
        train_groups=6,
        balanced_quotas=quotas,
    )
    request_ids = [group.record.request_id for group in selected]
    assert len(request_ids) == len(set(request_ids)) == 6
    assert manifest["selected_surface_counts"] == {
        "recurrence": 2,
        "strict_transfer": 2,
        "other_overlap": 2,
    }
    assert manifest["excluded_without_registered_surface"] == 1
    assert manifest["finalized_before_model_load_and_loss"] is True
    assert request_ids == [group.record.request_id for group in reversed_selected]
    assert manifest["exposure_schedule_sha256"] == reversed_manifest[
        "exposure_schedule_sha256"
    ]


def test_original_mixture_is_stable_hash_union_sample_without_stratified_quota() -> None:
    groups, gains = _surface_fixture()
    selected, manifest = select_matched_training_groups(
        groups, gains, "original_mixture", train_groups=7
    )
    selected_again, manifest_again = select_matched_training_groups(
        groups, gains, "original_mixture", train_groups=7
    )
    assert [group.record.request_id for group in selected] == [
        group.record.request_id for group in selected_again
    ]
    assert sum(manifest["selected_surface_counts"].values()) == 7
    assert manifest["request_ids_sha256"] == manifest_again["request_ids_sha256"]
    assert "surface_union" in manifest["selection_algorithm"]


def test_registered_schedule_is_exact_4096_exposures_256_steps() -> None:
    assert BALANCED_QUOTAS == {
        "recurrence": 1366,
        "strict_transfer": 1365,
        "other_overlap": 1365,
    }
    assert sum(BALANCED_QUOTAS.values()) == TRAIN_GROUPS == 4096
    assert GRADIENT_ACCUMULATION_STEPS == 16
    assert OPTIMIZER_STEPS == 256
    assert TRAIN_GROUPS == GRADIENT_ACCUMULATION_STEPS * OPTIMIZER_STEPS


def test_run_and_checkpoint_directories_must_be_independent(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run"
    _assert_run_output_separation(run_dir, tmp_path / "checkpoints" / "condition")
    with pytest.raises(ValueError, match="must be independent"):
        _assert_run_output_separation(run_dir, run_dir)
    with pytest.raises(ValueError, match="must be independent"):
        _assert_run_output_separation(run_dir, run_dir / "checkpoint")


def test_formal_training_rejects_cpu_before_touching_data(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires an explicit cuda"):
        run_q2_matched_training_control(
            tmp_path / "missing-data",
            tmp_path / "missing-config.yaml",
            "original_mixture",
            "20260717_kuaisearch_q2_matched_cpu_forbidden",
            tmp_path / "checkpoint",
            device="cpu",
            runs_dir=tmp_path / "runs",
        )


def test_formal_resume_truncates_uncheckpointed_learning_curve(tmp_path: Path) -> None:
    path = tmp_path / "learning_curve.jsonl"
    path.write_text(
        "".join(
            json.dumps({"optimizer_step": step, "loss": float(step)}) + "\n"
            for step in range(1, 6)
        ),
        encoding="utf-8",
    )
    _truncate_learning_curve(path, checkpoint_optimizer_steps=3)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["optimizer_step"] for row in rows] == [1, 2, 3]


def test_cpu_mock_complete_schedule_is_permanently_non_result(tmp_path: Path) -> None:
    template, _ = _group("template", "strict_transfer")
    groups = [
        TrainingGroup(
            record=ModelRecord(
                request_id=f"r{index:04d}",
                query=template.record.query,
                history=template.record.history,
                candidates=template.record.candidates,
            ),
            candidates=template.candidates,
            gains=template.gains,
        )
        for index in range(TRAIN_GROUPS)
    ]
    run_dir = tmp_path / "runs" / "mock"
    output_dir = tmp_path / "checkpoints" / "mock"
    run_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    metadata = {
        "condition": "original_mixture",
        "evidence_mode": "smoke_non_result",
        "formal_checkpoint": False,
        "result_eligible": False,
        "run_contract_sha256": "contract",
        "selection_sha256": "selection",
    }
    progress = {
        "group_exposures": 0,
        "micro_steps": 0,
        "optimizer_steps": 0,
        "status": "selection_finalized",
    }
    paused = _run_mock_schedule(
        groups,
        run_dir=run_dir,
        output_model_dir=output_dir,
        metadata=metadata,
        progress=progress,
        started=0.0,
        max_wall_seconds=math.inf,
        stop_after_optimizer_steps=2,
    )
    assert paused["status"] == "mock_pause"
    assert paused["progress"]["group_exposures"] == 32
    resumed = _run_mock_schedule(
        groups,
        run_dir=run_dir,
        output_model_dir=output_dir,
        metadata=paused,
        progress=_read_json(run_dir / "progress.json"),
        started=0.0,
        max_wall_seconds=math.inf,
        stop_after_optimizer_steps=None,
    )
    assert resumed["status"] == "completed"
    assert resumed["progress"]["group_exposures"] == 4096
    assert resumed["progress"]["optimizer_steps"] == 256
    assert resumed["evidence_mode"] == "smoke_non_result"
    assert resumed["formal_checkpoint"] is False
    assert resumed["result_eligible"] is False
    state = _read_json(output_dir / "checkpoint_latest" / "mock_trainer_state.json")
    assert state["formal_checkpoint"] is False
    assert state["progress"]["optimizer_steps"] == 256


def test_scorer_checkpoint_admission_rejects_smoke_and_underexposure() -> None:
    config = _admission_config()
    valid = _formal_metadata("original_mixture")
    admitted = validate_matched_training_checkpoint_metadata(
        valid,
        sampling_condition="original_mixture",
        config=config,
    )
    assert admitted["status"] == "admitted"

    smoke = {**valid, "evidence_mode": "smoke_non_result", "formal_checkpoint": False}
    with pytest.raises(ValueError, match="evidence_mode"):
        validate_matched_training_checkpoint_metadata(
            smoke,
            sampling_condition="original_mixture",
            config=config,
        )
    underexposed = {
        **valid,
        "progress": {
            "group_exposures": 4080,
            "micro_steps": 4080,
            "optimizer_steps": 255,
        },
    }
    with pytest.raises(ValueError, match="4096/256"):
        validate_matched_training_checkpoint_metadata(
            underexposed,
            sampling_condition="original_mixture",
            config=config,
        )


def test_hand_computed_score_rows_require_exact_finite_candidate_coverage() -> None:
    record = ModelRecord(
        request_id="r1",
        query="q",
        history=(),
        candidates=({"item_id": "a"}, {"item_id": "b"}, {"item_id": "c"}),
    )
    rows, score_range = _validated_request_rows(
        record, {"a": -1.0, "b": 2.5, "c": 0.5}
    )
    assert [row["candidate_item_id"] for row in rows] == ["a", "b", "c"]
    assert score_range == pytest.approx(3.5)
    with pytest.raises(ValueError, match="coverage"):
        _validated_request_rows(record, {"a": 1.0, "b": 2.0})
    with pytest.raises(ValueError, match="non-finite"):
        _validated_request_rows(record, {"a": 1.0, "b": float("nan"), "c": 0.0})


def test_cross_checkpoint_admission_allows_sampling_only_and_catches_lr_drift() -> None:
    original = _formal_metadata("original_mixture")
    balanced = _formal_metadata("surface_balanced")
    admission = validate_cross_checkpoint_fixed_recipe(original, balanced)
    assert admission["status"] == "passed"
    assert admission["matched_recipe_sha256"] == original["matched_recipe_sha256"]

    drifted = json.loads(json.dumps(balanced))
    drifted["matched_recipe"]["optimizer"]["learning_rate"] = 2e-5
    drifted["matched_recipe_sha256"] = _canonical_sha256(drifted["matched_recipe"])
    with pytest.raises(ValueError, match="base/config/optimizer"):
        validate_cross_checkpoint_fixed_recipe(original, drifted)


def test_score_metadata_boundary_requires_8000_160753_and_full_null_identity() -> None:
    training = _formal_metadata("surface_balanced")
    metadata = {
        "base_scoring_signature": {
            "checkpoint_id": training["checkpoint_id"],
            "config_sha256": "config-sha",
            "matched_recipe_sha256": training["matched_recipe_sha256"],
            "method_id": METHOD_ID,
        },
        "checkpoint_id": training["checkpoint_id"],
        "condition_id": "surface_balanced__full",
        "config_sha256": "config-sha",
        "evidence_mode": "mechanism_diagnostic",
        "history_condition": "full",
        "matched_control_scorer_implementation_identity": {
            "digest": "scorer-digest"
        },
        "matched_training_control": {
            "matched_recipe": training["matched_recipe"],
            "matched_recipe_sha256": training["matched_recipe_sha256"],
            "role": ROLE,
            "sampling_condition": "surface_balanced",
        },
        "method_id": METHOD_ID,
        "qrels_read": False,
        "request_count": 8000,
        "result_eligible": True,
        "run_id": "score-run",
        "sampling_condition": "surface_balanced",
        "score_rows": 160753,
        "split": "dev",
        "status": "completed",
    }
    _validate_score_metadata(
        metadata,
        run_id="score-run",
        sampling_condition="surface_balanced",
        history_condition="full",
    )
    with pytest.raises(ValueError, match="score_rows"):
        _validate_score_metadata(
            {**metadata, "score_rows": 160752},
            run_id="score-run",
            sampling_condition="surface_balanced",
            history_condition="full",
        )


def test_score_resume_recovers_fsynced_block_when_progress_lagged(
    tmp_path: Path,
) -> None:
    records = [
        ModelRecord(
            request_id=request_id,
            query="q",
            history=(),
            candidates=({"item_id": f"{request_id}-a"}, {"item_id": f"{request_id}-b"}),
        )
        for request_id in ("r1", "r2")
    ]
    run_dir = tmp_path / "score"
    metadata, progress, _ = _prepare_score_run(
        run_dir,
        records=records,
        base_metadata={
            "condition_id": "original_mixture__full",
            "run_contract_sha256": "contract",
            "status": "initialized",
        },
        run_contract_sha256="contract",
        condition_id="original_mixture__full",
        resume=False,
    )
    del metadata, progress
    rows, _ = _validated_request_rows(
        records[0], {"r1-a": 2.0, "r1-b": -1.0}
    )
    block = {
        "condition_id": "original_mixture__full",
        "ordinal": 0,
        "prompt_at_max_boundary": False,
        "request_id": "r1",
        "rows": rows,
        "rows_sha256": _canonical_sha256(rows),
    }
    with (run_dir / "request_blocks.partial.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(scorer_canonical_json(block) + "\n")
    recovered_metadata, recovered_progress, ranges = _prepare_score_run(
        run_dir,
        records=records,
        base_metadata={},
        run_contract_sha256="contract",
        condition_id="original_mixture__full",
        resume=True,
    )
    assert recovered_progress["completed_requests"] == 1
    assert recovered_progress["completed_score_rows"] == 2
    assert recovered_metadata["partial_progress_recovery"] == {
        "durable_completed_requests": 1,
        "persisted_completed_requests": 0,
        "recovered_at": recovered_metadata["partial_progress_recovery"][
            "recovered_at"
        ],
    }
    assert ranges == [3.0]


def test_evaluator_persists_four_bundle_admission_before_shared_qrels_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "runs"
    ledger = tmp_path / "reports" / "dev_eval_log.jsonl"
    admission = {"qrels_read": False, "fixed_recipe_admission": {"status": "passed"}}
    monkeypatch.setattr(
        matched_evaluator,
        "admit_q2_matched_control_score_runs",
        lambda **_: admission,
    )
    calls = []

    def fake_shared_evaluator(**kwargs):
        audit_path = (
            runs_dir
            / "analysis"
            / "pre_qrels_cross_checkpoint_admission.json"
        )
        assert audit_path.is_file()
        assert json.loads(audit_path.read_text())["qrels_read"] is False
        calls.append(kwargs["analysis_run_id"])
        return _materialize_fake_shared_pair(kwargs)

    monkeypatch.setattr(
        matched_evaluator, "evaluate_mechanism_probe", fake_shared_evaluator
    )
    monkeypatch.setattr(matched_evaluator, "_git_revision", lambda: "test-revision")
    real_append = matched_evaluator._append_cross_checkpoint_ledger

    def observing_append(*args: object, **kwargs: object) -> None:
        assert not (runs_dir / "analysis" / "metadata.json").exists()
        real_append(*args, **kwargs)

    monkeypatch.setattr(
        matched_evaluator, "_append_cross_checkpoint_ledger", observing_append
    )
    command = [
        "scripts/evaluate_q2_matched_control.py",
        "--analysis-run-id",
        "analysis",
    ]
    metrics = matched_evaluator.evaluate_q2_matched_control(
        analysis_run_id="analysis",
        original_full_run_id="original-full",
        original_null_run_id="original-null",
        balanced_full_run_id="balanced-full",
        balanced_null_run_id="balanced-null",
        standardized_dir=tmp_path / "standardized",
        runs_dir=runs_dir,
        dev_eval_log_path=ledger,
        command=command,
    )
    assert len(calls) == 2
    assert metrics["balanced_minus_original_history_response_ndcg@10"] == pytest.approx(
        0.2
    )
    assert metrics["balanced_minus_original_target_margin_change"] == pytest.approx(
        0.5
    )
    assert metrics["command"] == command
    assert metrics["code_revision"] == "test-revision"
    assert metrics["status"] == "completed"
    assert metrics["qrels_read"] is True
    assert metrics["qrels_sha256"] == "a" * 64
    assert metrics["request_count"] == 8000
    assert metrics["score_rows_per_run"] == 160753
    assert metrics["method_id"] == METHOD_ID
    assert metrics["split"] == "dev"
    assert metrics["label_mode"] == "graded"
    assert metrics["pre_qrels_admission_sha256"] == sha256_file(
        runs_dir / "analysis" / "pre_qrels_cross_checkpoint_admission.json"
    )
    identity = metrics["matched_control_evaluator_implementation_identity"]
    assert {row["path"] for row in identity["files"]} == {
        "scripts/evaluate_q2_matched_control.py",
        "src/myrec/mechanism/evaluator.py",
        "src/myrec/mechanism/matched_control_evaluator.py",
    }
    for row in identity["files"]:
        assert row["sha256"] == sha256_file(ROOT / row["path"])
    assert identity["digest"] == _canonical_sha256(identity["files"])
    assert set(metrics["pair_analysis_artifacts"]) == {
        "original_mixture",
        "surface_balanced",
    }
    for lineage in metrics["pair_analysis_artifacts"].values():
        assert lineage["metrics_sha256"] == sha256_file(lineage["metrics_path"])
        assert lineage["metadata_sha256"] == sha256_file(lineage["metadata_path"])

    persisted_metrics = json.loads(
        (runs_dir / "analysis" / "metrics.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (runs_dir / "analysis" / "metadata.json").read_text(encoding="utf-8")
    )
    assert persisted_metrics == metrics
    assert metadata["command"] == command
    assert metadata["code_revision"] == "test-revision"
    assert metadata["matched_control_evaluator_implementation_identity"] == identity
    assert metadata["metrics_sha256"] == sha256_file(
        runs_dir / "analysis" / "metrics.json"
    )
    for key in (
        "label_mode",
        "method_id",
        "pair_analysis_artifacts",
        "pre_qrels_admission_path",
        "pre_qrels_admission_sha256",
        "qrels_read",
        "qrels_sha256",
        "request_count",
        "score_rows_per_run",
        "split",
        "status",
    ):
        assert metadata[key] == metrics[key]

    ledger_row = json.loads(ledger.read_text(encoding="utf-8"))
    assert ledger_row["run_id"] == "analysis"
    assert ledger_row["command"] == command
    assert ledger_row["code_revision"] == "test-revision"
    assert ledger_row["matched_control_evaluator_implementation_digest"] == identity[
        "digest"
    ]
    assert ledger_row["metrics_sha256"] == metadata["metrics_sha256"]
    assert ledger_row["method_id"] == "shared_mechanism_evaluator"
    assert ledger_row["subject_method_id"] == METHOD_ID
    for key in (
        "label_mode",
        "pair_analysis_artifacts",
        "pre_qrels_admission_path",
        "pre_qrels_admission_sha256",
        "qrels_read",
        "qrels_sha256",
        "request_count",
        "score_rows_per_run",
        "split",
        "status",
    ):
        assert ledger_row[key] == metrics[key]


def test_matched_evaluator_cli_passes_exact_sys_argv(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _load_matched_evaluator_cli()
    captured: dict[str, object] = {}

    def fake_evaluator(**kwargs: object) -> dict[str, str]:
        captured.update(kwargs)
        return {"status": "test-only"}

    argv = [
        "evaluate_q2_matched_control.py",
        "--analysis-run-id",
        "analysis",
        "--original-full-run-id",
        "original-full",
        "--original-null-run-id",
        "original-null",
        "--balanced-full-run-id",
        "balanced-full",
        "--balanced-null-run-id",
        "balanced-null",
        "--standardized-dir",
        "standardized",
    ]
    monkeypatch.setattr(module, "evaluate_q2_matched_control", fake_evaluator)
    monkeypatch.setattr(sys, "argv", argv)
    module.main()  # type: ignore[attr-defined]
    assert captured["command"] == argv
    assert captured["command"] is sys.argv
    assert "test-only" in capsys.readouterr().out


def test_matched_evaluator_rejects_explicit_empty_command() -> None:
    with pytest.raises(
        ValueError, match="matched-control evaluator command must be non-empty"
    ):
        matched_evaluator.evaluate_q2_matched_control(
            analysis_run_id="analysis",
            original_full_run_id="original-full",
            original_null_run_id="original-null",
            balanced_full_run_id="balanced-full",
            balanced_null_run_id="balanced-null",
            standardized_dir="standardized",
            command=[],
        )


@pytest.mark.parametrize(
    ("balanced_overrides", "message"),
    [
        ({"analysis_run_id": "wrong-analysis"}, "analysis_run_id"),
        ({"method_id": "wrong-method"}, "method_id"),
        ({"num_requests": 7999}, "num_requests"),
        ({"qrels_sha256": "d" * 64}, "qrels_sha256"),
        ({"split": "test"}, "split"),
    ],
)
def test_matched_evaluator_rejects_inconsistent_pair_completion_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    balanced_overrides: dict[str, Any],
    message: str,
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(
        matched_evaluator,
        "admit_q2_matched_control_score_runs",
        lambda **_: {
            "qrels_read": False,
            "fixed_recipe_admission": {"status": "passed"},
        },
    )

    def fake_shared_evaluator(**kwargs: Any) -> dict[str, Any]:
        return _materialize_fake_shared_pair(
            kwargs, balanced_overrides=balanced_overrides
        )

    monkeypatch.setattr(
        matched_evaluator, "evaluate_mechanism_probe", fake_shared_evaluator
    )
    monkeypatch.setattr(matched_evaluator, "_git_revision", lambda: "revision")
    with pytest.raises(ValueError, match=message):
        matched_evaluator.evaluate_q2_matched_control(
            analysis_run_id="analysis",
            original_full_run_id="original-full",
            original_null_run_id="original-null",
            balanced_full_run_id="balanced-full",
            balanced_null_run_id="balanced-null",
            standardized_dir=tmp_path / "standardized",
            runs_dir=runs_dir,
            dev_eval_log_path=tmp_path / "dev_eval_log.jsonl",
            command=["evaluate_q2_matched_control.py"],
        )
    assert not (runs_dir / "analysis" / "metrics.json").exists()
    assert not (runs_dir / "analysis" / "metadata.json").exists()


def test_matched_evaluator_rejects_returned_persisted_pair_metric_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(
        matched_evaluator,
        "admit_q2_matched_control_score_runs",
        lambda **_: {
            "qrels_read": False,
            "fixed_recipe_admission": {"status": "passed"},
        },
    )

    def fake_shared_evaluator(**kwargs: Any) -> dict[str, Any]:
        metrics = _materialize_fake_shared_pair(kwargs)
        if "surface_balanced" in str(kwargs["analysis_run_id"]):
            metrics["mean_target_margin_change"] = 9.0
        return metrics

    monkeypatch.setattr(
        matched_evaluator, "evaluate_mechanism_probe", fake_shared_evaluator
    )
    monkeypatch.setattr(matched_evaluator, "_git_revision", lambda: "revision")
    with pytest.raises(ValueError, match="returned/persisted metrics differ"):
        matched_evaluator.evaluate_q2_matched_control(
            analysis_run_id="analysis",
            original_full_run_id="original-full",
            original_null_run_id="original-null",
            balanced_full_run_id="balanced-full",
            balanced_null_run_id="balanced-null",
            standardized_dir=tmp_path / "standardized",
            runs_dir=runs_dir,
            dev_eval_log_path=tmp_path / "dev_eval_log.jsonl",
            command=["evaluate_q2_matched_control.py"],
        )
    assert not (runs_dir / "analysis" / "metadata.json").exists()


def test_matched_evaluator_ledger_failure_leaves_completion_marker_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(
        matched_evaluator,
        "admit_q2_matched_control_score_runs",
        lambda **_: {
            "qrels_read": False,
            "fixed_recipe_admission": {"status": "passed"},
        },
    )
    monkeypatch.setattr(
        matched_evaluator,
        "evaluate_mechanism_probe",
        lambda **kwargs: _materialize_fake_shared_pair(kwargs),
    )
    monkeypatch.setattr(matched_evaluator, "_git_revision", lambda: "revision")

    def fail_ledger(*args: object, **kwargs: object) -> None:
        assert (runs_dir / "analysis" / "metrics.json").is_file()
        assert not (runs_dir / "analysis" / "metadata.json").exists()
        raise OSError("simulated durable ledger failure")

    monkeypatch.setattr(
        matched_evaluator, "_append_cross_checkpoint_ledger", fail_ledger
    )
    with pytest.raises(OSError, match="simulated durable ledger failure"):
        matched_evaluator.evaluate_q2_matched_control(
            analysis_run_id="analysis",
            original_full_run_id="original-full",
            original_null_run_id="original-null",
            balanced_full_run_id="balanced-full",
            balanced_null_run_id="balanced-null",
            standardized_dir=tmp_path / "standardized",
            runs_dir=runs_dir,
            dev_eval_log_path=tmp_path / "dev_eval_log.jsonl",
            command=["evaluate_q2_matched_control.py"],
        )
    assert not (runs_dir / "analysis" / "metadata.json").exists()


def test_cross_checkpoint_ledger_flushes_before_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "dev_eval_log.jsonl"
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text("{}\n", encoding="utf-8")
    metrics = {
        "analysis_run_id": "analysis",
        "analysis_type": "q2_matched_training_control_cross_checkpoint",
        "balanced_minus_original_history_response_ndcg@10": 0.2,
        "balanced_minus_original_target_margin_change": 0.5,
        "code_revision": "revision",
        "command": ["evaluate_q2_matched_control.py"],
        "label_mode": "graded",
        "matched_control_evaluator_implementation_identity": {
            "digest": "e" * 64
        },
        "method_id": METHOD_ID,
        "original_mixture_analysis_run_id": "analysis_original",
        "pair_analysis_artifacts": {},
        "pre_qrels_admission_path": "pre_qrels.json",
        "pre_qrels_admission_sha256": "f" * 64,
        "qrels_read": True,
        "qrels_sha256": "a" * 64,
        "request_count": 8000,
        "score_rows_per_run": 160753,
        "split": "dev",
        "status": "completed",
        "surface_balanced_analysis_run_id": "analysis_balanced",
    }
    fsync_calls = []

    def observe_fsync(file_descriptor: int) -> None:
        assert json.loads(ledger.read_text(encoding="utf-8"))["status"] == "completed"
        fsync_calls.append(file_descriptor)

    monkeypatch.setattr(matched_evaluator.os, "fsync", observe_fsync)
    matched_evaluator._append_cross_checkpoint_ledger(
        ledger, metrics=metrics, metrics_path=metrics_path
    )
    assert len(fsync_calls) == 1


def test_scorer_source_has_no_evaluator_or_dev_qrels_file_dependency() -> None:
    source_path = (
        Path(__file__).resolve().parents[1]
        / "src/myrec/mechanism/matched_control_scorer.py"
    )
    source = source_path.read_text(encoding="utf-8")
    assert "myrec.mechanism.evaluator" not in source
    assert "qrels_dev.jsonl" not in source
