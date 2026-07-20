from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from myrec.baselines.motivation_v12_contracts import ModelRecord, TrainingGroup
from myrec.mechanism import gradient_diagnostic as diagnostic
from myrec.mechanism.gradient_diagnostic import (
    MODEL_INITIALIZATION_SEED,
    Q2_BLOCKS,
    classify_train_surface,
    deterministic_label_shuffle,
    discover_gradient_scopes,
    gradient_vector_metrics,
    mean_gradient_cosine,
    normalized_surface_update_shares,
    select_surface_training_groups,
    trainable_parameter_audit,
)


def _candidate(item_id: str) -> dict:
    return {"item_id": item_id, "title": item_id, "brand": "b", "cat": ["c"]}


def _history(item_id: str) -> dict:
    return {
        **_candidate(item_id),
        "event": "click",
        "query": "old",
        "ts": 1,
    }


def _group(request_id: str, surface: str, gains: tuple[float, ...] = (2.0, 0.0)) -> TrainingGroup:
    candidates = (_candidate(f"{request_id}-p"), _candidate(f"{request_id}-n"))
    if surface == "recurrence":
        history = (_history(f"{request_id}-p"),)
    elif surface == "other_overlap":
        history = (_history(f"{request_id}-n"),)
    elif surface == "strict_transfer":
        history = (_history(f"{request_id}-history"),)
    else:
        raise ValueError(surface)
    return TrainingGroup(
        record=ModelRecord(request_id, "query", history, candidates),
        candidates=candidates,
        gains=gains,
    )


def _positive_gains(group: TrainingGroup) -> dict[str, float]:
    return {
        str(candidate["item_id"]): gain
        for candidate, gain in zip(group.candidates, group.gains)
        if gain > 0
    }


def test_surface_classification_and_stable_exact_selection() -> None:
    groups = [
        _group(f"{surface}-{index}", surface)
        for surface in diagnostic.SURFACES
        for index in range(4)
    ]
    gains = {group.record.request_id: _positive_gains(group) for group in groups}
    for group in groups:
        assert classify_train_surface(group, gains[group.record.request_id]) in diagnostic.SURFACES

    first, first_manifest = select_surface_training_groups(
        groups,
        gains,
        requests_per_surface=2,
        selection_seed=17,
    )
    second, second_manifest = select_surface_training_groups(
        list(reversed(groups)),
        gains,
        requests_per_surface=2,
        selection_seed=17,
    )
    assert first_manifest == second_manifest
    assert {
        surface: [group.record.request_id for group in selected]
        for surface, selected in first.items()
    } == {
        surface: [group.record.request_id for group in selected]
        for surface, selected in second.items()
    }
    assert all(len(first[surface]) == 2 for surface in diagnostic.SURFACES)
    assert all(
        first_manifest["surfaces"][surface]["eligible_requests"] == 4
        for surface in diagnostic.SURFACES
    )


def test_registered_selection_fixes_exactly_96_per_surface() -> None:
    groups = [
        _group(f"registered-{surface}-{index:03d}", surface)
        for surface in diagnostic.SURFACES
        for index in range(100)
    ]
    gains = {group.record.request_id: _positive_gains(group) for group in groups}
    selected, manifest = select_surface_training_groups(groups, gains)
    assert diagnostic.REQUESTS_PER_SURFACE == 96
    assert manifest["requests_per_surface"] == 96
    assert all(len(selected[surface]) == 96 for surface in diagnostic.SURFACES)
    assert all(
        manifest["surfaces"][surface]["request_count"] == 96
        for surface in diagnostic.SURFACES
    )


def test_label_shuffle_changes_tied_zero_gains_deterministically() -> None:
    base = _group("tied-zero", "strict_transfer")
    candidates = (
        base.candidates[0],
        base.candidates[1],
        _candidate("tied-zero-n2"),
        _candidate("tied-zero-n3"),
    )
    group = TrainingGroup(
        record=ModelRecord(
            base.record.request_id,
            base.record.query,
            base.record.history,
            candidates,
        ),
        candidates=candidates,
        gains=(2.0, 0.0, 0.0, 0.0),
    )
    first, first_audit = deterministic_label_shuffle(group, seed=7)
    second, second_audit = deterministic_label_shuffle(group, seed=7)
    assert first == second
    assert first_audit == second_audit
    assert first.gains != group.gains
    assert sorted(first.gains) == sorted(group.gains)
    assert first_audit["changed_positions"] > 0
    assert first_audit["preserved_gain_multiset"] is True


def test_gradient_endpoints_are_hand_computed() -> None:
    gradients = {
        "a": torch.tensor([3.0, 4.0]),
        "b": torch.tensor([0.0, 12.0]),
    }
    metrics = gradient_vector_metrics(gradients)
    assert metrics["squared_gradient_norm"] == pytest.approx(169.0)
    assert metrics["gradient_norm"] == pytest.approx(13.0)
    assert metrics["scope_normalized_update_share"] == pytest.approx(
        {"a": 25.0 / 169.0, "b": 144.0 / 169.0}
    )
    assert mean_gradient_cosine(
        {"a": torch.tensor([1.0, 0.0])},
        {"a": torch.tensor([0.0, 2.0])},
    ) == pytest.approx(0.0)
    assert mean_gradient_cosine(
        {"a": torch.tensor([1.0, 2.0])},
        {"a": torch.tensor([2.0, 4.0])},
    ) == pytest.approx(1.0)
    assert normalized_surface_update_shares(
        {"recurrence": 1.0, "strict_transfer": 2.0, "other_overlap": 1.0}
    ) == pytest.approx(
        {"recurrence": 0.25, "strict_transfer": 0.5, "other_overlap": 0.25}
    )


class _Tokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        return {"yes": [1], "no": [2]}[text]


class _Attention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = torch.nn.Linear(1, 1, bias=False)
        self.v_proj = torch.nn.Linear(1, 1, bias=False)


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _Attention()
        self.mlp = torch.nn.Linear(1, 1, bias=False)


class _TinyQ2(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.unrelated = torch.nn.Parameter(torch.ones(1))
        self.layers = torch.nn.ModuleList([_Block() for _ in range(28)])
        self.embed_tokens = torch.nn.Embedding(4, 1)
        self.lm_head = torch.nn.Linear(1, 4, bias=False)
        self.lm_head.weight = self.embed_tokens.weight

    def get_output_embeddings(self) -> torch.nn.Module:
        return self.lm_head


def test_q2_scope_discovery_freezes_nonregistered_and_handles_tied_output() -> None:
    model = _TinyQ2()
    assert model.lm_head.weight is model.embed_tokens.weight
    scopes = discover_gradient_scopes(model, _Tokenizer(), "q2_recranker_generalqwen")
    assert len(scopes) == 11
    assert {scope.name for scope in scopes if scope.name.startswith("block_")} == {
        f"block_{block:02d}.{projection}.weight"
        for block in Q2_BLOCKS
        for projection in ("q_proj", "v_proj")
    }
    output_scope = next(scope for scope in scopes if scope.name == "lm_head.yes_no_rows")
    assert output_scope.parameter is model.embed_tokens.weight
    assert output_scope.parameter_name == "embed_tokens.weight"
    assert output_scope.row_indices == (1, 2)
    assert model.embed_tokens.weight.requires_grad
    assert not model.unrelated.requires_grad
    for index, block in enumerate(model.layers):
        expected = index in Q2_BLOCKS
        assert block.self_attn.q_proj.weight.requires_grad is expected
        assert block.self_attn.v_proj.weight.requires_grad is expected
        assert not block.mlp.weight.requires_grad
    audit = trainable_parameter_audit(model, scopes, "q2_recranker_generalqwen")
    assert audit["all_nonregistered_parameters_frozen"] is True
    assert audit["trainable_parameter_object_count"] == 11
    assert audit["trainable_parameter_numel"] == 14
    assert "embed_tokens.weight" in audit["trainable_parameter_names"]


class _LoRAProjection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, 2), requires_grad=False)
        self.lora_A = torch.nn.Parameter(torch.rand(2, 1))
        self.lora_B = torch.nn.Parameter(torch.rand(1, 2))


class _LoRAAttention(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _LoRAProjection()
        self.v_proj = _LoRAProjection()


class _TinyQ3(torch.nn.Module):
    def __init__(self, *, invalid_trainable: bool = False) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.layers[0].self_attn = _LoRAAttention()
        self.invalid = torch.nn.Parameter(
            torch.ones(1), requires_grad=invalid_trainable
        )


def test_q3_scope_is_exactly_all_trainable_lora_qv_parameters() -> None:
    model = _TinyQ3()
    scopes = discover_gradient_scopes(model, _Tokenizer(), "q3_tallrec_generalqwen")
    assert len(scopes) == 4
    assert all("lora_" in scope.name for scope in scopes)
    audit = trainable_parameter_audit(model, scopes, "q3_tallrec_generalqwen")
    assert audit["trainable_parameter_object_count"] == 4
    with pytest.raises(ValueError, match="non-LoRA-q/v"):
        discover_gradient_scopes(
            _TinyQ3(invalid_trainable=True),
            _Tokenizer(),
            "q3_tallrec_generalqwen",
        )


def test_q3_base_initialization_uses_frozen_seed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed_random_values: list[float] = []

    def fake_loader(*args: object, **kwargs: object) -> tuple[_Tokenizer, _TinyQ3]:
        del args, kwargs
        observed_random_values.append(random.random())
        return _Tokenizer(), _TinyQ3()

    monkeypatch.setattr(diagnostic, "_load_model_and_tokenizer", fake_loader)
    first_tokenizer, first_model = diagnostic._load_state_model(
        {"method_id": "q3_tallrec_generalqwen"},
        state="base_initialization",
        device="cpu",
        checkpoint_model_dir=tmp_path,
        torch_module=torch,
    )
    first = diagnostic._q3_trainable_initialization_fingerprint(
        discover_gradient_scopes(
            first_model, first_tokenizer, "q3_tallrec_generalqwen"
        )
    )
    random.random()
    torch.rand(5)
    second_tokenizer, second_model = diagnostic._load_state_model(
        {"method_id": "q3_tallrec_generalqwen"},
        state="base_initialization",
        device="cpu",
        checkpoint_model_dir=tmp_path,
        torch_module=torch,
    )
    second = diagnostic._q3_trainable_initialization_fingerprint(
        discover_gradient_scopes(
            second_model, second_tokenizer, "q3_tallrec_generalqwen"
        )
    )
    assert MODEL_INITIALIZATION_SEED == 20_260_714
    assert first == second
    assert observed_random_values[0] == observed_random_values[1]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _mock_run_fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict]:
    standardized = tmp_path / "standardized"
    standardized.mkdir()
    records = []
    qrels = []
    for surface in diagnostic.SURFACES:
        request_id = f"fixture-{surface}"
        candidates = [_candidate(f"{request_id}-{suffix}") for suffix in ("p", "n1", "n2", "n3")]
        if surface == "recurrence":
            history_id = f"{request_id}-p"
        elif surface == "other_overlap":
            history_id = f"{request_id}-n1"
        else:
            history_id = f"{request_id}-history"
        records.append(
            {
                "request_id": request_id,
                "query": "fixture query",
                "history": [_history(history_id)],
                "candidates": candidates,
            }
        )
        qrels.append(
            {
                "request_id": request_id,
                "relevance": {f"{request_id}-p": 2.0},
            }
        )
    _write_jsonl(standardized / "records_train.jsonl", records)
    _write_jsonl(standardized / "qrels_train.jsonl", qrels)
    for name in ("manifest.json", "candidate_manifest.json", "request_manifest.json"):
        (standardized / name).write_text("{}\n", encoding="utf-8")
    for forbidden in ("qrels_dev.jsonl", "qrels_confirmation.jsonl", "qrels_test.jsonl"):
        (standardized / forbidden).write_text("not-json\n", encoding="utf-8")

    population = {
        "candidate_manifest_sha256": diagnostic.sha256_file(standardized / "candidate_manifest.json"),
        "dataset_version": "mock_train_only_v1",
        "manifest_sha256": diagnostic.sha256_file(standardized / "manifest.json"),
        "qrels_train_sha256": diagnostic.sha256_file(standardized / "qrels_train.jsonl"),
        "records_train_sha256": diagnostic.sha256_file(standardized / "records_train.jsonl"),
        "request_manifest_sha256": diagnostic.sha256_file(standardized / "request_manifest.json"),
    }
    config = {
        "_config_sha256": "mock-config-sha256",
        "_protocol": {"data": {"development_population": population}},
        "method_id": "q2_recranker_generalqwen",
        "training": {
            "list_size": 4,
            "negatives_per_positive": 3,
            "seed": MODEL_INITIALIZATION_SEED,
        },
    }
    config_path = tmp_path / "q2_mock.yaml"
    config_path.write_text("method_id: q2_recranker_generalqwen\n", encoding="utf-8")
    checkpoint_root = tmp_path / "checkpoint"
    (checkpoint_root / diagnostic.CHECKPOINT_DIRNAME / "model").mkdir(parents=True)
    (checkpoint_root / diagnostic.TRAINING_METADATA).write_text(
        json.dumps({"checkpoint_id": "mock-checkpoint"}) + "\n",
        encoding="utf-8",
    )
    return standardized, config_path, checkpoint_root, config


def _patch_mock_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: dict,
    runs_dir: Path,
    run_id: str,
) -> list[bool]:
    selection_seen_before_model_load: list[bool] = []
    monkeypatch.setattr(diagnostic, "load_v12_ranker_config", lambda path: config)
    monkeypatch.setattr(diagnostic, "_assert_frozen_training_population", lambda *args: None)
    monkeypatch.setattr(diagnostic, "_validate_scoring_checkpoint_provenance", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        diagnostic,
        "_checkpoint_identity",
        lambda *args: ("mock-checkpoint", [{"name": "mock.safetensors", "sha256": "mock"}]),
    )
    monkeypatch.setattr(diagnostic, "_validate_probe_model_binding", lambda *args, **kwargs: None)
    monkeypatch.setattr(diagnostic, "_git_revision", lambda: "mock-revision")
    monkeypatch.setattr(
        diagnostic,
        "_frozen_ranker_implementation_identity",
        lambda: {"digest": "mock-frozen-ranker", "files": []},
    )
    monkeypatch.setattr(
        diagnostic,
        "_runtime_metadata",
        lambda *args: {"runtime": "cpu-mock"},
    )

    def fake_loader(*args: object, **kwargs: object) -> tuple[_Tokenizer, _TinyQ2]:
        del args, kwargs
        selection_path = runs_dir / run_id / "selection_manifest.json"
        observed = selection_path.is_file()
        if observed:
            payload = json.loads(selection_path.read_text(encoding="utf-8"))
            observed = payload["finalized_before_model_load_and_loss"] is True
        selection_seen_before_model_load.append(observed)
        return _Tokenizer(), _TinyQ2()

    def fake_loss(
        model: torch.nn.Module,
        tokenizer: object,
        groups: list[TrainingGroup],
        config_value: dict,
        *,
        device: str,
    ) -> tuple[torch.Tensor, dict]:
        del tokenizer, config_value, device
        assert (runs_dir / run_id / "selection_manifest.json").is_file()
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        weighted_parameters = sum(
            (index + 1) * parameter.sum()
            for index, parameter in enumerate(trainable)
        )
        gain_weight = sum(
            (index + 1) * float(gain)
            for index, gain in enumerate(groups[0].gains)
        )
        loss = torch.exp(weighted_parameters * 0.001) * gain_weight
        return loss, {"mock_gain_weight": gain_weight}

    monkeypatch.setattr(diagnostic, "_load_model_and_tokenizer", fake_loader)
    monkeypatch.setattr(diagnostic, "_training_batch_loss", fake_loss)
    return selection_seen_before_model_load


def test_cpu_mock_run_is_train_only_auditable_and_permanently_non_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    standardized, config_path, checkpoint_root, config = _mock_run_fixture(tmp_path)
    runs_dir = tmp_path / "runs"
    run_id = "20260717_kuaisearch_q2_m3_mock"
    selection_seen = _patch_mock_run(
        monkeypatch,
        config=config,
        runs_dir=runs_dir,
        run_id=run_id,
    )
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name in {"qrels_dev.jsonl", "qrels_confirmation.jsonl", "qrels_test.jsonl"}:
            raise AssertionError(f"forbidden qrels opened: {path.name}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    metadata = diagnostic.run_gradient_diagnostic(
        standardized,
        config_path,
        checkpoint_root,
        "base_initialization",
        run_id,
        device="cpu",
        runs_dir=runs_dir,
        command=["cpu-mock"],
        max_requests_per_surface=1,
    )
    assert selection_seen == [True]
    assert metadata["status"] == "completed"
    assert metadata["evidence_mode"] == "smoke_non_result"
    assert metadata["result_eligible"] is False
    assert metadata["non_result_reason"] == "request_cap"
    assert metadata["optimizer_steps_performed"] == 0
    assert metadata["matched_4096_group_256_step_training_control_executed"] is False
    assert metadata["qrels_access"] == {
        "qrels_train_path": str(standardized / "qrels_train.jsonl"),
        "qrels_train_read": True,
        "qrels_train_sha256": config["_protocol"]["data"]["development_population"]["qrels_train_sha256"],
        "qrels_dev_read": False,
        "qrels_confirmation_read": False,
        "qrels_test_read": False,
    }
    assert metadata["parameter_boundary_audit"]["trainable_parameter_object_count"] == 11
    selection = json.loads(
        (runs_dir / run_id / "selection_manifest.json").read_text(encoding="utf-8")
    )
    assert selection["registered_requests_per_surface"] == 96
    assert selection["smoke_request_cap"] == 1
    assert all(
        selection["surfaces"][surface]["request_count"] == 1
        for surface in diagnostic.SURFACES
    )
    rows = [
        json.loads(line)
        for line in (runs_dir / run_id / "per_request.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 6
    assert {row["status"] for row in rows} == {"completed"}
    shuffled = [row for row in rows if row["control"] == "within_request_label_shuffle"]
    assert len(shuffled) == 3
    assert all(row["label_shuffle"]["changed_positions"] > 0 for row in shuffled)
    assert all(row["label_shuffle"]["preserved_gain_multiset"] for row in shuffled)


def test_cpu_mock_wall_exit_resumes_without_losing_partial_attempt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    standardized, config_path, checkpoint_root, config = _mock_run_fixture(tmp_path)
    runs_dir = tmp_path / "runs"
    run_id = "20260717_kuaisearch_q2_m3_resume_mock"
    _patch_mock_run(
        monkeypatch,
        config=config,
        runs_dir=runs_dir,
        run_id=run_id,
    )
    monotonic_values = iter((0.0, 2.0, 3.0))
    monkeypatch.setattr(diagnostic, "_monotonic", lambda: next(monotonic_values))
    first = diagnostic.run_gradient_diagnostic(
        standardized,
        config_path,
        checkpoint_root,
        "base_initialization",
        run_id,
        device="cpu",
        runs_dir=runs_dir,
        command=["cpu-mock"],
        max_wall_seconds=1.0,
        max_requests_per_surface=1,
    )
    assert first["status"] == "wall_time_exhausted"
    partial_path = Path(first["partial_attempt_path"])
    assert partial_path.is_file()
    partial_sha256 = first["partial_attempt_sha256"]

    monkeypatch.setattr(diagnostic, "_monotonic", lambda: 0.0)
    second = diagnostic.run_gradient_diagnostic(
        standardized,
        config_path,
        checkpoint_root,
        "base_initialization",
        run_id,
        device="cpu",
        runs_dir=runs_dir,
        command=["cpu-mock"],
        resume=True,
        max_wall_seconds=1.0,
        max_requests_per_surface=1,
    )
    assert second["status"] == "completed"
    assert partial_path.is_file()
    assert diagnostic.sha256_file(partial_path) == partial_sha256
    assert second["resume_lineage"] == [
        {
            "completed_cells": [],
            "from_status": "wall_time_exhausted",
            "partial_attempt_path": str(partial_path),
            "partial_attempt_sha256": partial_sha256,
            "resumed_at": second["resume_lineage"][0]["resumed_at"],
        }
    ]
    progress = json.loads(
        (runs_dir / run_id / "progress.json").read_text(encoding="utf-8")
    )
    assert progress["resume_count"] == 1
    assert len(progress["completed_cells"]) == 6
