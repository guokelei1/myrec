"""Pre-outcome executable-contract tests for C08 G1."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from g1_protocol import (
    FROZEN,
    MECHANISMS,
    STREAM_OFFSETS,
    corrupt_supported,
    generate_split,
    initialized_models,
    item_recurrence_accuracy,
    make_batch_schedule,
    tensor_state_hash,
    top1_with_ties,
)


ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def evaluation():
    return generate_split(FROZEN.seeds[0], "eval")


def test_frozen_constants_are_literal_and_complete() -> None:
    assert FROZEN.seeds == (20260711, 20260712, 20260713)
    assert FROZEN.train_requests == 4096
    assert FROZEN.evaluation_requests == 1024
    assert FROZEN.candidate_count == 8
    assert FROZEN.history_length == 8
    assert FROZEN.evidence_width == 16
    assert FROZEN.batch_size == 64
    assert FROZEN.optimizer_steps == 400
    assert FROZEN.learning_rate == 0.003
    assert FROZEN.repeat_noninferiority_pp == 0.01
    assert FROZEN.supported_best_control_advantage_pp == 0.05
    assert FROZEN.supported_ordinary_advantage_pp == 0.03
    assert FROZEN.corruption_margin_retention_max == 0.25
    assert MECHANISMS == ("rwpu", "ordinary", "attention", "pooled_ffn")
    assert len(set(STREAM_OFFSETS.values())) == len(STREAM_OFFSETS)


def test_generator_counts_repeat_and_supported_semantics(evaluation) -> None:
    assert evaluation.query.shape == (1024, 16)
    assert evaluation.history.shape == (1024, 8, 16)
    assert evaluation.candidates.shape == (1024, 8, 16)
    assert evaluation.kind.bincount().tolist() == [512, 512]
    assert evaluation.history_mask.all()
    assert torch.equal(
        torch.sort(evaluation.candidate_ids, dim=1).values,
        torch.arange(8).view(1, 8).expand(1024, -1),
    )

    repeated = evaluation.subset(torch.nonzero(evaluation.kind.eq(0)).view(-1))
    equal = repeated.candidates.unsqueeze(2).eq(repeated.history.unsqueeze(1)).all(-1)
    target_is_repeated = equal.any(2).gather(1, repeated.target_index.unsqueeze(1))
    assert target_is_repeated.all()
    assert equal.any(2).sum(1).eq(1).all()
    assert item_recurrence_accuracy(repeated, FROZEN.seeds[0]) == 1.0

    supported = evaluation.subset(torch.nonzero(evaluation.kind.eq(1)).view(-1))
    equal = supported.candidates.unsqueeze(2).eq(supported.history.unsqueeze(1)).all(-1)
    assert not equal.any()
    for row in range(supported.query.shape[0]):
        topic = int(supported.query_topic[row])
        positions = torch.nonzero(supported.history_topic[row].eq(topic)).view(-1)
        latest = int(positions.max())
        style = int(supported.history_style[row, latest])
        semantic = 2 * topic + (1 if style > 0 else 0)
        assert int(supported.candidate_ids[row, supported.target_index[row]]) == semantic


def test_named_rng_streams_are_repeatable_and_seed_separated(evaluation) -> None:
    repeated = generate_split(FROZEN.seeds[0], "eval")
    for field in (
        "query",
        "history",
        "candidates",
        "candidate_ids",
        "target_index",
        "kind",
    ):
        assert torch.equal(getattr(evaluation, field), getattr(repeated, field))
    other = generate_split(FROZEN.seeds[1], "eval")
    assert not torch.equal(evaluation.history, other.history)
    assert not torch.equal(evaluation.target_index, other.target_index)


def test_corruptions_change_only_the_frozen_fields(evaluation) -> None:
    supported = evaluation.subset(torch.nonzero(evaluation.kind.eq(1)).view(-1))
    wrong = corrupt_supported(supported, FROZEN.seeds[0], "wrong_history")
    shuffled = corrupt_supported(supported, FROZEN.seeds[0], "shuffled_event")
    masked = corrupt_supported(supported, FROZEN.seeds[0], "query_mask")
    disjoint = corrupt_supported(supported, FROZEN.seeds[0], "disjoint")

    for corrupted in (wrong, shuffled, masked, disjoint):
        assert torch.equal(corrupted.candidates, supported.candidates)
        assert torch.equal(corrupted.candidate_ids, supported.candidate_ids)
        assert torch.equal(corrupted.target_index, supported.target_index)
        assert torch.equal(corrupted.kind, supported.kind)

    assert not wrong.history.eq(supported.history).all(dim=(1, 2)).any()
    assert not shuffled.history.eq(supported.history).all(dim=(1, 2)).all()
    assert torch.allclose(
        torch.sort(torch.linalg.vector_norm(shuffled.history, dim=-1), dim=1).values,
        torch.sort(torch.linalg.vector_norm(supported.history, dim=-1), dim=1).values,
    )
    assert torch.equal(masked.query, torch.zeros_like(masked.query))
    assert torch.equal(masked.history, supported.history)
    assert torch.equal(disjoint.history[..., :8], torch.zeros_like(disjoint.history[..., :8]))
    assert torch.equal(disjoint.history[..., 8:], supported.history[..., :8])


def test_controls_have_identical_parameters_initialization_and_active_gradients(
    evaluation,
) -> None:
    models = initialized_models(FROZEN.seeds[0])
    counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    assert len(set(counts.values())) == 1
    hashes = {name: tensor_state_hash(model) for name, model in models.items()}
    assert len(set(hashes.values())) == 1
    names = {
        name: [(key, tuple(value.shape)) for key, value in model.named_parameters()]
        for name, model in models.items()
    }
    assert all(value == names["rwpu"] for value in names.values())

    batch = evaluation.subset(torch.arange(32))
    for model in models.values():
        scores = model(**batch.as_model_inputs())
        loss = F.cross_entropy(scores, batch.target_index)
        loss.backward()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum().item() > 0.0


def test_schedule_is_exact_shared_and_deterministic() -> None:
    schedule = make_batch_schedule(FROZEN.seeds[0])
    assert schedule.shape == (400, 64)
    assert torch.equal(schedule, make_batch_schedule(FROZEN.seeds[0]))
    assert torch.equal(torch.sort(schedule[:64].reshape(-1)).values, torch.arange(4096))
    assert not torch.equal(schedule, make_batch_schedule(FROZEN.seeds[1]))


def test_tie_break_is_input_order_invariant() -> None:
    scores = torch.zeros(1, 8)
    candidate_ids = torch.tensor([[7, 5, 3, 1, 6, 4, 2, 0]])
    request = torch.tensor([19])
    winner = top1_with_ties(scores, candidate_ids, request, FROZEN.seeds[0])
    winner_id = int(candidate_ids[0, winner[0]])
    permutation = torch.tensor([3, 1, 7, 0, 6, 2, 5, 4])
    permuted_winner = top1_with_ties(
        scores[:, permutation],
        candidate_ids[:, permutation],
        request,
        FROZEN.seeds[0],
    )
    permuted_winner_id = int(candidate_ids[0, permutation[permuted_winner[0]]])
    assert winner_id == permuted_winner_id


def test_rwpu_empty_fallback_and_candidate_permutation(evaluation) -> None:
    model = initialized_models(FROZEN.seeds[0])["rwpu"].eval()
    batch = evaluation.subset(torch.arange(16))
    empty = batch.subset(torch.arange(16))
    empty.history = torch.zeros_like(empty.history)
    empty.history_mask = torch.zeros_like(empty.history_mask)
    with torch.no_grad():
        history_scores = model(**empty.as_model_inputs())
        query_only = model.query_only(empty.query, empty.candidates)
    assert torch.equal(history_scores, query_only)

    permutation = torch.arange(7, -1, -1)
    permuted = batch.subset(torch.arange(16))
    permuted.candidates = batch.candidates[:, permutation]
    permuted.candidate_ids = batch.candidate_ids[:, permutation]
    with torch.no_grad():
        original = model(**batch.as_model_inputs())
        changed = model(**permuted.as_model_inputs())
    assert torch.allclose(changed, original[:, permutation], atol=1e-6, rtol=0.0)


def test_executor_has_no_repository_data_or_gpu_execution_path() -> None:
    source = "\n".join(
        (ROOT / name).read_text(encoding="utf-8")
        for name in ("g1_protocol.py", "run_g1.py")
    )
    for forbidden in (
        "qrels_dev",
        "qrels_test",
        "data/standardized",
        "evaluate_scores",
        'device="cuda',
        "device='cuda",
        ".cuda()",
    ):
        assert forbidden not in source


def test_execution_lock_manifest_passes_independent_precheck() -> None:
    lock_path = ROOT / "G1_EXECUTION_LOCK.json"
    if not lock_path.exists():
        pytest.skip("execution lock is created only after source/test freeze")
    from verify_g1_lock import verify

    result = verify("pre")
    assert result["passed"], json.dumps(result, indent=2)
