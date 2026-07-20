from __future__ import annotations

from types import SimpleNamespace

import pytest

from myrec.mechanism.rope_evaluator import _common_implementation_digest
from myrec.mechanism.rope_interventions import (
    QwenRoPEPhaseIntervention,
    _rotate_vector_by_delta,
)


def _tiny_qwen():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        rope_theta=1_000_000.0,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([4, 4])
    starts = torch.tensor([1, 1])
    ends = torch.tensor([3, 3])
    return torch, model, ids, mask, positions, starts, ends


def test_zero_rope_phase_delta_is_exact_score_identity():
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenRoPEPhaseIntervention(model, 13, "zero_phase_delta") as intervention:
            intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
            observed = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            summary = intervention.disarm()
    torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)
    assert summary["maximum_query_norm_error"] == 0.0


def test_common_offset_audits_fp32_geometry_but_keeps_native_backend_noop():
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad():
        baseline = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        with QwenRoPEPhaseIntervention(
            model, 13, "common_offset_plus_17"
        ) as intervention:
            intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
            observed = model(
                input_ids=ids, attention_mask=mask, use_cache=False
            ).logits
            summary = intervention.disarm()
    torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)
    assert (
        summary["common_offset_backend_policy"]
        == "fp32_geometry_audit_then_native_qk_noop"
    )
    assert summary["maximum_query_norm_low_precision_ratio"] <= 1.0
    assert summary["maximum_key_norm_low_precision_ratio"] <= 1.0


def test_phase_rotation_preserves_norm_with_float_tolerance():
    torch = pytest.importorskip("torch")
    value = torch.randn(3, 4, 8)
    rotated = _rotate_vector_by_delta(value, 17)
    torch.testing.assert_close(
        rotated.norm(dim=-1), value.norm(dim=-1), rtol=1.0e-5, atol=1.0e-5
    )


def test_rope_supports_two_native_query_positions():
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    positions = torch.stack((positions - 1, positions), dim=1)
    with torch.no_grad(), QwenRoPEPhaseIntervention(
        model, 27, "paired_qk_distance_compression"
    ) as intervention:
        intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
        output = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        summary = intervention.disarm()
    assert torch.isfinite(output).all()
    assert summary["query_positions_per_row"] == 2


@pytest.mark.parametrize(
    "mode",
    [
        "readout_q_distance_compression",
        "history_k_distance_expansion",
        "paired_qk_distance_compression",
    ],
)
def test_rope_intervention_modes_are_finite(mode):
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()
    with torch.no_grad(), QwenRoPEPhaseIntervention(model, 20, mode) as intervention:
        intervention.arm(positions, starts, ends, sequence_length=ids.shape[1])
        output = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        summary = intervention.disarm()
    assert torch.isfinite(output).all()
    assert summary["history_tokens"] == 4


@pytest.mark.parametrize(
    ("mode", "changes_query", "changes_key"),
    [
        ("readout_q_distance_compression", True, False),
        ("history_k_distance_compression", False, True),
        ("paired_qk_distance_compression", True, True),
    ],
)
def test_rope_modes_change_only_registered_post_rope_qk_rows(
    mode, changes_query, changes_key
):
    torch, model, ids, mask, positions, starts, ends = _tiny_qwen()

    def capture(mode_name):
        observed = {}
        with torch.no_grad(), QwenRoPEPhaseIntervention(
            model, 20, mode_name
        ) as intervention:
            native = intervention.original_function

            def record(module, query, key, value, attention_mask, **kwargs):
                if int(module.layer_idx) == 20:
                    observed["query"] = query.detach().clone()
                    observed["key"] = key.detach().clone()
                return native(
                    module,
                    query,
                    key,
                    value,
                    attention_mask,
                    **kwargs,
                )

            intervention.original_function = record
            try:
                intervention.arm(
                    positions, starts, ends, sequence_length=ids.shape[1]
                )
                model(input_ids=ids, attention_mask=mask, use_cache=False)
                summary = intervention.disarm()
            finally:
                intervention.original_function = native
        return observed, summary

    baseline, _baseline_summary = capture("zero_phase_delta")
    observed, summary = capture(mode)
    rows = torch.arange(ids.shape[0])
    query_mask = torch.zeros(ids.shape, dtype=torch.bool)
    query_mask[rows, positions] = changes_query
    key_mask = torch.zeros(ids.shape, dtype=torch.bool)
    if changes_key:
        for row, (start, end) in enumerate(zip(starts.tolist(), ends.tolist())):
            key_mask[row, start:end] = True

    baseline_query = baseline["query"].transpose(1, 2)
    observed_query = observed["query"].transpose(1, 2)
    baseline_key = baseline["key"].transpose(1, 2)
    observed_key = observed["key"].transpose(1, 2)
    query_mask = query_mask.to(observed_query.device)
    key_mask = key_mask.to(observed_key.device)
    if changes_query:
        assert not torch.equal(
            observed_query[query_mask], baseline_query[query_mask]
        )
    torch.testing.assert_close(
        observed_query[~query_mask],
        baseline_query[~query_mask],
        rtol=0.0,
        atol=0.0,
    )
    if changes_key:
        assert not torch.equal(observed_key[key_mask], baseline_key[key_mask])
    torch.testing.assert_close(
        observed_key[~key_mask],
        baseline_key[~key_mask],
        rtol=0.0,
        atol=0.0,
    )
    assert summary["maximum_query_norm_low_precision_ratio"] <= 1.0
    assert summary["maximum_key_norm_low_precision_ratio"] <= 1.0


def test_rope_evaluator_requires_one_implementation_digest():
    bundles = {
        "q2": {
            13: SimpleNamespace(
                metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
            ),
            20: SimpleNamespace(
                metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
            ),
        },
        "q3": {
            13: SimpleNamespace(
                metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
            ),
        },
    }
    assert _common_implementation_digest(bundles) == "fixed"
    bundles["q3"][13].metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _common_implementation_digest(bundles)
    bundles["q3"][13].metadata["run_contract"]["implementation_digest"] = "fixed"
    bundles["q3"][13].metadata["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests"):
        _common_implementation_digest(bundles)
