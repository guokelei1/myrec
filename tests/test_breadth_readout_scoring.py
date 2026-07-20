from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from myrec.mechanism import breadth_readout_scoring as scoring
from myrec.mechanism.breadth_readout_evaluator import (
    _common_implementation_digest,
)
from myrec.mechanism.breadth_readout_runtime import _stable_smoke_records
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
)


def test_breadth_readout_nodes_and_conditions_are_fixed():
    assert scoring.BREADTH_READOUT_NODES == (
        "final_rmsnorm_input",
        "final_rmsnorm_output",
    )
    assert len(scoring.BREADTH_READOUT_CONDITIONS) == 6
    assert len(set(scoring.BREADTH_READOUT_CONDITIONS)) == 6


@pytest.mark.parametrize("node_id", scoring.BREADTH_READOUT_NODES)
def test_q0_final_readout_identity_is_exact_on_tiny_qwen(node_id):
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
        max_position_embeddings=64,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    mask = torch.ones_like(ids)
    positions = torch.tensor([[3], [3]])
    path = {"ids": ids, "mask": mask, "positions": positions}
    spec = NodeSpec(node_id=node_id, block=None)
    with torch.no_grad():
        with QwenNodeCapture(model, (spec,)) as capture:
            output, donors = capture.capture_forward(
                input_ids=ids,
                attention_mask=mask,
                positions=positions,
                model_kwargs={"logits_to_keep": 1},
            )
            expected = (
                output.logits[:, -1, 9] - output.logits[:, -1, 10]
            ).float().cpu().numpy()
        observed = scoring._patched_q0_scores(
            model,
            spec,
            path,
            donors[spec.key],
            yes_token_id=9,
            no_token_id=10,
        )
    np.testing.assert_array_equal(observed, expected)


def test_q1_readout_uses_prefix_and_every_continuation_for_both_nodes(monkeypatch):
    record = SimpleNamespace(
        history=({"item_id": "h"},),
        candidates=({"item_id": "a"}, {"item_id": "b"}),
    )
    monkeypatch.setattr(
        scoring,
        "instrument_q1_selection_prompt",
        lambda _tokenizer, _record, history, _config: SimpleNamespace(
            full=bool(history)
        ),
    )

    def capture(_model, _tokenizer, _record, _prompt, specs, **_kwargs):
        return {
            "scores": np.asarray([1.0, 2.0], dtype=np.float32),
            "donors": {
                spec.key: {"prefix": spec.node_id, "continuations": [spec.node_id]}
                for spec in specs
            },
            "response_tokens": 7,
            "call_audit": {
                "prefix_calls": 1,
                "continuation_calls": 1,
                "response_tokens": 7,
                "all_response_tokens_captured": True,
            },
        }

    def score(_model, _tokenizer, _record, prompt, *, patch_spec, **_kwargs):
        if patch_spec is None:
            values = [0.1, 0.2]
        elif prompt.full:
            values = [1.0, 2.0]
        else:
            values = [0.5, 0.6]
        return {
            "scores": np.asarray(values, dtype=np.float32),
            "response_tokens": 7,
            "call_audit": {
                "prefix_calls": 1,
                "continuation_calls": 1,
                "response_tokens": 7,
                "all_response_tokens_patched": patch_spec is not None,
            },
        }

    monkeypatch.setattr(scoring, "_capture_q1_paths", capture)
    monkeypatch.setattr(scoring, "_score_q1_paths", score)
    result = scoring.score_q1_readout_record(
        object(),
        object(),
        record,
        {"method_id": "q1_instructrec_generalqwen"},
        device="cpu",
        batch_size=2,
    )
    assert result["maximum_identity_delta"] == 0.0
    assert result["response_tokens"] == 7
    assert set(result["call_audit"]["patched"]) == set(
        scoring.BREADTH_READOUT_NODES
    )
    for node in scoring.BREADTH_READOUT_NODES:
        np.testing.assert_allclose(
            result["conditions"][f"{node}_same_to_null"], [0.5, 0.6]
        )


@pytest.mark.parametrize("node_id", scoring.BREADTH_READOUT_NODES)
def test_q1_final_readout_identity_covers_prefix_cache_and_all_response_tokens(
    node_id,
):
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
        max_position_embeddings=64,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    tokenizer = SimpleNamespace(pad_token_id=0)
    record = SimpleNamespace(
        candidates=({"item_id": "a"}, {"item_id": "b"}),
    )
    prompt = SimpleNamespace(
        token_ids=(1, 2, 3, 4),
        prompt_readout=3,
        response_by_item={
            "a": (5, 6, 7),
            "b": (8, 9, 10, 11),
        },
    )
    spec = NodeSpec(node_id=node_id, block=None)
    with torch.no_grad():
        captured = scoring._capture_q1_paths(
            model,
            tokenizer,
            record,
            prompt,
            (spec,),
            device="cpu",
            batch_size=2,
        )
        replayed = scoring._score_q1_paths(
            model,
            tokenizer,
            record,
            prompt,
            patch_spec=spec,
            donors=captured["donors"][spec.key],
            device="cpu",
            batch_size=2,
        )
    np.testing.assert_array_equal(replayed["scores"], captured["scores"])
    assert replayed["response_tokens"] == captured["response_tokens"] == 7
    assert captured["call_audit"] == {
        "prefix_calls": 1,
        "continuation_calls": 1,
        "response_tokens": 7,
        "all_response_tokens_captured": True,
    }
    assert replayed["call_audit"] == {
        "prefix_calls": 1,
        "continuation_calls": 1,
        "response_tokens": 7,
        "all_response_tokens_patched": True,
    }


def test_breadth_readout_smoke_sample_is_order_independent():
    rows = [SimpleNamespace(request_id=f"r{index}") for index in range(20)]
    left = [row.request_id for row in _stable_smoke_records(rows, 5)]
    right = [row.request_id for row in _stable_smoke_records(rows[::-1], 5)]
    assert left == right


def test_breadth_readout_evaluator_requires_one_implementation_digest():
    bundles = {
        "q0": SimpleNamespace(
            metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
        ),
        "q1": SimpleNamespace(
            metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
        ),
    }
    assert _common_implementation_digest(bundles) == "fixed"
    bundles["q1"].metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _common_implementation_digest(bundles)
    bundles["q1"].metadata["run_contract"]["implementation_digest"] = "fixed"
    bundles["q1"].metadata["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests"):
        _common_implementation_digest(bundles)
