from __future__ import annotations

from types import SimpleNamespace

import pytest

from myrec.mechanism.contextual_control_evaluator import (
    _common_implementation_digest,
)
from myrec.mechanism.contextual_control_scoring import _attention_null_path


def test_attention_null_changes_only_registered_mask_span():
    torch = pytest.importorskip("torch")
    path = {
        "ids": torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]),
        "mask": torch.ones(2, 5, dtype=torch.long),
        "starts": torch.tensor([1, 2]),
        "ends": torch.tensor([3, 4]),
    }
    result = _attention_null_path(path)
    assert result["ids"].data_ptr() == path["ids"].data_ptr()
    assert result["mask"].tolist() == [[1, 0, 0, 1, 1], [1, 1, 0, 0, 1]]
    assert path["mask"].tolist() == [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]


def test_internal_history_mask_does_not_compact_qwen_rope_positions():
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
    path = {
        "ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "mask": torch.ones(1, 5, dtype=torch.long),
        "starts": torch.tensor([1]),
        "ends": torch.tensor([3]),
    }
    observed = []

    def capture_positions(_module, args):
        observed.append(args[1].detach().clone())

    handle = model.model.rotary_emb.register_forward_pre_hook(capture_positions)
    try:
        with torch.no_grad():
            model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
            )
            masked = _attention_null_path(path)
            model(
                input_ids=masked["ids"],
                attention_mask=masked["mask"],
                use_cache=False,
            )
    finally:
        handle.remove()
    assert len(observed) == 2
    torch.testing.assert_close(observed[0], observed[1], rtol=0, atol=0)
    torch.testing.assert_close(
        observed[0], torch.arange(5, dtype=torch.long)[None, :], rtol=0, atol=0
    )


def test_contextual_evaluator_requires_one_implementation_digest():
    bundles = {
        "q2": SimpleNamespace(
            metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
        ),
        "q3": SimpleNamespace(
            metadata={"implementation_identity": {"digest": "fixed"}, "run_contract": {"implementation_digest": "fixed"}}
        ),
    }
    assert _common_implementation_digest(bundles) == "fixed"
    bundles["q3"].metadata["run_contract"]["implementation_digest"] = "drifted"
    with pytest.raises(ValueError, match="differs from run contract"):
        _common_implementation_digest(bundles)
    bundles["q3"].metadata["run_contract"]["implementation_digest"] = "fixed"
    bundles["q3"].metadata["implementation_identity"]["digest"] = "drifted"
    with pytest.raises(ValueError, match="different implementation digests"):
        _common_implementation_digest(bundles)
