from types import SimpleNamespace

import numpy as np
import pytest

from myrec.mechanism.q1_branch_scoring import (
    Q1_BRANCH_CONDITIONS,
    Q1_BRANCH_NODES,
    _capture_q1_paths,
    _score_q1_paths,
)
from myrec.mechanism.transformer_instrumentation import NodeSpec


def test_q1_branch_nodes_and_conditions_match_registered_q0_breadth():
    assert Q1_BRANCH_NODES == (
        "block_input_residual",
        "attention_o_projection",
        "mlp_down_projection",
        "block_output_residual",
    )
    assert len(Q1_BRANCH_CONDITIONS) == 10
    assert len(set(Q1_BRANCH_CONDITIONS)) == 10


def test_q1_all_internal_branch_nodes_identity_replay_cached_response_path():
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
    specs = tuple(NodeSpec(node_id=node, block=13) for node in Q1_BRANCH_NODES)
    with torch.no_grad():
        captured = _capture_q1_paths(
            model,
            tokenizer,
            record,
            prompt,
            specs,
            device="cpu",
            batch_size=2,
        )
        for spec in specs:
            replayed = _score_q1_paths(
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
            assert replayed["call_audit"]["all_response_tokens_patched"] is True
    assert set(captured["donors"]) == {spec.key for spec in specs}
    assert all(
        len(captured["donors"][spec.key]["continuations"]) == 1
        for spec in specs
    )
