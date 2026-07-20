from __future__ import annotations

import pytest

from myrec.mechanism.q3_native_readout_scoring import (
    Q3_READOUT_SCOPES,
    capture_q3_native_readout,
    compose_q3_readout_terms,
    q3_native_score_from_terms,
    q3_score_low_precision_bound,
)


def test_q3_native_score_and_scope_substitution_are_hand_computed():
    torch = pytest.importorskip("torch")
    recipient = torch.tensor(
        [[-1.0, -2.0, -3.0, -4.0], [-2.0, -4.0, -6.0, -8.0]]
    )
    donor = torch.tensor(
        [[-11.0, -12.0, -13.0, -14.0], [-12.0, -14.0, -16.0, -18.0]]
    )
    expected_columns = {
        "shared_prompt": (0, 2),
        "yes_context": (1,),
        "no_context": (3,),
        "joint": (0, 1, 2, 3),
    }
    assert tuple(expected_columns) == Q3_READOUT_SCOPES
    for scope, columns in expected_columns.items():
        result = compose_q3_readout_terms(recipient, donor, scope=scope)
        expected = recipient.clone()
        expected[:, list(columns)] = donor[:, list(columns)]
        torch.testing.assert_close(result["terms"], expected, rtol=0, atol=0)
        hand = 0.5 * (
            expected[:, 0] + expected[:, 1] - expected[:, 2] - expected[:, 3]
        )
        torch.testing.assert_close(result["score"], hand, rtol=0, atol=0)
    torch.testing.assert_close(
        q3_native_score_from_terms(recipient), torch.tensor([2.0, 4.0])
    )
    # Both signed path values are O(1), even if their difference is small.
    bound = q3_score_low_precision_bound(
        torch.tensor([[-10.0, -10.0, -10.0, -10.0]])
    )
    assert bound.item() == pytest.approx(0.625)


def test_q3_native_readout_captures_all_three_states_and_exact_shared_prompt():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()
    yes_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    no_ids = torch.tensor([[1, 2, 3, 4, 7, 6]])
    context = {
        "paths": {
            "yes": {
                "ids": yes_ids,
                "mask": torch.ones_like(yes_ids),
                "positions": torch.tensor([[3, 4]]),
                "target": [5, 6],
            },
            "no": {
                "ids": no_ids,
                "mask": torch.ones_like(no_ids),
                "positions": torch.tensor([[3, 4]]),
                "target": [7, 6],
            },
        }
    }
    with torch.no_grad():
        result = capture_q3_native_readout(model, context)
    assert result["terms"].shape == (1, 4)
    assert result["score"].shape == (1,)
    assert result["shared_prompt_path_max_abs_delta"] == {
        "final_rmsnorm_input": 0.0,
        "final_rmsnorm_output": 0.0,
    }
    assert result["algebra_low_precision_max_ratio"] <= 1.0
    for branch in ("yes", "no"):
        assert result["branches"][branch]["final_rmsnorm_input"].shape == (1, 2, 32)
        assert result["branches"][branch]["final_rmsnorm_output"].shape == (1, 2, 32)


def test_q3_term_substitution_equals_direct_final_norm_state_substitution():
    """Each registered term scope is exactly the matching final-state patch.

    This is the load-bearing boundary behind treating the D6 intervention as a
    native-readout decomposition rather than as an additional hidden-layer
    patch.  The donor and recipient share the frozen Yes/No target tokens but
    differ in their prompt prefix, as they do in the registered cross-request
    control.
    """

    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen3Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=28,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
    )
    model = transformers.Qwen3ForCausalLM(config).eval()

    def context(prefix_token):
        yes_ids = torch.tensor([[prefix_token, 2, 3, 4, 5, 6]])
        no_ids = torch.tensor([[prefix_token, 2, 3, 4, 7, 6]])
        return {
            "paths": {
                "yes": {
                    "ids": yes_ids,
                    "mask": torch.ones_like(yes_ids),
                    "positions": torch.tensor([[3, 4]]),
                    "target": [5, 6],
                },
                "no": {
                    "ids": no_ids,
                    "mask": torch.ones_like(no_ids),
                    "positions": torch.tensor([[3, 4]]),
                    "target": [7, 6],
                },
            }
        }

    recipient_context = context(1)
    donor_context = context(9)
    with torch.no_grad():
        recipient = capture_q3_native_readout(model, recipient_context)
        donor = capture_q3_native_readout(model, donor_context)

    state_indices = {
        "shared_prompt": {"yes": (0,), "no": (0,)},
        "yes_context": {"yes": (1,), "no": ()},
        "no_context": {"yes": (), "no": (1,)},
        "joint": {"yes": (0, 1), "no": (0, 1)},
    }
    output_embeddings = model.get_output_embeddings()
    for scope, branch_indices in state_indices.items():
        recomputed_paths = {}
        for branch in ("yes", "no"):
            hidden = recipient["branches"][branch]["final_rmsnorm_output"].clone()
            for index in branch_indices[branch]:
                hidden[:, index] = donor["branches"][branch][
                    "final_rmsnorm_output"
                ][:, index]
            logits = output_embeddings(hidden).float()
            targets = torch.tensor(
                recipient_context["paths"][branch]["target"], dtype=torch.long
            )[None, :, None]
            recomputed_paths[branch] = torch.nn.functional.log_softmax(
                logits, dim=-1
            ).gather(2, targets).squeeze(2)

        direct_terms = torch.stack(
            (
                recomputed_paths["yes"][:, 0],
                recomputed_paths["yes"][:, 1],
                recomputed_paths["no"][:, 0],
                recomputed_paths["no"][:, 1],
            ),
            dim=1,
        )
        composed = compose_q3_readout_terms(
            recipient["algebra_terms"], donor["algebra_terms"], scope=scope
        )
        torch.testing.assert_close(composed["terms"], direct_terms, rtol=0, atol=0)
        torch.testing.assert_close(
            composed["score"], q3_native_score_from_terms(direct_terms), rtol=0, atol=0
        )


def test_q3_readout_scope_rejects_unregistered_or_misaligned_terms():
    torch = pytest.importorskip("torch")
    terms = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="unregistered"):
        compose_q3_readout_terms(terms, terms, scope="outcome_selected")
    with pytest.raises(ValueError, match="misaligned"):
        compose_q3_readout_terms(terms, torch.zeros(1, 4), scope="joint")
    with pytest.raises(ValueError, match="shape"):
        q3_native_score_from_terms(torch.zeros(2, 3))
