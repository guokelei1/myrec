from __future__ import annotations

import pytest

from myrec.mechanism.q3_optimizer_replay_runtime import (
    _lora_pairs,
    _next_linear_lr,
    _path_function_summaries,
)


def test_q3_lora_pair_coverage_is_exactly_28_by_qv():
    identities = {}
    for block in range(28):
        for projection in ("q", "v"):
            for factor in ("A", "B"):
                identities[f"{block}.{projection}.{factor}"] = {
                    "block_zero_based": block,
                    "projection": projection,
                    "factor": factor,
                }
    pairs = _lora_pairs(identities)
    assert len(pairs) == 56
    assert pairs[(13, "q")] == {"A": "13.q.A", "B": "13.q.B"}


def test_q3_next_scheduler_lr_is_hand_computed():
    state = {
        "training_contract": {"total_optimizer_steps": 967},
        "scheduler": {"base_lrs": [2e-4], "last_epoch": 500},
    }
    assert _next_linear_lr(state) == pytest.approx(2e-4 * 466 / 871)


def test_q3_path_summary_separates_existing_weight_from_step_delta():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    b = torch.tensor([[3.0, 0.0], [0.0, 4.0]])
    delta_a = torch.tensor([[0.5, 0.0], [0.0, 0.0]])
    delta_b = torch.tensor([[0.0, 0.0], [0.0, 1.0]])
    names = {"A": "layer.A", "B": "layer.B"}
    rows = _path_function_summaries(
        [(names["A"], a), (names["B"], b)],
        {(0, "q"): names},
        {
            "a_only": {names["A"]: delta_a, names["B"]: None},
            "b_only": {names["A"]: None, names["B"]: delta_b},
            "joint": {names["A"]: delta_a, names["B"]: delta_b},
        },
    )
    row = rows[0]
    pre = 2.0 * (b @ a)
    post = 2.0 * ((b + delta_b) @ (a + delta_a))
    delta = post - pre
    assert row["step500_effective_weight_norm"] == pytest.approx(
        pre.double().norm().item()
    )
    assert row["post_step501_effective_weight_norm"] == pytest.approx(
        post.double().norm().item()
    )
    assert row["step501_effective_delta_norm"] == pytest.approx(
        delta.double().norm().item()
    )
    assert row["joint_function_norm"] == pytest.approx(delta.double().norm().item())
