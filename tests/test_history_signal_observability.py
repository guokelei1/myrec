import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.analysis.history_signal_observability import (
    MODES,
    CompactFoldLabels,
    HistorySignalTransformer,
    fold_for_user,
    listwise_loss,
)


def make_model(mode: str) -> HistorySignalTransformer:
    model = HistorySignalTransformer(
        mode=mode,
        input_dim=8,
        width=16,
        heads=4,
        context_layers=1,
        candidate_layers=1,
        ffn_dim=32,
        dropout=0.0,
        id_buckets=32,
        id_dim=4,
        max_history=3,
        zero_initial_output=False,
    )
    model.eval()
    return model


def batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {
        "query_semantic": torch.randn(2, 8, generator=generator),
        "candidate_semantic": torch.randn(2, 4, 8, generator=generator),
        "candidate_indices": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 0]]),
        "candidate_mask": torch.tensor(
            [[True, True, True, True], [True, True, True, False]]
        ),
        "candidate_popularity": torch.tensor(
            [[1.0, 2.0, 0.0, 4.0], [2.0, 0.0, 1.0, 0.0]]
        ),
        "history_semantic": torch.randn(2, 3, 8, generator=generator),
        "history_indices": torch.tensor([[8, 9, 10], [11, 12, 0]]),
        "history_mask": torch.tensor([[True, True, True], [True, True, False]]),
        "history_weight": torch.tensor([[0.2, 0.5, 1.0], [0.4, 1.0, 0.0]]),
    }


def test_all_modes_have_identical_parameter_count() -> None:
    counts = {make_model(mode).parameter_count() for mode in MODES}
    assert len(counts) == 1


def test_null_mode_ignores_history() -> None:
    model = make_model("null")
    values = batch()
    altered = dict(values)
    altered["history_semantic"] = values["history_semantic"] * 100.0
    altered["history_indices"] = values["history_indices"] + 7
    altered["history_weight"] = values["history_weight"] + 10.0
    with torch.inference_mode():
        first = model(**values).scores
        second = model(**altered).scores
    assert torch.equal(first, second)


def test_candidate_permutation_is_equivariant() -> None:
    model = make_model("full")
    values = batch()
    permutation = torch.tensor([3, 1, 0, 2])
    inverse = torch.argsort(permutation)
    changed = dict(values)
    for name in (
        "candidate_semantic",
        "candidate_indices",
        "candidate_mask",
        "candidate_popularity",
    ):
        changed[name] = values[name][:, permutation]
    with torch.inference_mode():
        first = model(**values).scores
        second = model(**changed).scores[:, inverse]
    assert torch.allclose(first, second, atol=2e-6, rtol=0.0)


def test_listwise_loss_is_finite_and_backward_reaches_history() -> None:
    model = make_model("full")
    values = batch()
    output = model(**values)
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    loss = listwise_loss(output, labels, values["candidate_mask"], residual_l2_weight=0.0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.history_position.grad is not None
    assert model.history_position.grad.ne(0).any()


def test_compact_labels_reject_heldout_request() -> None:
    labels = CompactFoldLabels(
        request_indices=np.asarray([2, 5]),
        offsets=np.asarray([0, 2, 5]),
        values=np.asarray([1, 0, 0, 1, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(labels.row(5, 3), np.asarray([0, 1, 0]))
    try:
        labels.row(4, 2)
    except PermissionError:
        pass
    else:
        raise AssertionError("heldout request unexpectedly had a compact label")


def test_user_fold_is_deterministic() -> None:
    assert fold_for_user("u1", "hso-v1", 3) == fold_for_user("u1", "hso-v1", 3)
    assert 0 <= fold_for_user("u2", "hso-v1", 3) < 3
