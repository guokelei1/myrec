import pytest
import torch

from myrec.mechanism.attention_mlp_interaction import (
    FACTORIAL_CELLS,
    factorial_interaction,
    summarize_factorial_cells,
)


def test_factorial_interaction_has_expected_sign_and_shape():
    cells = {
        "native_native": torch.tensor([1.0, 2.0]),
        "removed_native": torch.tensor([0.5, 1.0]),
        "native_removed": torch.tensor([0.75, 1.5]),
        "removed_removed": torch.tensor([0.25, 0.5]),
    }
    # 0.25 - 0.5 - 0.75 + 1 = 0 for both rows.
    result = factorial_interaction(cells)
    torch.testing.assert_close(result, torch.zeros(2))


def test_factorial_interaction_detects_nonadditive_term():
    cells = {
        "native_native": torch.tensor([10.0]),
        "removed_native": torch.tensor([8.0]),
        "native_removed": torch.tensor([7.0]),
        "removed_removed": torch.tensor([9.0]),
    }
    torch.testing.assert_close(factorial_interaction(cells), torch.tensor([4.0]))


def test_factorial_interaction_requires_all_four_core_cells_and_equal_shapes():
    with pytest.raises(ValueError, match="missing"):
        factorial_interaction({"native_native": torch.ones(1)})
    bad = {
        "native_native": torch.ones(1),
        "removed_native": torch.ones(2),
        "native_removed": torch.ones(1),
        "removed_removed": torch.ones(1),
    }
    with pytest.raises(ValueError, match="shapes"):
        factorial_interaction(bad)


def test_factorial_summary_requires_no_qrels_and_reports_all_cells():
    cells = {name: torch.ones(2) for name in FACTORIAL_CELLS}
    summary = summarize_factorial_cells(cells)
    assert summary["all_factorial_cells_present"] is True
    assert summary["observed_cells"] == list(FACTORIAL_CELLS)
