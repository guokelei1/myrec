from __future__ import annotations

import pytest
import torch

from myrec.mechanism.training_boundary_diagnostics import (
    dtype_cast_boundary_summary,
    effective_update_summary,
    gradient_path_summary,
    lora_dropout_forward,
    summarize_cast_variants,
)


def test_dtype_cast_summary_reports_hand_computed_residual():
    native = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    reference = torch.tensor([1.0, 1.5], dtype=torch.float32)
    result = dtype_cast_boundary_summary(native, reference)
    assert result["maximum_absolute_residual"] == pytest.approx(0.5)
    assert result["residual_l2"] == pytest.approx(0.5)
    assert result["relative_l2_residual"] == pytest.approx(0.5 / (3.25**0.5))
    assert result["finite"] is True


def test_cast_variants_require_and_report_one_frozen_reference():
    variants = {
        "native": torch.tensor([1.0, 2.0]),
        "aligned": torch.tensor([1.0, 2.5]),
        "fp32": torch.tensor([1.0, 2.0]),
    }
    result = summarize_cast_variants(variants, "fp32")
    assert result["reference"] == "fp32"
    assert result["variants"]["native"]["maximum_absolute_residual"] == 0.0
    assert result["variants"]["aligned"]["maximum_absolute_residual"] == pytest.approx(0.5)


def test_lora_dropout_fixed_mask_replays_exactly():
    values = torch.arange(6.0).reshape(2, 3)
    mask = torch.tensor([[True, False, True], [False, True, True]])
    first, summary = lora_dropout_forward(values, 0.5, mask=mask)
    second, replay = lora_dropout_forward(values, 0.5, mask=mask)
    expected = values * mask.to(values.dtype) * 2.0
    torch.testing.assert_close(first, expected)
    torch.testing.assert_close(second, first)
    assert summary["kept_fraction"] == pytest.approx(4.0 / 6.0)
    assert replay["mask_replayable"] is True


def test_lora_dropout_rejects_mismatched_replay_mask():
    with pytest.raises(ValueError, match="mask shape"):
        lora_dropout_forward(torch.ones(2, 3), 0.1, mask=torch.ones(3, dtype=torch.bool))


def test_gradient_path_summary_exposes_missing_bridge_gradients():
    result = gradient_path_summary(
        ["q", "v"],
        [torch.ones(2), None],
        [torch.ones(2), torch.ones(2)],
        family_by_name={"q": "q", "v": "v"},
    )
    assert result["complete"] is False
    assert result["missing_native"] == ["v"]
    assert result["cosine"] is None


def test_gradient_path_summary_reports_complete_alignment():
    result = gradient_path_summary(
        ["q", "v"],
        [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 2.0])],
        [torch.tensor([2.0, 0.0]), torch.tensor([0.0, 4.0])],
        family_by_name={"q": "q", "v": "v"},
    )
    assert result["complete"] is True
    assert result["cosine"] == pytest.approx(1.0)


def test_effective_update_summary_partitions_raw_and_applied_vectors():
    raw = {"q": torch.tensor([3.0, 4.0]), "v": torch.tensor([0.0])}
    applied = {"q": torch.tensor([0.3, 0.4]), "v": torch.tensor([0.0])}
    result = effective_update_summary(
        raw, applied, family_by_name={"q": "attention_q", "v": "attention_v"}
    )
    assert result["coverage_complete"] is True
    assert result["raw"]["squared_norm"] == pytest.approx(25.0)
    assert result["applied"]["squared_norm"] == pytest.approx(0.25)
    assert result["raw_to_applied_cosine"] == pytest.approx(1.0)
