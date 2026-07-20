from __future__ import annotations

import pytest

from myrec.mechanism.lora_path_runtime import (
    _low_rank_frobenius_norm,
    _low_rank_function_cosine,
    _orthogonal_gauge_error,
)


def test_low_rank_function_geometry_matches_explicit_delta_w():
    torch = pytest.importorskip("torch")
    a1 = torch.randn(3, 5, dtype=torch.float64)
    b1 = torch.randn(7, 3, dtype=torch.float64)
    a2 = torch.randn(3, 5, dtype=torch.float64)
    b2 = torch.randn(7, 3, dtype=torch.float64)
    w1 = 2.0 * (b1 @ a1)
    w2 = 2.0 * (b2 @ a2)
    assert _low_rank_frobenius_norm(a1, b1, scaling=2.0) == pytest.approx(
        w1.norm().item(), rel=1e-12
    )
    expected = torch.nn.functional.cosine_similarity(w1.reshape(1, -1), w2.reshape(1, -1)).item()
    assert _low_rank_function_cosine(a1, b1, a2, b2) == pytest.approx(
        expected, abs=1e-12
    )
    assert _orthogonal_gauge_error(a1, b1, "block_00.q_proj") < 1e-12


def test_zero_lora_function_has_undefined_direction():
    torch = pytest.importorskip("torch")
    a = torch.randn(3, 5, dtype=torch.float64)
    b = torch.zeros(7, 3, dtype=torch.float64)
    assert _low_rank_frobenius_norm(a, b, scaling=2.0) == 0.0
    assert _low_rank_function_cosine(a, b, a, b) is None
