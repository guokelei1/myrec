"""Factorial attention--MLP interaction bookkeeping for N29."""

from __future__ import annotations

from typing import Any, Mapping


FACTORIAL_CELLS = (
    "native_native",
    "removed_native",
    "native_removed",
    "removed_removed",
    "matched_scale",
    "sign_flip",
)


def factorial_interaction(
    cells: Mapping[str, Any],
    *,
    endpoint_name: str = "score",
) -> Any:
    """Compute ``both_removed - attention_only - mlp_only + native``.

    ``removed_native`` means attention removed and MLP native; ``native_removed``
    means attention native and MLP removed.  The output has the same shape as a
    cell and is never reduced, so the shared evaluator can cluster at request
    level before inference.
    """

    required = FACTORIAL_CELLS[:4]
    missing = [name for name in required if name not in cells]
    if missing:
        raise ValueError(f"factorial cells missing for {endpoint_name}: {missing}")
    native = _as_finite(cells["native_native"], "native_native")
    removed_native = _as_finite(cells["removed_native"], "removed_native")
    native_removed = _as_finite(cells["native_removed"], "native_removed")
    removed_removed = _as_finite(cells["removed_removed"], "removed_removed")
    if tuple(native.shape) != tuple(removed_native.shape) or tuple(native.shape) != tuple(native_removed.shape) or tuple(native.shape) != tuple(removed_removed.shape):
        raise ValueError("factorial cell shapes differ")
    return removed_removed - removed_native - native_removed + native


def summarize_factorial_cells(cells: Mapping[str, Any]) -> dict[str, Any]:
    """Return a machine-auditable finite/shape summary without opening qrels."""

    observed = []
    shapes: dict[str, list[int]] = {}
    for name in FACTORIAL_CELLS:
        if name not in cells:
            continue
        value = _as_finite(cells[name], name)
        observed.append(name)
        shapes[name] = [int(v) for v in value.shape]
    return {
        "observed_cells": observed,
        "all_factorial_cells_present": set(observed) == set(FACTORIAL_CELLS),
        "shapes": shapes,
    }


def _as_finite(value: Any, name: str) -> Any:
    import torch

    tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f"factorial cell is non-finite or non-floating: {name}")
    return tensor
