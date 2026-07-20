"""Pure diagnostics for the preregistered N21--N24 training boundaries.

The functions in this module deliberately do not load a model, read qrels, or
perform an optimizer step.  They make the quantities that the future matched
training controls must record explicit and testable: mixed-dtype recomposition
residuals, replayable LoRA dropout masks, gradient-path coverage, and effective
update accounting.  A caller can therefore attach these measurements to the
existing train-only runtime without silently changing the frozen recipe.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from myrec.mechanism.optimizer_replay_math import (
    gradient_pair_summary,
    vector_cosine,
    vector_summary,
)


def dtype_cast_boundary_summary(
    native: Any,
    reference: Any,
    *,
    label: str = "native",
) -> dict[str, Any]:
    """Summarize the numerical residual between a native and reference path.

    ``native`` and ``reference`` may have different dtypes, but must represent
    the same tensor shape.  The comparison is made in float64 so that the
    reported residual is not itself hidden by the low-precision output dtype.
    """

    _check_tensor_pair(native, reference, label)
    torch = _torch()
    left = native.detach().double()
    right = reference.detach().double()
    residual = left - right
    native_norm = float(left.norm().item())
    reference_norm = float(right.norm().item())
    residual_norm = float(residual.norm().item())
    cosine = None
    if native_norm > 0.0 and reference_norm > 0.0:
        cosine = float(torch.sum(left * right).item() / (native_norm * reference_norm))
        cosine = max(-1.0, min(1.0, cosine))
    return {
        "label": str(label),
        "shape": list(native.shape),
        "native_dtype": str(native.dtype),
        "reference_dtype": str(reference.dtype),
        "maximum_absolute_residual": float(residual.abs().max().item()),
        "residual_l2": residual_norm,
        "relative_l2_residual": residual_norm / reference_norm
        if reference_norm > 0.0
        else 0.0,
        "native_l2": native_norm,
        "reference_l2": reference_norm,
        "direction_cosine": cosine,
        "finite": True,
    }


def summarize_cast_variants(
    variants: Mapping[str, Any],
    reference_name: str,
) -> dict[str, Any]:
    """Compare several N21 forward variants to one predeclared reference."""

    if not variants:
        raise ValueError("cast variants cannot be empty")
    reference_name = str(reference_name)
    if reference_name not in variants:
        raise KeyError(f"missing cast reference variant={reference_name}")
    reference = variants[reference_name]
    summaries = {
        str(name): dtype_cast_boundary_summary(value, reference, label=str(name))
        for name, value in variants.items()
    }
    return {
        "reference": reference_name,
        "variants": summaries,
        "variant_names": sorted(str(name) for name in variants),
    }


def lora_dropout_forward(
    values: Any,
    probability: float,
    *,
    mask: Any | None = None,
    generator: Any | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply inverted LoRA input dropout with optional fixed-mask replay.

    The returned mask is boolean and can be passed back to this function to
    replay the exact stochastic path.  Supplying both ``mask`` and
    ``generator`` is rejected so that a recorded replay cannot accidentally be
    replaced by a fresh random draw.
    """

    probability = float(probability)
    if not math.isfinite(probability) or not 0.0 <= probability < 1.0:
        raise ValueError("dropout probability must be finite and in [0, 1)")
    _check_finite_tensor(values, "dropout values")
    torch = _torch()
    if mask is not None and generator is not None:
        raise ValueError("dropout replay accepts mask or generator, not both")
    if mask is None:
        if probability == 0.0:
            mask = torch.ones_like(values, dtype=torch.bool)
        else:
            mask = torch.rand(
                values.shape,
                device=values.device,
                generator=generator,
                dtype=torch.float32,
            ) >= probability
    else:
        if not hasattr(mask, "shape") or tuple(mask.shape) != tuple(values.shape):
            raise ValueError("dropout replay mask shape differs from values")
        if mask.device != values.device:
            raise ValueError("dropout replay mask device differs from values")
        mask = mask.to(dtype=torch.bool)
    keep_scale = 1.0 / (1.0 - probability)
    output = values * mask.to(dtype=values.dtype) * keep_scale
    _check_finite_tensor(output, "dropout output")
    return output, {
        "probability": probability,
        "keep_scale": keep_scale,
        "mask_shape": list(mask.shape),
        "kept_fraction": float(mask.float().mean().item()),
        "mask_replayable": True,
        "finite": True,
    }


def gradient_path_summary(
    names: Sequence[str],
    native: Sequence[Any],
    reference: Sequence[Any],
    *,
    family_by_name: Mapping[str, str],
) -> dict[str, Any]:
    """Audit gradient coverage and alignment for N23 bridge/recompute paths."""

    if len(names) != len(native) or len(names) != len(reference):
        raise ValueError("gradient path sequences have different lengths")
    if set(names) != set(family_by_name):
        raise ValueError("gradient path family partition differs")
    missing_native = [str(name) for name, value in zip(names, native) if value is None]
    missing_reference = [
        str(name) for name, value in zip(names, reference) if value is None
    ]
    if missing_native or missing_reference:
        return {
            "complete": False,
            "parameter_count": len(names),
            "missing_native": missing_native,
            "missing_reference": missing_reference,
            "cosine": None,
        }
    pair = gradient_pair_summary(
        names,
        native,
        reference,
        family_by_name=family_by_name,
    )
    return {
        "complete": True,
        "parameter_count": len(names),
        "missing_native": [],
        "missing_reference": [],
        **pair,
    }


def effective_update_summary(
    raw_gradients: Mapping[str, Any],
    applied_updates: Mapping[str, Any],
    *,
    family_by_name: Mapping[str, str],
) -> dict[str, Any]:
    """Compare raw objective pressure with the update actually applied.

    This is intentionally optimizer-agnostic: the caller supplies the exact
    post-clip/AdamW/scheduler delta produced by its frozen training runtime.
    The function only summarizes the two vectors and their family partition.
    """

    if set(raw_gradients) != set(applied_updates):
        raise ValueError("raw gradients and applied updates have different coverage")
    if set(raw_gradients) != set(family_by_name):
        raise ValueError("effective update family partition differs")
    for name in raw_gradients:
        left, right = raw_gradients[name], applied_updates[name]
        if left is None or right is None:
            if left is not None or right is not None:
                raise ValueError(f"effective update active mismatch: {name}")
            continue
        if tuple(left.shape) != tuple(right.shape):
            raise ValueError(f"effective update shape mismatch: {name}")
        _check_finite_tensor(left, f"raw gradient {name}")
        _check_finite_tensor(right, f"applied update {name}")
    return {
        "raw": vector_summary(raw_gradients, family_by_name=family_by_name),
        "applied": vector_summary(applied_updates, family_by_name=family_by_name),
        "raw_to_applied_cosine": vector_cosine(raw_gradients, applied_updates),
        "coverage_complete": True,
    }


def _check_tensor_pair(left: Any, right: Any, label: str) -> None:
    if not hasattr(left, "shape") or not hasattr(right, "shape"):
        raise TypeError(f"{label} values must be tensors")
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"{label} tensor shapes differ")
    _check_finite_tensor(left, f"{label} native")
    _check_finite_tensor(right, f"{label} reference")


def _check_finite_tensor(value: Any, label: str) -> None:
    torch = _torch()
    if not bool(torch.isfinite(value).all().item()):
        raise FloatingPointError(f"{label} contains non-finite values")


def _torch() -> Any:
    import torch

    return torch
