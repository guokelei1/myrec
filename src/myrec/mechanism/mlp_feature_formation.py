"""Selected-token SwiGLU gate/up/product observation and exact delta algebra."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from myrec.mechanism.mlp_group_interventions import frozen_mlp_groups
from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


MLP_FEATURE_STAGES = ("gate_pre", "gate_activated", "up", "product")


class QwenMLPFeatureObserver:
    """Capture four SwiGLU formation stages at registered token positions."""

    def __init__(self, model: Any, block: int) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("MLP feature block must be in [0,27]")
        self.layer = resolve_qwen_backbone(model).layers[block]
        self.positions: Any = None
        self.captures: dict[str, Any] = {}
        self.fire_counts: dict[str, int] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "QwenMLPFeatureObserver":
        if self.handles:
            raise RuntimeError("MLP feature observer is already active")
        self.handles = [
            self.layer.mlp.gate_proj.register_forward_hook(self._gate_hook),
            self.layer.mlp.up_proj.register_forward_hook(self._up_hook),
            self.layer.mlp.down_proj.register_forward_pre_hook(self._product_hook),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._clear()

    def arm(self, positions: Any, *, sequence_length: int) -> None:
        if not self.handles or self.positions is not None:
            raise RuntimeError("MLP feature observer cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("MLP feature positions must be [batch,positive]")
        if int(positions.min()) < 0 or int(positions.max()) >= int(sequence_length):
            raise ValueError("MLP feature position is outside sequence")
        self.positions = positions
        self.captures = {}
        self.fire_counts = {"gate": 0, "up": 0, "product": 0}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None or any(
            self.fire_counts.get(name) != 1 for name in ("gate", "up", "product")
        ):
            raise RuntimeError("MLP feature observer did not fire exactly once")
        if set(self.captures) != set(MLP_FEATURE_STAGES):
            raise RuntimeError("MLP feature stage coverage differs")
        reconstructed = (
            self.captures["gate_activated"] * self.captures["up"]
        )
        error = float(
            (
                reconstructed.float() - self.captures["product"].float()
            ).abs().max().item()
        )
        reference = float(self.captures["product"].float().abs().max().item())
        dtype = self.captures["product"].dtype
        bound = 4.0 * float(_torch().finfo(dtype).eps) * max(1.0, reference)
        result = {
            "captures": dict(self.captures),
            "product_recomposition_max_abs_error": error,
            "product_recomposition_low_precision_ratio": error / bound,
        }
        if result["product_recomposition_low_precision_ratio"] > 1.0:
            raise ValueError("MLP feature product recomposition exceeds precision bound")
        self._clear()
        return result

    def _select(self, tensor: Any) -> Any:
        if self.positions is None or tensor.ndim != 3:
            raise RuntimeError("MLP feature hook fired while unarmed")
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        return tensor[rows, self.positions.to(tensor.device)].detach()

    def _gate_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        gate = self._select(output)
        self.captures["gate_pre"] = gate
        self.captures["gate_activated"] = self.layer.mlp.act_fn(gate).detach()
        self.fire_counts["gate"] += 1

    def _up_hook(self, _module: Any, _inputs: Any, output: Any) -> None:
        self.captures["up"] = self._select(output)
        self.fire_counts["up"] += 1

    def _product_hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("MLP feature down projection has no input")
        self.captures["product"] = self._select(inputs[0])
        self.fire_counts["product"] += 1

    def _clear(self) -> None:
        self.positions = None
        self.captures = {}
        self.fire_counts = {}


def summarize_mlp_feature_pair(
    full: Mapping[str, Any],
    null: Mapping[str, Any],
    *,
    groups: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Return fixed-group stage geometry and exact SwiGLU delta terms."""

    if set(full) != set(MLP_FEATURE_STAGES) or set(null) != set(MLP_FEATURE_STAGES):
        raise ValueError("MLP feature pair stage coverage differs")
    shape = tuple(full["product"].shape)
    if len(shape) != 2 or any(tuple(full[name].shape) != shape for name in MLP_FEATURE_STAGES) or any(
        tuple(null[name].shape) != shape for name in MLP_FEATURE_STAGES
    ):
        raise ValueError("MLP feature pair shape differs")
    intermediate = shape[-1]
    normalized_groups = (
        tuple(tuple(int(index) for index in group) for group in groups)
        if groups is not None
        else frozen_mlp_groups(intermediate)
    )
    _validate_groups(normalized_groups, intermediate)
    torch = _torch()
    maximum_product_identity_ratio = 0.0
    full_float = {name: value.float() for name, value in full.items()}
    null_float = {name: value.float() for name, value in null.items()}
    for condition, source, values in (
        ("full", full, full_float),
        ("null", null, null_float),
    ):
        error = float(
            (
                values["product"]
                - values["gate_activated"] * values["up"]
            ).abs().max().item()
        )
        reference = float(values["product"].abs().max().item())
        bound = 4.0 * float(torch.finfo(source["product"].dtype).eps) * max(
            1.0, reference
        )
        ratio = error / bound
        maximum_product_identity_ratio = max(maximum_product_identity_ratio, ratio)
        if ratio > 1.0:
            raise ValueError(
                f"MLP feature {condition} product identity exceeds precision bound"
            )

    positions = []
    maximum_delta_recomposition_error = 0.0
    maximum_actual_product_quantization_error = 0.0
    for position in range(shape[0]):
        group_rows = []
        for group_id, group in enumerate(normalized_groups):
            indices = torch.tensor(group, dtype=torch.long, device=full["product"].device)
            stage_rows = {}
            for stage in MLP_FEATURE_STAGES:
                full_values = full_float[stage][position].index_select(0, indices)
                null_values = null_float[stage][position].index_select(0, indices)
                stage_rows[stage] = _stage_geometry(full_values, null_values)
            a_full = full_float["gate_activated"][position].index_select(0, indices)
            a_null = null_float["gate_activated"][position].index_select(0, indices)
            u_full = full_float["up"][position].index_select(0, indices)
            u_null = null_float["up"][position].index_select(0, indices)
            delta_a = a_full - a_null
            delta_u = u_full - u_null
            gate_term = delta_a * u_null
            up_term = a_null * delta_u
            interaction = delta_a * delta_u
            actual_product_delta = (
                full_float["product"][position].index_select(0, indices)
                - null_float["product"][position].index_select(0, indices)
            )
            algebraic_product_delta = a_full * u_full - a_null * u_null
            recomposed = gate_term + up_term + interaction
            error = float(
                (recomposed - algebraic_product_delta).abs().max().item()
            )
            quantization_error = float(
                (actual_product_delta - algebraic_product_delta).abs().max().item()
            )
            maximum_delta_recomposition_error = max(
                maximum_delta_recomposition_error, error
            )
            maximum_actual_product_quantization_error = max(
                maximum_actual_product_quantization_error, quantization_error
            )
            group_rows.append(
                {
                    "group_id": group_id,
                    "dimensions": len(group),
                    "stages": stage_rows,
                    "product_delta_decomposition": {
                        "gate_change_times_null_up": _term_geometry(
                            gate_term, algebraic_product_delta
                        ),
                        "null_gate_times_up_change": _term_geometry(
                            up_term, algebraic_product_delta
                        ),
                        "gate_up_interaction": _term_geometry(
                            interaction, algebraic_product_delta
                        ),
                        "actual_product_delta_rms": _rms(actual_product_delta),
                        "algebraic_product_delta_rms": _rms(
                            algebraic_product_delta
                        ),
                        "maximum_recomposition_abs_error": error,
                        "maximum_actual_product_quantization_abs_error": (
                            quantization_error
                        ),
                    },
                }
            )
        positions.append({"position_index": position, "groups": group_rows})
    return {
        "positions": positions,
        "groups": len(normalized_groups),
        "maximum_product_delta_recomposition_abs_error": (
            maximum_delta_recomposition_error
        ),
        "maximum_actual_product_quantization_abs_error": (
            maximum_actual_product_quantization_error
        ),
        "maximum_product_identity_low_precision_ratio": (
            maximum_product_identity_ratio
        ),
        "interpretation_boundary": (
            "Exact algebraic decomposition of the captured gate/up factors; the "
            "native low-precision product and its quantization residual are reported "
            "separately. This is not additive causal attribution or a selector."
        ),
    }


def _stage_geometry(full: Any, null: Any) -> dict[str, float]:
    denominator = float(full.norm().item() * null.norm().item())
    cosine = (
        float((full * null).sum().item()) / denominator if denominator > 0 else 0.0
    )
    return {
        "full_rms": _rms(full),
        "null_rms": _rms(null),
        "delta_rms": _rms(full - null),
        "full_null_cosine": max(-1.0, min(1.0, cosine)),
        "sign_flip_fraction": float((full * null < 0).float().mean().item()),
        "full_hoyer_sparsity": _hoyer(full),
        "null_hoyer_sparsity": _hoyer(null),
    }


def _term_geometry(term: Any, product_delta: Any) -> dict[str, float]:
    denominator = float(term.norm().item() * product_delta.norm().item())
    cosine = (
        float((term * product_delta).sum().item()) / denominator
        if denominator > 0
        else 0.0
    )
    return {
        "rms": _rms(term),
        "cosine_to_product_delta": max(-1.0, min(1.0, cosine)),
    }


def _rms(value: Any) -> float:
    result = float(value.square().mean().sqrt().item())
    if not math.isfinite(result):
        raise ValueError("MLP feature RMS is non-finite")
    return result


def _hoyer(value: Any) -> float:
    flat = value.reshape(-1)
    if flat.numel() <= 1:
        return 0.0
    l2 = float(flat.norm(p=2).item())
    if l2 == 0:
        return 1.0
    root = math.sqrt(flat.numel())
    result = (root - float(flat.norm(p=1).item()) / l2) / (root - 1.0)
    return max(0.0, min(1.0, result))


def _validate_groups(groups: Sequence[Sequence[int]], intermediate: int) -> None:
    if not groups or any(not group for group in groups):
        raise ValueError("MLP feature groups are empty")
    flattened = [int(index) for group in groups for index in group]
    if sorted(flattened) != list(range(intermediate)):
        raise ValueError("MLP feature groups do not partition dimensions")


def _torch() -> Any:
    import torch

    return torch
