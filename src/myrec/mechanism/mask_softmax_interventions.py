"""Causal-visibility and softmax controls for the registered N27 boundary."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.attention_edge_interventions import (
    _apply_selected_attention_mask,
    _repeat_kv_for_query_heads,
)
from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


MASK_SOFTMAX_MODES = (
    "identity",
    "prefix_history_swap",
    "candidate_visibility_swap",
    "temperature_half",
    "temperature_double",
)


def apply_visibility_mask(logits: Any, allowed_keys: Any) -> Any:
    """Mask a selected ``[batch,positions,heads,keys]`` logit tensor."""

    torch = _torch()
    if logits.ndim != 4 or allowed_keys.ndim != 3:
        raise ValueError("visibility logits/mask must have ranks 4 and 3")
    if tuple(logits.shape[:2]) != tuple(allowed_keys.shape[:2]) or logits.shape[-1] != allowed_keys.shape[-1]:
        raise ValueError("visibility mask does not align with selected logits")
    if allowed_keys.dtype is not torch.bool:
        allowed_keys = allowed_keys.to(dtype=torch.bool)
    if not bool(allowed_keys.any(dim=-1).all()):
        raise ValueError("visibility mask removes every key for at least one query")
    return logits.masked_fill(~allowed_keys[:, :, None, :], -torch.inf)


def apply_softmax_temperature(logits: Any, temperature: float) -> Any:
    """Apply temperature to finite logits while preserving masked ``-inf``."""

    torch = _torch()
    temperature = float(temperature)
    if not temperature > 0:
        raise ValueError("softmax temperature must be positive")
    if logits.ndim != 4 or not logits.is_floating_point():
        raise ValueError("softmax logits must be a floating rank-four tensor")
    finite = torch.isfinite(logits)
    transformed = torch.where(finite, logits * temperature, logits)
    return transformed


class QwenMaskSoftmaxIntervention:
    """Change selected attention visibility or softmax temperature only."""

    def __init__(self, model: Any, block: int, mode: str) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("mask/softmax block must be in [0, 27]")
        if mode not in MASK_SOFTMAX_MODES:
            raise ValueError(f"unsupported mask/softmax mode={mode}")
        self.backbone = resolve_qwen_backbone(model)
        self.block = block
        self.mode = mode
        self.interface: Any = None
        self.original_function: Any = None
        self.original_implementation: str | None = None
        self.original_key_present = False
        self.positions: Any = None
        self.allowed_keys: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> "QwenMaskSoftmaxIntervention":
        if self._active:
            raise RuntimeError("mask/softmax intervention is already active")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        implementations = {
            str(layer.self_attn.config._attn_implementation)
            for layer in self.backbone.layers
        }
        if len(implementations) != 1:
            raise ValueError("Qwen layers do not share one attention backend")
        implementation = next(iter(implementations))
        self.interface = ALL_ATTENTION_FUNCTIONS
        self.original_implementation = implementation
        self.original_key_present = implementation in self.interface
        self.original_function = self.interface.get_interface(
            implementation, eager_attention_forward
        )
        self.interface[implementation] = self._wrapper
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active and self.interface is not None and self.original_implementation:
            if self.original_key_present:
                self.interface[self.original_implementation] = self.original_function
            elif self.original_implementation in self.interface:
                del self.interface[self.original_implementation]
        self._active = False
        self.interface = None
        self.original_function = None
        self.original_implementation = None
        self.original_key_present = False
        self._clear()

    def arm(
        self,
        readout_positions: Any,
        *,
        allowed_keys: Any | None,
        sequence_length: int,
    ) -> None:
        if not self._active or self.positions is not None:
            raise RuntimeError("mask/softmax intervention cannot be armed")
        if readout_positions.ndim == 1:
            readout_positions = readout_positions[:, None]
        if readout_positions.ndim != 2 or readout_positions.shape[1] <= 0:
            raise ValueError("mask/softmax positions must be [batch,positive_count]")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("mask/softmax sequence length must be positive")
        if int(readout_positions.min()) < 0 or int(readout_positions.max()) >= sequence_length:
            raise ValueError("mask/softmax position is outside sequence")
        if self.mode in {"prefix_history_swap", "candidate_visibility_swap"}:
            if allowed_keys is None or allowed_keys.ndim != 3:
                raise ValueError("visibility modes require [batch,positions,keys] allowed_keys")
            if tuple(allowed_keys.shape[:2]) != tuple(readout_positions.shape):
                raise ValueError("visibility mask and query positions are not aligned")
            if int(allowed_keys.shape[-1]) != sequence_length:
                raise ValueError("visibility mask sequence length drifted")
            # Future-token visibility would make the cell mechanically invalid.
            for row in range(int(readout_positions.shape[0])):
                for position_index in range(int(readout_positions.shape[1])):
                    query_position = int(readout_positions[row, position_index])
                    if bool(allowed_keys[row, position_index, query_position + 1 :].any()):
                        raise ValueError("visibility mask exposes a future token")
        elif allowed_keys is not None:
            raise ValueError("allowed_keys is only valid for visibility modes")
        self.positions = readout_positions
        self.allowed_keys = None if allowed_keys is None else allowed_keys.to(dtype=_torch().bool)
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None or self.fire_count != 1:
            raise RuntimeError(
                f"mask/softmax registered block fired {self.fire_count} times; expected one"
            )
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _clear(self) -> None:
        self.positions = None
        self.allowed_keys = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}

    def _wrapper(
        self,
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        assert self.original_function is not None
        baseline_output, baseline_weights = self.original_function(
            module, query, key, value, attention_mask, **kwargs
        )
        if int(module.layer_idx) != self.block:
            return baseline_output, baseline_weights
        if self.positions is None or self.sequence_length is None:
            raise RuntimeError("registered mask/softmax block fired while unarmed")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("mask/softmax wrapper expects rank-four Q/K/V")
        if query.shape[0] != self.positions.shape[0] or query.shape[2] != self.sequence_length:
            raise ValueError("mask/softmax wrapper requires non-cache full sequences")
        self.fire_count += 1
        if self.mode == "identity":
            self.last_summary = {"mode": self.mode, "maximum_applied_delta": 0.0}
            return baseline_output, baseline_weights

        torch = _torch()
        rows = torch.arange(query.shape[0], device=query.device)[:, None]
        positions = self.positions.to(query.device)
        selected_query = query.transpose(1, 2)[rows, positions]
        repeated_key, repeated_value = _repeat_kv_for_query_heads(module, key, value)
        raw_logits = torch.einsum("bphd,bhkd->bphk", selected_query, repeated_key)
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
        native_logits = _apply_selected_attention_mask(
            raw_logits * float(scaling), attention_mask, rows, positions
        )
        transformed_logits = native_logits
        if self.mode in {"prefix_history_swap", "candidate_visibility_swap"}:
            assert self.allowed_keys is not None
            allowed = self.allowed_keys.to(device=query.device)
            native_allowed = torch.isfinite(native_logits).any(dim=2)
            if bool((allowed & ~native_allowed).any()):
                raise RuntimeError("visibility mode enables a key rejected by native causal mask")
            transformed_logits = apply_visibility_mask(native_logits, allowed)
        elif self.mode == "temperature_half":
            transformed_logits = apply_softmax_temperature(native_logits, 0.5)
        elif self.mode == "temperature_double":
            transformed_logits = apply_softmax_temperature(native_logits, 2.0)
        else:  # pragma: no cover - constructor validates modes
            raise AssertionError(self.mode)
        native_probabilities = torch.softmax(native_logits.float(), dim=-1).to(value.dtype)
        transformed_probabilities = torch.softmax(transformed_logits.float(), dim=-1).to(value.dtype)
        manual_native = torch.einsum("bphk,bhkd->bphd", native_probabilities, repeated_value)
        desired = torch.einsum("bphk,bhkd->bphd", transformed_probabilities, repeated_value)
        native_selected = baseline_output[rows, positions]
        delta = desired - manual_native
        modified = baseline_output.clone()
        modified[rows, positions] = native_selected + delta.to(native_selected.dtype)
        self.last_summary = {
            "mode": self.mode,
            "query_positions_per_row": int(positions.shape[1]),
            "maximum_applied_delta": float(torch.max(torch.abs(delta.float())).item()),
            "manual_baseline_native_max_abs_error": float(
                torch.max(torch.abs(native_selected.float() - manual_native.float())).item()
            ),
            "future_visibility_checked": self.mode in {"prefix_history_swap", "candidate_visibility_swap"},
        }
        return modified, baseline_weights


def _torch() -> Any:
    import torch

    return torch
