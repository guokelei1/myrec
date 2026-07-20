"""Scoped scaled-QK-logit interventions for the Qwen attention backend.

The existing attention-edge probes change *which* history keys are visible.  This
module changes only the pre-mask scaled QK logits at selected readout rows,
leaving Q, K, V, the causal/additive mask, and the output projection untouched.
It is therefore a separate operator-level diagnostic rather than another edge
masking variant.
"""

from __future__ import annotations

from typing import Any

from myrec.mechanism.attention_edge_interventions import (
    _apply_selected_attention_mask,
    _repeat_kv_for_query_heads,
)
from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


LOGIT_MODES = (
    "identity",
    "scale_half",
    "scale_double",
    "sign_flip",
)


class QwenAttentionLogitIntervention:
    """Change only the scaled QK logits for selected query rows in one block."""

    def __init__(self, model: Any, block: int, mode: str) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("attention logit block must be in [0, 27]")
        if mode not in LOGIT_MODES:
            raise ValueError(f"unsupported attention logit mode={mode}")
        self.model = model
        self.backbone = resolve_qwen_backbone(model)
        self.block = block
        self.mode = mode
        self.interface: Any = None
        self.original_function: Any = None
        self.original_implementation: str | None = None
        self.original_key_present = False
        self.positions: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> "QwenAttentionLogitIntervention":
        if self._active:
            raise RuntimeError("attention logit intervention is already active")
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

    def arm(self, readout_positions: Any, *, sequence_length: int) -> None:
        if not self._active or self.positions is not None:
            raise RuntimeError("attention logit intervention cannot be armed")
        if readout_positions.ndim == 1:
            readout_positions = readout_positions[:, None]
        if readout_positions.ndim != 2 or readout_positions.shape[1] <= 0:
            raise ValueError("attention logit positions must be [batch, positive_count]")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("attention logit sequence length must be positive")
        if int(readout_positions.min()) < 0 or int(readout_positions.max()) >= sequence_length:
            raise ValueError("attention logit position is outside sequence")
        self.positions = readout_positions
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("attention logit intervention is not armed")
        if self.fire_count != 1:
            raise RuntimeError(
                f"attention logit registered block fired {self.fire_count} times"
            )
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _clear(self) -> None:
        self.positions = None
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
            raise RuntimeError("registered attention block fired while unarmed")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("attention logit wrapper expected rank-four Q/K/V")
        if query.shape[0] != self.positions.shape[0] or query.shape[2] != self.sequence_length:
            raise ValueError("attention logit wrapper requires non-cache full sequences")

        self.fire_count += 1
        if self.mode == "identity":
            self.last_summary = {
                "mode": self.mode,
                "rows": int(query.shape[0]),
                "query_positions_per_row": int(self.positions.shape[1]),
                "maximum_applied_delta": 0.0,
                "manual_baseline_native_max_abs_error": None,
            }
            return baseline_output, baseline_weights

        torch = _torch()
        rows = torch.arange(query.shape[0], device=query.device)[:, None]
        positions = self.positions.to(query.device)
        selected_query = query.transpose(1, 2)[rows, positions]
        repeated_key, repeated_value = _repeat_kv_for_query_heads(module, key, value)
        raw_logits = torch.einsum(
            "bphd,bhkd->bphk", selected_query, repeated_key
        )
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
        native_logits = raw_logits * float(scaling)
        if self.mode == "scale_half":
            transformed_logits = native_logits * 0.5
        elif self.mode == "scale_double":
            transformed_logits = native_logits * 2.0
        elif self.mode == "sign_flip":
            transformed_logits = native_logits * -1.0
        else:  # pragma: no cover - guarded by constructor
            raise AssertionError(self.mode)
        native_logits = _apply_selected_attention_mask(
            native_logits, attention_mask, rows, positions
        )
        transformed_logits = _apply_selected_attention_mask(
            transformed_logits, attention_mask, rows, positions
        )
        native_probabilities = torch.softmax(native_logits.float(), dim=-1).to(value.dtype)
        transformed_probabilities = torch.softmax(
            transformed_logits.float(), dim=-1
        ).to(value.dtype)
        manual_native = torch.einsum(
            "bphk,bhkd->bphd", native_probabilities, repeated_value
        )
        desired = torch.einsum(
            "bphk,bhkd->bphd", transformed_probabilities, repeated_value
        )
        native_selected = baseline_output[rows, positions]
        delta = desired - manual_native
        modified = baseline_output.clone()
        modified[rows, positions] = native_selected + delta.to(native_selected.dtype)
        self.last_summary = {
            "mode": self.mode,
            "rows": int(query.shape[0]),
            "query_positions_per_row": int(positions.shape[1]),
            "maximum_applied_delta": float(torch.max(torch.abs(delta.float())).item()),
            "manual_baseline_native_max_abs_error": float(
                torch.max(torch.abs(native_selected.float() - manual_native.float())).item()
            ),
        }
        return modified, baseline_weights


def _torch() -> Any:
    import torch

    return torch

