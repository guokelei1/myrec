"""Scoped history-to-readout attention edge interventions for Qwen3."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


EDGE_MODES = (
    "zero_additive_delta",
    "history_logits_mask",
    "history_value_edge_zero",
    "mask_then_restore_output",
)


class QwenAttentionEdgeIntervention:
    """Change only one block's registered readout-query/history-key edge.

    The active Transformers backend still computes the baseline output.  A
    project-owned selected-row calculation supplies the exact intervention
    delta; every unregistered query row and every other block remains the
    native backend output.
    """

    def __init__(self, model: Any, block: int, mode: str) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("attention edge block must be in [0, 27]")
        if mode not in EDGE_MODES:
            raise ValueError(f"unsupported attention edge mode={mode}")
        self.model = model
        self.backbone = resolve_qwen_backbone(model)
        self.block = block
        self.mode = mode
        self.interface: Any = None
        self.original_function: Any = None
        self.original_implementation: str | None = None
        self.original_key_present = False
        self.positions: Any = None
        self.history_starts: Any = None
        self.history_ends: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> "QwenAttentionEdgeIntervention":
        if self._active:
            raise RuntimeError("attention edge intervention is already active")
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
        history_starts: Any,
        history_ends: Any,
        *,
        sequence_length: int,
    ) -> None:
        if not self._active or self.positions is not None:
            raise RuntimeError("attention edge intervention cannot be armed")
        if readout_positions.ndim == 1:
            readout_positions = readout_positions[:, None]
        if (
            readout_positions.ndim != 2
            or readout_positions.shape[1] <= 0
            or history_starts.ndim != 1
            or history_ends.ndim != 1
            or readout_positions.shape[0] != history_starts.shape[0]
            or history_starts.shape != history_ends.shape
        ):
            raise ValueError("attention edge positions/spans are not batch-aligned")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("attention edge sequence length must be positive")
        if int(history_starts.min()) < 0 or int(history_ends.max()) > sequence_length:
            raise ValueError("attention history span is outside sequence")
        if bool((history_ends <= history_starts).any()):
            raise ValueError("attention history span is empty")
        if int(readout_positions.min()) < 0 or int(readout_positions.max()) >= sequence_length:
            raise ValueError("attention readout position is outside sequence")
        if bool((history_ends > readout_positions.min(dim=1).values).any()):
            # end is exclusive, so end <= readout is strictly before readout.
            raise ValueError("attention history span is not before readout")
        self.positions = readout_positions
        self.history_starts = history_starts
        self.history_ends = history_ends
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("attention edge intervention is not armed")
        if self.fire_count != 1:
            raise RuntimeError(
                f"attention edge registered block fired {self.fire_count} times"
            )
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _clear(self) -> None:
        self.positions = None
        self.history_starts = None
        self.history_ends = None
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
        if self.positions is None or self.history_starts is None or self.history_ends is None:
            raise RuntimeError("registered attention block fired while unarmed")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("attention edge wrapper expected rank-four Q/K/V")
        if query.shape[0] != self.positions.shape[0] or query.shape[2] != key.shape[2]:
            raise ValueError("attention edge wrapper requires non-cache full sequences")
        if int(query.shape[2]) != self.sequence_length:
            raise ValueError("attention edge runtime sequence length drifted")
        if self.mode == "zero_additive_delta":
            self.fire_count += 1
            self.last_summary = {
                "mode": self.mode,
                "rows": int(query.shape[0]),
                "maximum_applied_delta": 0.0,
                "manual_baseline_native_max_abs_error": None,
            }
            return baseline_output, baseline_weights

        torch = _torch()
        rows = torch.arange(query.shape[0], device=query.device)[:, None]
        positions = self.positions.to(query.device)
        selected_query = query.transpose(1, 2)[rows, positions]
        repeated_key, repeated_value = _repeat_kv_for_query_heads(module, key, value)
        logits = torch.einsum("bphd,bhkd->bphk", selected_query, repeated_key)
        scaling = kwargs.get("scaling")
        if scaling is None:
            scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
        logits = logits * float(scaling)
        logits = _apply_selected_attention_mask(
            logits,
            attention_mask,
            rows,
            positions,
        )
        baseline_probabilities = torch.softmax(logits.float(), dim=-1).to(value.dtype)
        manual_baseline = torch.einsum(
            "bphk,bhkd->bphd", baseline_probabilities, repeated_value
        )
        history_selector = torch.zeros(
            (query.shape[0], key.shape[2]), dtype=torch.bool, device=query.device
        )
        for row in range(query.shape[0]):
            start = int(self.history_starts[row])
            end = int(self.history_ends[row])
            history_selector[row, start:end] = True
        if self.mode in {"history_logits_mask", "mask_then_restore_output"}:
            masked_logits = logits.masked_fill(
                history_selector[:, None, None, :], -torch.inf
            )
            probabilities = torch.softmax(masked_logits.float(), dim=-1).to(value.dtype)
            desired = torch.einsum(
                "bphk,bhkd->bphd", probabilities, repeated_value
            )
        elif self.mode == "history_value_edge_zero":
            contribution = torch.einsum(
                "bphk,bhkd->bphd",
                baseline_probabilities * history_selector[:, None, None, :],
                repeated_value,
            )
            desired = manual_baseline - contribution
        else:
            raise AssertionError(self.mode)
        delta = desired - manual_baseline
        native_selected = baseline_output[rows, positions]
        modified = baseline_output.clone()
        # Native attention output before the o-projection is [B,S,H,D].  The
        # restore control computes the registered masked edge and then restores
        # the exact native selected-row value.  It therefore validates span and
        # mask isolation instead of degenerating into a wrapper no-op.
        if self.mode != "mask_then_restore_output":
            modified[rows, positions] = (
                modified[rows, positions] + delta.to(modified.dtype)
            )
        else:
            modified[rows, positions] = native_selected
        manual_error = float(
            torch.max(torch.abs(native_selected.float() - manual_baseline.float())).item()
        )
        self.fire_count += 1
        self.last_summary = {
            "mode": self.mode,
            "rows": int(query.shape[0]),
            "query_positions_per_row": int(positions.shape[1]),
            "history_tokens": int(history_selector.sum().item()),
            "maximum_applied_delta": float(torch.max(torch.abs(delta.float())).item()),
            "maximum_returned_delta": (
                0.0
                if self.mode == "mask_then_restore_output"
                else float(torch.max(torch.abs(delta.float())).item())
            ),
            "manual_baseline_native_max_abs_error": manual_error,
        }
        return modified, baseline_weights


def _repeat_kv_for_query_heads(module: Any, key: Any, value: Any) -> tuple[Any, Any]:
    groups = int(getattr(module, "num_key_value_groups", 1))
    if groups <= 0 or key.shape[1] * groups <= 0:
        raise ValueError("attention GQA group count is invalid")
    if groups == 1:
        return key, value
    return (
        key.repeat_interleave(groups, dim=1),
        value.repeat_interleave(groups, dim=1),
    )


def _apply_selected_attention_mask(
    logits: Any,
    attention_mask: Any,
    rows: Any,
    positions: Any,
) -> Any:
    torch = _torch()
    if attention_mask is None:
        keys = torch.arange(logits.shape[-1], device=logits.device)
        return logits.masked_fill(
            keys[None, None, None, :] > positions[:, :, None, None], -torch.inf
        )
    if attention_mask.ndim != 4 or attention_mask.shape[0] != logits.shape[0]:
        raise ValueError("attention edge additive mask has unexpected shape")
    batch = rows.expand_as(positions)
    selected = attention_mask[batch, 0, positions, : logits.shape[-1]]
    if selected.dtype == torch.bool:
        return logits.masked_fill(~selected[:, :, None, :], -torch.inf)
    return logits + selected[:, :, None, :].to(logits.dtype)


def _torch() -> Any:
    import torch

    return torch
