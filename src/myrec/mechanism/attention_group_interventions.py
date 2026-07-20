"""GQA-group-scoped attention edge and history K/V interventions."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.attention_edge_interventions import (
    QwenAttentionEdgeIntervention,
    _apply_selected_attention_mask,
    _repeat_kv_for_query_heads,
)
from myrec.mechanism.history_kv_interventions import QwenHistoryKVIntervention


GQA_GROUPS = 8
QUERY_HEADS_PER_GQA = 2


class QwenAttentionGQAIntervention(QwenAttentionEdgeIntervention):
    """Apply one registered attention-edge intervention to one GQA group."""

    def __init__(self, model: Any, block: int, mode: str, gqa_group: int) -> None:
        super().__init__(model, block, mode)
        self.gqa_group = int(gqa_group)
        if not 0 <= self.gqa_group < GQA_GROUPS:
            raise ValueError("attention GQA group must be in [0,7]")
        attention = self.backbone.layers[self.block].self_attn
        if (
            int(getattr(attention.config, "num_key_value_heads", -1)) != GQA_GROUPS
            or int(getattr(attention, "num_key_value_groups", -1))
            != QUERY_HEADS_PER_GQA
        ):
            raise ValueError("attention GQA topology differs from frozen 16/8 design")

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
            raise RuntimeError("registered attention GQA block fired while unarmed")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("attention GQA wrapper expected rank-four Q/K/V")
        if query.shape[0] != self.positions.shape[0] or query.shape[2] != key.shape[2]:
            raise ValueError("attention GQA wrapper requires non-cache full sequences")
        if int(query.shape[2]) != self.sequence_length:
            raise ValueError("attention GQA sequence length drifted")
        if query.shape[1] != GQA_GROUPS * QUERY_HEADS_PER_GQA:
            raise ValueError("attention query-head count differs from frozen design")
        if self.mode == "zero_additive_delta":
            self.fire_count += 1
            self.last_summary = {
                "mode": self.mode,
                "gqa_group": self.gqa_group,
                "query_heads": self._query_heads(),
                "maximum_applied_delta": 0.0,
                "maximum_returned_delta": 0.0,
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
        logits = _apply_selected_attention_mask(
            logits * float(scaling), attention_mask, rows, positions
        )
        baseline_probabilities = torch.softmax(logits.float(), dim=-1).to(value.dtype)
        manual_baseline = torch.einsum(
            "bphk,bhkd->bphd", baseline_probabilities, repeated_value
        )
        history_selector = torch.zeros(
            (query.shape[0], key.shape[2]), dtype=torch.bool, device=query.device
        )
        for row in range(query.shape[0]):
            history_selector[
                row, int(self.history_starts[row]) : int(self.history_ends[row])
            ] = True
        if self.mode in {"history_logits_mask", "mask_then_restore_output"}:
            probabilities = torch.softmax(
                logits.masked_fill(
                    history_selector[:, None, None, :], -torch.inf
                ).float(),
                dim=-1,
            ).to(value.dtype)
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
        head_start = self.gqa_group * QUERY_HEADS_PER_GQA
        selector = torch.zeros(
            (1, 1, query.shape[1], 1), dtype=torch.bool, device=query.device
        )
        selector[:, :, head_start : head_start + QUERY_HEADS_PER_GQA] = True
        delta = torch.where(selector, delta, torch.zeros_like(delta))
        native_selected = baseline_output[rows, positions]
        modified = baseline_output.clone()
        if self.mode != "mask_then_restore_output":
            modified[rows, positions] = native_selected + delta.to(modified.dtype)
        else:
            modified[rows, positions] = native_selected
        manual_error = float(
            torch.max(torch.abs(native_selected.float() - manual_baseline.float())).item()
        )
        self.fire_count += 1
        self.last_summary = {
            "mode": self.mode,
            "gqa_group": self.gqa_group,
            "query_heads": self._query_heads(),
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

    def _query_heads(self) -> list[int]:
        start = self.gqa_group * QUERY_HEADS_PER_GQA
        return [start, start + 1]


class QwenHistoryKVGroupIntervention(QwenHistoryKVIntervention):
    """Patch exactly one of eight post-RoPE history K/V heads."""

    def __init__(self, model: Any, block: int, gqa_group: int) -> None:
        super().__init__(model, block)
        self.gqa_group = int(gqa_group)
        if not 0 <= self.gqa_group < GQA_GROUPS:
            raise ValueError("history K/V GQA group must be in [0,7]")

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
        if int(module.layer_idx) != self.block:
            return self.original_function(
                module, query, key, value, attention_mask, **kwargs
            )
        if self.mode is None or self.starts is None or self.ends is None:
            raise RuntimeError("registered history K/V GQA block fired while unarmed")
        if (
            key.ndim != 4
            or value.shape != key.shape
            or key.shape[0] != len(self.starts)
            or key.shape[1] != GQA_GROUPS
        ):
            raise ValueError("history K/V GQA tensor shape mismatch")
        if int(self.ends.max()) > key.shape[2]:
            raise ValueError("history K/V GQA span is outside runtime sequence")
        if self.mode == "capture":
            self.captured_keys = tuple(
                key[row, :, int(self.starts[row]) : int(self.ends[row])]
                .detach()
                .clone()
                for row in range(key.shape[0])
            )
            self.captured_values = tuple(
                value[row, :, int(self.starts[row]) : int(self.ends[row])]
                .detach()
                .clone()
                for row in range(value.shape[0])
            )
            modified_key, modified_value = key, value
        elif self.mode == "patch":
            assert self.donor_keys is not None and self.donor_values is not None
            modified_key = key.clone()
            modified_value = value.clone()
            for row in range(key.shape[0]):
                start, end = int(self.starts[row]), int(self.ends[row])
                expected = key[row, :, start:end].shape
                donor_key = self.donor_keys[row].to(key.device, dtype=key.dtype)
                donor_value = self.donor_values[row].to(value.device, dtype=value.dtype)
                if donor_key.shape != expected or donor_value.shape != expected:
                    raise ValueError("history K/V GQA donor shape differs")
                modified_key[row, self.gqa_group, start:end] = donor_key[
                    self.gqa_group
                ]
                modified_value[row, self.gqa_group, start:end] = donor_value[
                    self.gqa_group
                ]
        else:
            raise AssertionError(self.mode)
        self.fire_count += 1
        return self.original_function(
            module,
            query,
            modified_key,
            modified_value,
            attention_mask,
            **kwargs,
        )


def _torch() -> Any:
    import torch

    return torch
