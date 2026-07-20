"""Scoped Qwen attention-interface instrumentation for the deep-dive stage."""

from __future__ import annotations

from typing import Any, Mapping

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


SCOPED_INTERFACE_KEY = "myrec_transformer_deep_dive_scoped_v1"


class QwenAttentionInterfaceAudit:
    """Delegate to the active backend while exposing selected post-RoPE rows.

    The wrapper is installed only in the local mapping of Transformers'
    ``ALL_ATTENTION_FUNCTIONS`` instance and is removed on exit.  It does not
    materialize a full attention matrix and does not alter Q/K/V or the mask.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.backbone = resolve_qwen_backbone(model)
        self.interface: Any = None
        self.original_function: Any = None
        self.original_implementation: str | None = None
        self.original_key_present = False
        self.configs: list[Any] = []
        self.positions: Any = None
        self.rows: Any = None
        self.maximum_position: int | None = None
        self.captured: dict[int, dict[str, Any]] = {}
        self.call_counts: dict[int, int] = {}
        self._active = False
        self._all_calls_mode = False
        self.all_call_shapes: dict[int, list[dict[str, Any]]] = {}

    def __enter__(self) -> "QwenAttentionInterfaceAudit":
        if self._active:
            raise RuntimeError("attention audit is already active")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        self.interface = ALL_ATTENTION_FUNCTIONS
        implementations = {
            str(layer.self_attn.config._attn_implementation)
            for layer in self.backbone.layers
        }
        if len(implementations) != 1:
            raise ValueError("Qwen layers do not share one attention implementation")
        implementation = next(iter(implementations))
        if implementation in {"", "None"}:
            raise ValueError("Qwen attention implementation is unresolved")
        self.original_implementation = implementation
        self.original_key_present = implementation in self.interface
        self.original_function = self.interface.get_interface(
            implementation,
            eager_attention_forward,
        )
        self.interface[implementation] = self._wrapper
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active:
            if self.interface is not None and self.original_implementation is not None:
                if self.original_key_present:
                    self.interface[self.original_implementation] = self.original_function
                elif self.original_implementation in self.interface:
                    del self.interface[self.original_implementation]
        self._active = False
        self.positions = None
        self.rows = None
        self.maximum_position = None
        self.captured = {}
        self.call_counts = {}
        self._all_calls_mode = False
        self.all_call_shapes = {}
        self.configs = []
        self.original_function = None
        self.original_key_present = False
        self.interface = None

    def arm(self, positions: Any, *, sequence_length: int) -> None:
        if not self._active:
            raise RuntimeError("attention audit is not active")
        if self.positions is not None:
            raise RuntimeError("attention audit is already armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("attention positions must be [batch, positive_count]")
        minimum = int(positions.min().item())
        maximum = int(positions.max().item())
        if minimum < 0 or maximum >= int(sequence_length):
            raise ValueError("attention audit position is outside sequence")
        self.positions = positions
        self.rows = _torch().arange(positions.shape[0], device=positions.device)[:, None]
        self.maximum_position = maximum
        self.captured = {}
        self.call_counts = {block: 0 for block in range(28)}

    def disarm(self) -> dict[int, dict[str, Any]]:
        if self.positions is None:
            raise RuntimeError("attention audit is not armed")
        invalid = {block: count for block, count in self.call_counts.items() if count != 1}
        if invalid:
            raise RuntimeError(f"attention interface call-count mismatch: {invalid}")
        result = dict(self.captured)
        self.positions = None
        self.rows = None
        self.maximum_position = None
        self.captured = {}
        self.call_counts = {}
        return result

    def arm_all_calls(self) -> None:
        """Audit every native cache/prefill call without selecting token rows."""

        if not self._active or self.positions is not None or self._all_calls_mode:
            raise RuntimeError("attention all-call audit cannot be armed")
        self._all_calls_mode = True
        self.call_counts = {block: 0 for block in range(28)}
        self.all_call_shapes = {block: [] for block in range(28)}

    def disarm_all_calls(self) -> dict[str, Any]:
        if not self._all_calls_mode:
            raise RuntimeError("attention all-call audit is not armed")
        counts = {block: self.call_counts[block] for block in range(28)}
        if not counts or len(set(counts.values())) != 1 or next(iter(counts.values())) <= 0:
            raise RuntimeError(f"attention all-call count mismatch: {counts}")
        reference = self.all_call_shapes[0]
        if any(self.all_call_shapes[block] != reference for block in range(1, 28)):
            raise RuntimeError("attention call shapes differ across blocks")
        result = {
            "calls_per_block": next(iter(counts.values())),
            "all_blocks_identical": True,
            "block_0_shapes": list(reference),
        }
        self._all_calls_mode = False
        self.call_counts = {}
        self.all_call_shapes = {}
        return result

    def _wrapper(
        self,
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        if self._all_calls_mode:
            block = int(module.layer_idx)
            if block not in self.call_counts:
                raise ValueError(f"unexpected attention layer_idx={block}")
            self.call_counts[block] += 1
            self.all_call_shapes[block].append(
                {
                    "query": [int(value) for value in query.shape],
                    "key": [int(value) for value in key.shape],
                    "value": [int(value) for value in value.shape],
                    "mask": (
                        None
                        if attention_mask is None
                        else [int(value) for value in attention_mask.shape]
                    ),
                }
            )
            assert self.original_function is not None
            return self.original_function(
                module, query, key, value, attention_mask, **kwargs
            )
        if self.positions is None or self.rows is None or self.maximum_position is None:
            raise RuntimeError("attention wrapper fired while unarmed")
        block = int(module.layer_idx)
        if block not in self.call_counts:
            raise ValueError(f"unexpected attention layer_idx={block}")
        if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
            raise ValueError("attention wrapper expected rank-four Q/K/V")
        if query.shape[0] != self.positions.shape[0] or query.shape[2] <= self.maximum_position:
            raise ValueError("attention query shape differs from armed positions")
        if key.shape[0] != query.shape[0] or value.shape[:3] != key.shape[:3]:
            raise ValueError("attention K/V shape mismatch")
        query_by_token = query.transpose(1, 2)
        key_by_token = key.transpose(1, 2)
        value_by_token = value.transpose(1, 2)
        selected_query = query_by_token[self.rows, self.positions].detach()
        selected_key = key_by_token[self.rows, self.positions].detach()
        selected_value = value_by_token[self.rows, self.positions].detach()
        self.captured[block] = {
            "post_rope_query": selected_query,
            "post_rope_key": selected_key,
            "value": selected_value,
            "query_shape": tuple(int(v) for v in query.shape),
            "key_shape": tuple(int(v) for v in key.shape),
            "value_shape": tuple(int(v) for v in value.shape),
            "attention_mask_shape": (
                None
                if attention_mask is None
                else tuple(int(v) for v in attention_mask.shape)
            ),
            "attention_mask_dtype": (
                None if attention_mask is None else str(attention_mask.dtype)
            ),
        }
        self.call_counts[block] += 1
        assert self.original_function is not None
        return self.original_function(
            module,
            query,
            key,
            value,
            attention_mask,
            **kwargs,
        )


def attention_audit_summary(captured: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    if set(captured) != set(range(28)):
        raise ValueError("attention audit summary requires all 28 blocks")
    return {
        "blocks": 28,
        "query_shapes": {
            str(block): list(captured[block]["query_shape"]) for block in range(28)
        },
        "key_shapes": {
            str(block): list(captured[block]["key_shape"]) for block in range(28)
        },
        "attention_mask_shapes": {
            str(block): (
                None
                if captured[block]["attention_mask_shape"] is None
                else list(captured[block]["attention_mask_shape"])
            )
            for block in range(28)
        },
    }


def _torch() -> Any:
    import torch

    return torch
