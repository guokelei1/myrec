"""Exact same-length history-span K/V capture and transplantation for Qwen3."""

from __future__ import annotations

from typing import Any, Sequence

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


class QwenHistoryKVIntervention:
    """Capture or replace one block's post-RoPE history K/V span."""

    def __init__(self, model: Any, block: int) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("history K/V block must be in [0, 27]")
        self.model = model
        self.backbone = resolve_qwen_backbone(model)
        self.block = block
        self.interface: Any = None
        self.original_function: Any = None
        self.original_implementation: str | None = None
        self.original_key_present = False
        self.starts: Any = None
        self.ends: Any = None
        self.mode: str | None = None
        self.donor_keys: tuple[Any, ...] | None = None
        self.donor_values: tuple[Any, ...] | None = None
        self.fire_count = 0
        self.captured_keys: tuple[Any, ...] | None = None
        self.captured_values: tuple[Any, ...] | None = None
        self._active = False

    def __enter__(self) -> "QwenHistoryKVIntervention":
        if self._active:
            raise RuntimeError("history K/V intervention is already active")
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

    def arm_capture(self, starts: Any, ends: Any) -> None:
        self._arm(starts, ends, mode="capture")

    def arm_patch(
        self,
        starts: Any,
        ends: Any,
        donor_keys: Sequence[Any],
        donor_values: Sequence[Any],
    ) -> None:
        self._arm(starts, ends, mode="patch")
        if len(donor_keys) != len(starts) or len(donor_values) != len(starts):
            raise ValueError("history K/V donor count differs from batch")
        self.donor_keys = tuple(donor_keys)
        self.donor_values = tuple(donor_values)

    def disarm_capture(self) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        if self.mode != "capture" or self.fire_count != 1:
            raise RuntimeError("history K/V capture did not fire exactly once")
        assert self.captured_keys is not None and self.captured_values is not None
        result = self.captured_keys, self.captured_values
        self._clear()
        return result

    def disarm_patch(self) -> None:
        if self.mode != "patch" or self.fire_count != 1:
            raise RuntimeError("history K/V patch did not fire exactly once")
        self._clear()

    def _arm(self, starts: Any, ends: Any, *, mode: str) -> None:
        if not self._active or self.mode is not None:
            raise RuntimeError("history K/V intervention cannot be armed")
        if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
            raise ValueError("history K/V spans must be aligned rank-one arrays")
        if int(starts.min()) < 0 or bool((ends <= starts).any()):
            raise ValueError("history K/V span is invalid")
        lengths = ends - starts
        if bool((lengths != lengths[0]).any()):
            raise ValueError("history K/V batch spans must have equal lengths")
        self.starts = starts
        self.ends = ends
        self.mode = mode
        self.fire_count = 0
        self.captured_keys = None
        self.captured_values = None

    def _clear(self) -> None:
        self.starts = None
        self.ends = None
        self.mode = None
        self.donor_keys = None
        self.donor_values = None
        self.fire_count = 0
        self.captured_keys = None
        self.captured_values = None

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
            raise RuntimeError("registered history K/V block fired while unarmed")
        if key.ndim != 4 or value.shape != key.shape or key.shape[0] != len(self.starts):
            raise ValueError("history K/V runtime tensor shape mismatch")
        if int(self.ends.max()) > key.shape[2]:
            raise ValueError("history K/V span is outside runtime sequence")
        if self.mode == "capture":
            self.captured_keys = tuple(
                key[row, :, int(self.starts[row]) : int(self.ends[row])].detach().clone()
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
                    raise ValueError("history K/V donor shape differs from recipient span")
                modified_key[row, :, start:end] = donor_key
                modified_value[row, :, start:end] = donor_value
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

