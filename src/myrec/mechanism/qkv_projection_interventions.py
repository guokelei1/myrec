"""Scoped Q/K/V projection-stage operators for Qwen attention."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


PROJECTION_COMPONENTS = ("q", "k", "v")
PROJECTION_MODES = ("identity", "scale_half", "scale_double", "sign_flip")
_MODE_SCALE = {
    "identity": 1.0,
    "scale_half": 0.5,
    "scale_double": 2.0,
    "sign_flip": -1.0,
}


class QwenQKVProjectionIntervention:
    """Scale or sign-flip one projection at registered rows in one block.

    Q is selected only at native readout rows. K and V are selected only over
    the fixed history span. The projection hook is before RoPE for Q/K and
    before the attention value transport for V; all unregistered rows remain
    native. ``identity`` deliberately returns the untouched tensor so its
    score is an exact no-op even under low-precision kernels.
    """

    def __init__(self, model: Any, block: int, component: str, mode: str) -> None:
        block = int(block)
        component = str(component)
        mode = str(mode)
        if not 0 <= block < 28:
            raise ValueError("QKV projection block must be in [0, 27]")
        if component not in PROJECTION_COMPONENTS:
            raise ValueError(f"unsupported QKV component={component}")
        if mode not in PROJECTION_MODES:
            raise ValueError(f"unsupported QKV mode={mode}")
        self.block = block
        self.component = component
        self.mode = mode
        self.layer = resolve_qwen_backbone(model).layers[block]
        self.module = getattr(self.layer.self_attn, f"{component}_proj")
        self.positions: Any = None
        self.history_starts: Any = None
        self.history_ends: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self.handle: Any = None

    def __enter__(self) -> "QwenQKVProjectionIntervention":
        if self.handle is not None:
            raise RuntimeError("QKV projection intervention is already active")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self._clear()

    def arm(
        self,
        readout_positions: Any,
        history_starts: Any,
        history_ends: Any,
        *,
        sequence_length: int,
    ) -> None:
        if self.handle is None or self.positions is not None:
            raise RuntimeError("QKV projection intervention cannot be armed")
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
            raise ValueError("QKV positions and history spans are not batch-aligned")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("QKV sequence length must be positive")
        if int(readout_positions.min()) < 0 or int(readout_positions.max()) >= sequence_length:
            raise ValueError("QKV readout position is outside sequence")
        if int(history_starts.min()) < 0 or int(history_ends.max()) > sequence_length:
            raise ValueError("QKV history span is outside sequence")
        if bool((history_ends <= history_starts).any()):
            raise ValueError("QKV history span is empty")
        if bool((history_ends > readout_positions.min(dim=1).values).any()):
            raise ValueError("QKV history span is not before readout")
        self.positions = readout_positions
        self.history_starts = history_starts
        self.history_ends = history_ends
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("QKV projection intervention is not armed")
        if self.fire_count != 1:
            raise RuntimeError(
                f"QKV projection hook fired {self.fire_count} times for {self.component}"
            )
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self.positions is None or self.history_starts is None or self.history_ends is None:
            raise RuntimeError("QKV projection hook fired while unarmed")
        if output.ndim != 3 or output.shape[0] != self.positions.shape[0]:
            raise ValueError("QKV projection output has unexpected shape")
        self.fire_count += 1
        scale = float(_MODE_SCALE[self.mode])
        if self.mode == "identity":
            self.last_summary = {
                "component": self.component,
                "mode": self.mode,
                "rows": int(output.shape[0]),
                "selected_positions": int(self.positions.shape[1]),
                "maximum_applied_delta": 0.0,
            }
            return output
        torch = _torch()
        modified = output.clone()
        rows = torch.arange(output.shape[0], device=output.device)
        if self.component == "q":
            positions = self.positions.to(output.device)
            selected = output[rows[:, None], positions]
            transformed = selected * scale
            modified[rows[:, None], positions] = transformed
            selected_count = int(positions.shape[0] * positions.shape[1])
        else:
            starts = self.history_starts.to(output.device)
            ends = self.history_ends.to(output.device)
            selected_count = 0
            maximum = 0.0
            for row in range(output.shape[0]):
                start = int(starts[row])
                end = int(ends[row])
                selected = output[row, start:end]
                transformed = selected * scale
                modified[row, start:end] = transformed
                selected_count += end - start
                maximum = max(
                    maximum,
                    float((transformed.float() - selected.float()).abs().max().item()),
                )
            self.last_summary = {
                "component": self.component,
                "mode": self.mode,
                "rows": int(output.shape[0]),
                "selected_positions": selected_count,
                "maximum_applied_delta": maximum,
            }
            return modified
        maximum = float((transformed.float() - selected.float()).abs().max().item())
        self.last_summary = {
            "component": self.component,
            "mode": self.mode,
            "rows": int(output.shape[0]),
            "selected_positions": selected_count,
            "maximum_applied_delta": maximum,
        }
        return modified

    def _clear(self) -> None:
        self.positions = None
        self.history_starts = None
        self.history_ends = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}


def _torch() -> Any:
    import torch

    return torch
