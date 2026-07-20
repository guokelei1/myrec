"""Operator-local Q/K head RMSNorm interventions for frozen Qwen3 models.

This primitive is deliberately not wired into a scoring queue yet.  It isolates
the ``q_proj/k_proj -> q_norm/k_norm -> RoPE`` boundary registered by N17 while
keeping all unselected tokens and the native identity path untouched.
"""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


HEAD_NORM_COMPONENTS = ("q", "k")
HEAD_NORM_MODES = (
    "identity",
    "variance_scale_half",
    "variance_scale_double",
    "gain_scale_half",
    "gain_scale_double",
    "gain_sign_flip",
    "zero",
)

_MODE_SCALE = {
    "variance_scale_half": 0.5,
    "variance_scale_double": 2.0,
    "gain_scale_half": 0.5,
    "gain_scale_double": 2.0,
    "gain_sign_flip": -1.0,
}


class QwenQKHeadRMSNormPatch:
    """Change one Q/K head-norm factor on fixed token spans in one block.

    ``q`` interventions apply to the registered readout positions.  ``k``
    interventions apply to each request's history span.  The hook sees the
    native rank-four ``[batch, sequence, heads, head_dim]`` tensor before the
    attention transpose, so RoPE, the other Q/K side, V, masking, softmax and
    output projection remain native.
    """

    def __init__(self, model: Any, block: int, component: str, mode: str) -> None:
        block = int(block)
        component = str(component)
        mode = str(mode)
        if not 0 <= block < 28:
            raise ValueError("Q/K head norm block must be in [0, 27]")
        if component not in HEAD_NORM_COMPONENTS:
            raise ValueError(f"unsupported Q/K head norm component={component}")
        if mode not in HEAD_NORM_MODES:
            raise ValueError(f"unsupported Q/K head norm mode={mode}")
        self.block = block
        self.component = component
        self.mode = mode
        backbone = resolve_qwen_backbone(model)
        self.layer = backbone.layers[block]
        self.module = getattr(self.layer.self_attn, f"{component}_norm")
        self.readout_positions: Any = None
        self.history_starts: Any = None
        self.history_ends: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.handle: Any = None
        self.last_summary: dict[str, Any] = {}

    def __enter__(self) -> "QwenQKHeadRMSNormPatch":
        if self.handle is not None:
            raise RuntimeError("Q/K head norm patch is already active")
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
        if self.handle is None or self.readout_positions is not None:
            raise RuntimeError("Q/K head norm patch cannot be armed")
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
            raise ValueError("Q/K head norm positions and spans are not batch-aligned")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("Q/K head norm sequence length must be positive")
        if int(readout_positions.min()) < 0 or int(readout_positions.max()) >= sequence_length:
            raise ValueError("Q/K head norm readout position is outside sequence")
        if int(history_starts.min()) < 0 or int(history_ends.max()) > sequence_length:
            raise ValueError("Q/K head norm history span is outside sequence")
        if bool((history_ends <= history_starts).any()):
            raise ValueError("Q/K head norm history span is empty")
        if bool((history_ends > readout_positions.min(dim=1).values).any()):
            raise ValueError("Q/K head norm history span is not before readout")
        self.readout_positions = readout_positions
        self.history_starts = history_starts
        self.history_ends = history_ends
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.readout_positions is None:
            raise RuntimeError("Q/K head norm patch is not armed")
        if self.fire_count != 1:
            raise RuntimeError(f"Q/K head norm hook fired {self.fire_count} times")
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _hook(self, _module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if self.readout_positions is None or self.history_starts is None or self.history_ends is None:
            raise RuntimeError("Q/K head norm hook fired while unarmed")
        if not inputs or output.ndim != 4 or inputs[0].ndim != 4:
            raise ValueError("Q/K head norm expects rank-four input and output")
        hidden = inputs[0]
        if output.shape != hidden.shape or output.shape[0] != self.readout_positions.shape[0]:
            raise ValueError("Q/K head norm tensor shape mismatch")
        if self.sequence_length != int(output.shape[1]):
            raise ValueError("Q/K head norm sequence length drifted")
        self.fire_count += 1
        if self.mode == "identity":
            self.last_summary = {
                "component": self.component,
                "mode": self.mode,
                "selected_positions": self._selected_count(),
                "maximum_applied_delta": 0.0,
                "native_recomposition_max_abs_error": None,
            }
            return output

        torch = _torch()
        hidden_fp = hidden.float()
        variance = hidden_fp.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_fp * torch.rsqrt(variance + float(self.module.variance_epsilon))
        native = normalized * self.module.weight.float()
        if self.mode == "zero":
            transformed = torch.zeros_like(native)
        elif self.mode.startswith("variance_"):
            transformed = native * float(_MODE_SCALE[self.mode])
        elif self.mode.startswith("gain_"):
            transformed = normalized * (self.module.weight.float() * float(_MODE_SCALE[self.mode]))
        else:  # pragma: no cover - constructor validates modes
            raise AssertionError(self.mode)

        rows = torch.arange(output.shape[0], device=output.device)
        modified = output.clone()
        if self.component == "q":
            positions = self.readout_positions.to(output.device)
            selected_native = native[rows[:, None], positions]
            selected_transformed = transformed[rows[:, None], positions]
            modified[rows[:, None], positions] = selected_transformed.to(output.dtype)
            selected_count = int(positions.numel())
            native_error = (selected_native.to(output.dtype).float() - output[rows[:, None], positions].float()).abs().max()
        else:
            starts = self.history_starts.to(output.device)
            ends = self.history_ends.to(output.device)
            selected_count = 0
            native_error = torch.tensor(0.0, device=output.device)
            for row in range(output.shape[0]):
                start, end = int(starts[row]), int(ends[row])
                selected_native = native[row, start:end]
                selected_transformed = transformed[row, start:end]
                modified[row, start:end] = selected_transformed.to(output.dtype)
                selected_count += end - start
                native_error = torch.maximum(
                    native_error,
                    (selected_native.to(output.dtype).float() - output[row, start:end].float()).abs().max(),
                )
        delta = (modified.float() - output.float())
        self.last_summary = {
            "component": self.component,
            "mode": self.mode,
            "selected_positions": selected_count,
            "maximum_applied_delta": float(delta.abs().max().item()),
            "native_recomposition_max_abs_error": float(native_error.item()),
        }
        return modified

    def _selected_count(self) -> int:
        if self.component == "q":
            return int(self.readout_positions.numel())
        return int((self.history_ends - self.history_starts).sum().item())

    def _clear(self) -> None:
        self.readout_positions = None
        self.history_starts = None
        self.history_ends = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}


def _torch() -> Any:
    import torch

    return torch

