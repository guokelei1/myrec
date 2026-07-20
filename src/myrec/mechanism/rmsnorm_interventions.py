"""Variance/gain-separated RMSNorm operator probes for frozen Qwen3 models."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


RMSNORM_SCOPES = ("input", "post_attention", "final")
RMSNORM_MODES = (
    "identity",
    "variance_scale_half",
    "variance_scale_double",
    "gain_scale_half",
    "gain_scale_double",
    "gain_sign_flip",
    "zero",
)


class QwenRMSNormPatch:
    """Intervene on variance rescaling or learned gain at explicit rows.

    The hook recomputes the native RMSNorm from the module input in float32
    and changes exactly one factor.  ``identity`` returns the untouched native
    output, making it a strict numerical control under BF16 execution.
    """

    def __init__(self, model: Any, scope: str, *, block: int | None = None, mode: str) -> None:
        scope = str(scope)
        mode = str(mode)
        if scope not in RMSNORM_SCOPES:
            raise ValueError(f"unsupported RMSNorm scope={scope}")
        if mode not in RMSNORM_MODES:
            raise ValueError(f"unsupported RMSNorm mode={mode}")
        if scope == "final":
            if block is not None:
                raise ValueError("final RMSNorm does not take a block")
            module = resolve_qwen_backbone(model).final_norm
            block_value = None
        else:
            if block is None or not 0 <= int(block) < 28:
                raise ValueError("block RMSNorm scope requires block in [0,27]")
            layer = resolve_qwen_backbone(model).layers[int(block)]
            module = (
                layer.input_layernorm if scope == "input" else layer.post_attention_layernorm
            )
            block_value = int(block)
        self.scope = scope
        self.block = block_value
        self.mode = mode
        self.module = module
        self.positions: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self.handle: Any = None

    def __enter__(self) -> "QwenRMSNormPatch":
        if self.handle is not None:
            raise RuntimeError("RMSNorm patch is already active")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self._clear()

    def arm(self, positions: Any, *, sequence_length: int) -> None:
        if self.handle is None or self.positions is not None:
            raise RuntimeError("RMSNorm patch cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("RMSNorm positions must be [batch,positive]")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("RMSNorm sequence length must be positive")
        if int(positions.min()) < 0 or int(positions.max()) >= sequence_length:
            raise ValueError("RMSNorm position is outside sequence")
        self.positions = positions
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("RMSNorm patch is not armed")
        if self.fire_count != 1:
            raise RuntimeError(f"RMSNorm hook fired {self.fire_count} times")
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _hook(self, _module: Any, inputs: Any, output: Any) -> Any:
        if self.positions is None:
            raise RuntimeError("RMSNorm hook fired while unarmed")
        if not inputs or output.ndim != 3 or inputs[0].ndim != 3:
            raise ValueError("RMSNorm hook expected rank-three input/output")
        hidden = inputs[0]
        if hidden.shape[:2] != output.shape[:2] or hidden.shape[0] != self.positions.shape[0]:
            raise ValueError("RMSNorm tensor does not align with positions")
        self.fire_count += 1
        if self.mode == "identity":
            self.last_summary = {
                "scope": self.scope,
                "block": self.block,
                "mode": self.mode,
                "maximum_applied_delta": 0.0,
            }
            return output
        torch = _torch()
        rows = torch.arange(hidden.shape[0], device=hidden.device)[:, None]
        positions = self.positions.to(hidden.device)
        selected_hidden = hidden[rows, positions].float()
        variance = selected_hidden.pow(2).mean(dim=-1, keepdim=True)
        epsilon = float(getattr(self.module, "variance_epsilon", 1e-6))
        normalized = selected_hidden * torch.rsqrt(variance + epsilon)
        weight = self.module.weight.to(device=hidden.device, dtype=torch.float32)
        transformed = normalized * weight
        if self.mode == "variance_scale_half":
            transformed = normalized * 0.5 * weight
        elif self.mode == "variance_scale_double":
            transformed = normalized * 2.0 * weight
        elif self.mode == "gain_scale_half":
            transformed = normalized * (weight * 0.5)
        elif self.mode == "gain_scale_double":
            transformed = normalized * (weight * 2.0)
        elif self.mode == "gain_sign_flip":
            transformed = normalized * (weight * -1.0)
        elif self.mode == "zero":
            transformed = torch.zeros_like(normalized)
        else:  # pragma: no cover - constructor guards this
            raise AssertionError(self.mode)
        modified = output.clone()
        modified[rows, positions] = transformed.to(dtype=output.dtype)
        native = output[rows, positions]
        self.last_summary = {
            "scope": self.scope,
            "block": self.block,
            "mode": self.mode,
            "selected_positions": int(positions.numel()),
            "maximum_applied_delta": float(
                (transformed.to(dtype=output.dtype).float() - native.float()).abs().max().item()
            ),
        }
        return modified

    def _clear(self) -> None:
        self.positions = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}


def _torch() -> Any:
    import torch

    return torch
