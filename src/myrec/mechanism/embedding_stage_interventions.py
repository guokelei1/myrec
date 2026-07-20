"""Scoped history-token embedding operators for Qwen3."""

from __future__ import annotations

from typing import Any

EMBEDDING_MODES = ("identity", "scale_half", "scale_double", "sign_flip", "zero")
_MODE_SCALE = {
    "identity": 1.0,
    "scale_half": 0.5,
    "scale_double": 2.0,
    "sign_flip": -1.0,
    "zero": 0.0,
}


class QwenHistoryEmbeddingIntervention:
    """Transform only history rows of the token-embedding output.

    The operator sits at ``embed_tokens`` output, before any Transformer block.
    Query, candidate, mask, position and all non-history rows remain native.
    ``identity`` returns the original tensor to make the numerical identity
    gate exact under low-precision kernels.
    """

    def __init__(self, model: Any, mode: str) -> None:
        mode = str(mode)
        if mode not in EMBEDDING_MODES:
            raise ValueError(f"unsupported embedding mode={mode}")
        self.mode = mode
        self.module = _resolve_embedding_module(model)
        self.starts: Any = None
        self.ends: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self.handle: Any = None

    def __enter__(self) -> "QwenHistoryEmbeddingIntervention":
        if self.handle is not None:
            raise RuntimeError("history embedding intervention is already active")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self._clear()

    def arm(self, starts: Any, ends: Any, *, sequence_length: int) -> None:
        if self.handle is None or self.starts is not None:
            raise RuntimeError("history embedding intervention cannot be armed")
        if starts.ndim != 1 or ends.ndim != 1 or starts.shape != ends.shape:
            raise ValueError("embedding history spans must be aligned vectors")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("embedding sequence length must be positive")
        if int(starts.min()) < 0 or int(ends.max()) > sequence_length:
            raise ValueError("embedding history span is outside sequence")
        if bool((ends <= starts).any()):
            raise ValueError("embedding history span is empty")
        self.starts = starts
        self.ends = ends
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.starts is None:
            raise RuntimeError("history embedding intervention is not armed")
        if self.fire_count != 1:
            raise RuntimeError(f"embedding hook fired {self.fire_count} times")
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self.starts is None or self.ends is None:
            raise RuntimeError("embedding hook fired while unarmed")
        if output.ndim != 3 or output.shape[0] != self.starts.shape[0]:
            raise ValueError("embedding output has unexpected shape")
        self.fire_count += 1
        if self.mode == "identity":
            self.last_summary = {
                "mode": self.mode,
                "rows": int(output.shape[0]),
                "selected_positions": 0,
                "maximum_applied_delta": 0.0,
            }
            return output
        scale = float(_MODE_SCALE[self.mode])
        torch = _torch()
        modified = output.clone()
        starts = self.starts.to(output.device)
        ends = self.ends.to(output.device)
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
            "mode": self.mode,
            "rows": int(output.shape[0]),
            "selected_positions": selected_count,
            "maximum_applied_delta": maximum,
        }
        return modified

    def _clear(self) -> None:
        self.starts = None
        self.ends = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}


def _torch() -> Any:
    import torch

    return torch


def _resolve_embedding_module(model: Any) -> Any:
    """Resolve the unique token embedding through optional PEFT wrappers."""

    torch = _torch()
    candidates = [
        module
        for name, module in model.named_modules()
        if (name == "embed_tokens" or name.endswith(".embed_tokens"))
        and isinstance(module, torch.nn.Embedding)
    ]
    if len(candidates) != 1:
        raise TypeError(
            f"expected one Qwen embed_tokens module, observed {len(candidates)}"
        )
    return candidates[0]
