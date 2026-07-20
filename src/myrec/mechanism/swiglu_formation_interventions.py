"""Operator-isolated SwiGLU formation interventions for the N25 boundary."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


SWIGLU_OPERATORS = ("gate_pre", "up", "silu_gate", "product")
SWIGLU_MODES = (
    "identity",
    "zero",
    "scale_half",
    "scale_double",
    "sign_flip",
    "output_norm_matched_random",
)


class QwenSwiGLUFormationPatch:
    """Patch one complete SwiGLU formation stage at selected token rows.

    The patch leaves all other MLP stages and the down projection untouched.
    Hooks are armed for one model forward and enforce exactly-once coverage.
    ``output_norm_matched_random`` uses a fixed local generator and matches
    each selected row's native L2 norm, so no outcome-dependent direction is
    introduced.
    """

    def __init__(self, model: Any, block: int, operator: str, mode: str) -> None:
        if operator not in SWIGLU_OPERATORS:
            raise ValueError(f"unsupported SwiGLU operator={operator}")
        if mode not in SWIGLU_MODES:
            raise ValueError(f"unsupported SwiGLU mode={mode}")
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("SwiGLU block must be in [0,27]")
        self.layer = resolve_qwen_backbone(model).layers[block]
        self.operator = operator
        self.mode = mode
        self.positions: Any = None
        self.fire_count = 0
        self.handles: list[Any] = []

    def __enter__(self) -> "QwenSwiGLUFormationPatch":
        if self.handles:
            raise RuntimeError("SwiGLU patch is already active")
        module = {
            "gate_pre": self.layer.mlp.gate_proj,
            "up": self.layer.mlp.up_proj,
            "silu_gate": self.layer.mlp.act_fn,
        }.get(self.operator)
        if module is not None:
            self.handles.append(module.register_forward_hook(self._output_hook))
        else:
            self.handles.append(
                self.layer.mlp.down_proj.register_forward_pre_hook(self._product_hook)
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._clear()

    def arm(self, positions: Any, *, sequence_length: int) -> None:
        if not self.handles or self.positions is not None:
            raise RuntimeError("SwiGLU patch cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("SwiGLU positions must be [batch,positive]")
        if int(positions.min()) < 0 or int(positions.max()) >= int(sequence_length):
            raise ValueError("SwiGLU position is outside sequence")
        self.positions = positions
        self.fire_count = 0

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("SwiGLU patch is not armed")
        if self.fire_count != 1:
            raise RuntimeError("SwiGLU patch did not fire exactly once")
        result = {"operator": self.operator, "mode": self.mode, "fire_count": self.fire_count}
        self._clear()
        return result

    def _output_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        self.fire_count += 1
        return self._patch(output)

    def _product_hook(self, _module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        self.fire_count += 1
        if not inputs:
            raise RuntimeError("SwiGLU down projection received no product")
        product = self._patch(inputs[0])
        return (product, *inputs[1:])

    def _patch(self, tensor: Any) -> Any:
        if self.positions is None or tensor.ndim != 3:
            raise RuntimeError("SwiGLU hook fired while unarmed")
        positions = self.positions.to(tensor.device)
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        native = tensor[rows, positions]
        if self.mode == "identity":
            return tensor
        if self.mode == "zero":
            replacement = _torch().zeros_like(native)
        elif self.mode == "scale_half":
            replacement = native * 0.5
        elif self.mode == "scale_double":
            replacement = native * 2.0
        elif self.mode == "sign_flip":
            replacement = -native
        else:
            generator = _torch().Generator(device=tensor.device)
            generator.manual_seed(20260720 + 1009 * SWIGLU_OPERATORS.index(self.operator))
            random = _torch().randn(native.shape, generator=generator, device=tensor.device, dtype=tensor.dtype)
            native_norm = native.float().norm(dim=-1, keepdim=True)
            random_norm = random.float().norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
            replacement = (random.float() * (native_norm / random_norm)).to(tensor.dtype)
        if _torch().equal(native, replacement):
            return tensor
        modified = tensor.clone()
        modified[rows, positions] = replacement.to(dtype=tensor.dtype)
        return modified

    def _clear(self) -> None:
        self.positions = None
        self.fire_count = 0


def _torch() -> Any:
    import torch

    return torch

