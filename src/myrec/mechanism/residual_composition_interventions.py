"""Coefficient-isolated residual composition probes for Qwen3 blocks.

The probe changes only the coefficient of one already-computed branch
increment in the frozen decoder equation ``r + a`` or ``u + m``.  It does not
replace the residual state with a donor and therefore keeps the incoming
residual and branch tensor from the same forward pass.  This is the narrow
operator test that ordinary post-block state patching cannot provide.
"""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


RESIDUAL_BRANCHES = ("attention", "mlp")
RESIDUAL_MODES = (
    "identity",
    "scale_half",
    "scale_double",
    "sign_flip",
    "zero",
)
_MODE_SCALE = {
    "identity": 1.0,
    "scale_half": 0.5,
    "scale_double": 2.0,
    "sign_flip": -1.0,
    "zero": 0.0,
}


class QwenResidualCompositionPatch:
    """Scale one branch increment at registered token rows.

    ``attention`` hooks the self-attention output immediately before the
    decoder's ``residual + attention`` expression. ``mlp`` hooks the MLP
    output immediately before ``post_attention_residual + mlp``. Identity is
    a literal no-op to avoid low-precision clone/write drift.
    """

    def __init__(self, model: Any, block: int, branch: str, mode: str) -> None:
        block = int(block)
        branch = str(branch)
        mode = str(mode)
        if not 0 <= block < 28:
            raise ValueError("residual composition block must be in [0, 27]")
        if branch not in RESIDUAL_BRANCHES:
            raise ValueError(f"unsupported residual branch={branch}")
        if mode not in RESIDUAL_MODES:
            raise ValueError(f"unsupported residual mode={mode}")
        self.block = block
        self.branch = branch
        self.mode = mode
        self.layer = resolve_qwen_backbone(model).layers[block]
        self.module = self.layer.self_attn if branch == "attention" else self.layer.mlp
        self.positions: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self.handle: Any = None

    def __enter__(self) -> "QwenResidualCompositionPatch":
        if self.handle is not None:
            raise RuntimeError("residual composition patch is already active")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self._clear()

    def arm(self, positions: Any, *, sequence_length: int) -> None:
        if self.handle is None or self.positions is not None:
            raise RuntimeError("residual composition patch cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("residual composition positions must be [batch,positive]")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("residual composition sequence length must be positive")
        if int(positions.min()) < 0 or int(positions.max()) >= sequence_length:
            raise ValueError("residual composition position is outside sequence")
        self.positions = positions
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("residual composition patch is not armed")
        if self.fire_count != 1:
            raise RuntimeError(
                f"residual composition hook fired {self.fire_count} times"
            )
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self.positions is None:
            raise RuntimeError("residual composition hook fired while unarmed")
        tensor, is_tuple = _tensor_and_tuple(output)
        if tensor.ndim != 3 or tensor.shape[0] != self.positions.shape[0]:
            raise ValueError("residual branch output has unexpected shape")
        self.fire_count += 1
        scale = float(_MODE_SCALE[self.mode])
        if self.mode == "identity":
            self.last_summary = {
                "branch": self.branch,
                "mode": self.mode,
                "selected_positions": int(self.positions.numel()),
                "maximum_applied_delta": 0.0,
            }
            return output
        torch = _torch()
        rows = torch.arange(tensor.shape[0], device=tensor.device)[:, None]
        positions = self.positions.to(tensor.device)
        selected = tensor[rows, positions]
        transformed = selected * scale
        modified = tensor.clone()
        modified[rows, positions] = transformed
        self.last_summary = {
            "branch": self.branch,
            "mode": self.mode,
            "selected_positions": int(positions.numel()),
            "maximum_applied_delta": float(
                (transformed.float() - selected.float()).abs().max().item()
            ),
        }
        if is_tuple:
            return (modified, *output[1:])
        return modified

    def _clear(self) -> None:
        self.positions = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}


def _tensor_and_tuple(output: Any) -> tuple[Any, bool]:
    if isinstance(output, tuple):
        if not output:
            raise ValueError("residual branch returned an empty tuple")
        return output[0], True
    return output, False


def _torch() -> Any:
    import torch

    return torch
