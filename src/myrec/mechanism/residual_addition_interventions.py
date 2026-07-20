"""Residual-addition operator interventions for the registered N32 boundary."""

from __future__ import annotations

from typing import Any, Sequence

from myrec.mechanism.embedding_interface_interventions import EMBEDDING_MODES
from myrec.mechanism.transformer_instrumentation import (
    _tensor_output,
    _torch,
    resolve_qwen_backbone,
    rms_matched_random_direction,
)


RESIDUAL_SCOPES = ("attention_residual_add", "mlp_residual_add")
RESIDUAL_MODES = EMBEDDING_MODES


def apply_increment_operator(
    increment: Any,
    mode: str,
    *,
    identity_keys: Sequence[Sequence[str]] | None = None,
    random_seed: int = 20260720,
) -> Any:
    """Transform an attention/MLP increment with fixed coefficient controls."""

    torch = _torch()
    if mode not in RESIDUAL_MODES:
        raise ValueError(f"unsupported residual mode={mode}")
    if increment.ndim < 3 or not increment.is_floating_point():
        raise ValueError("residual increment must be a floating rank-three tensor")
    if not torch.isfinite(increment).all():
        raise ValueError("residual increment contains non-finite values")
    if mode == "identity":
        return increment
    if mode == "zero":
        return torch.zeros_like(increment)
    if mode == "scale_half":
        return increment * 0.5
    if mode == "scale_double":
        return increment * 2.0
    if mode == "sign_flip":
        return -increment
    if identity_keys is None:
        raise ValueError("random residual mode requires identity keys")
    return rms_matched_random_direction(
        increment,
        seed=int(random_seed),
        identity_keys=identity_keys,
    )


class QwenResidualAdditionPatch:
    """Change only one residual addition at a fixed decoder block.

    All three native states are captured from the same forward.  The attention
    and MLP modules return their native outputs unchanged, and the selected
    final block rows are reconstructed as ``r + a' + m`` or ``r + a + m'``.
    Thus an attention-addition intervention does not silently re-run the MLP on
    a changed state, and an MLP-addition intervention does not alter attention.
    """

    def __init__(
        self,
        model: Any,
        block: int,
        scope: str,
        mode: str,
        *,
        random_seed: int = 20260720,
    ) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("residual block must be in [0, 27]")
        if scope not in RESIDUAL_SCOPES:
            raise ValueError(f"unsupported residual scope={scope}")
        if mode not in RESIDUAL_MODES:
            raise ValueError(f"unsupported residual mode={mode}")
        self.backbone = resolve_qwen_backbone(model)
        self.layer = self.backbone.layers[block]
        self.block = block
        self.scope = scope
        self.mode = mode
        self.random_seed = int(random_seed)
        self.positions: Any = None
        self.identity_keys: Sequence[Sequence[str]] | None = None
        self.sequence_length: int | None = None
        self.block_input: Any = None
        self.attention_increment: Any = None
        self.mlp_increment: Any = None
        self.counts = {"block_input": 0, "attention": 0, "mlp": 0, "block_output": 0}
        self.last_summary: dict[str, Any] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> "QwenResidualAdditionPatch":
        if self.handles:
            raise RuntimeError("residual addition patch is already active")
        self.handles = [
            self.layer.register_forward_pre_hook(self._block_input_hook),
            self.layer.self_attn.register_forward_hook(self._attention_hook),
            self.layer.mlp.register_forward_hook(self._mlp_hook),
            self.layer.register_forward_hook(self._block_output_hook),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self._clear()

    def arm(
        self,
        positions: Any,
        *,
        identity_keys: Sequence[Sequence[str]],
        sequence_length: int,
    ) -> None:
        if not self.handles or self.positions is not None:
            raise RuntimeError("residual addition patch cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("residual positions must have shape [batch, positive_count]")
        if len(identity_keys) != int(positions.shape[0]) or any(
            len(row) != int(positions.shape[1]) for row in identity_keys
        ):
            raise ValueError("residual identity keys do not align with positions")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("residual sequence length must be positive")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= sequence_length:
            raise ValueError("residual position is outside sequence")
        self.positions = positions
        self.identity_keys = identity_keys
        self.sequence_length = sequence_length
        self.block_input = None
        self.attention_increment = None
        self.mlp_increment = None
        self.counts = {key: 0 for key in self.counts}
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("residual addition patch is not armed")
        if self.counts != {"block_input": 1, "attention": 1, "mlp": 1, "block_output": 1}:
            raise RuntimeError(f"residual addition fire-count mismatch: {self.counts}")
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _clear(self) -> None:
        self.positions = None
        self.identity_keys = None
        self.sequence_length = None
        self.block_input = None
        self.attention_increment = None
        self.mlp_increment = None
        self.counts = {"block_input": 0, "attention": 0, "mlp": 0, "block_output": 0}
        self.last_summary = {}

    def _selected(self, tensor: Any) -> Any:
        if self.positions is None:
            raise RuntimeError("residual hook fired while unarmed")
        if tensor.ndim < 3 or tensor.shape[0] != self.positions.shape[0]:
            raise ValueError("residual tensor and positions do not align")
        if tensor.device != self.positions.device:
            raise ValueError("residual positions and tensor use different devices")
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        return tensor[rows, self.positions].detach()

    def _block_input_hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if not inputs:
            raise RuntimeError("decoder block received no hidden state")
        self.block_input = self._selected(inputs[0])
        self.counts["block_input"] += 1

    def _attention_hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        self.attention_increment = self._selected(_tensor_output(output, "attention_increment"))
        self.counts["attention"] += 1
        return output

    def _mlp_hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        self.mlp_increment = self._selected(_tensor_output(output, "mlp_increment"))
        self.counts["mlp"] += 1
        return output

    def _block_output_hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        if self.block_input is None or self.attention_increment is None or self.mlp_increment is None:
            raise RuntimeError("residual states were not captured before block output")
        native = _tensor_output(output, "block_output")
        if self.mode == "identity":
            self.counts["block_output"] += 1
            self.last_summary = {
                "block": self.block,
                "scope": self.scope,
                "mode": self.mode,
                "maximum_output_delta": 0.0,
            }
            return output
        transformed = (
            apply_increment_operator(
                self.attention_increment,
                self.mode,
                identity_keys=self.identity_keys,
                random_seed=self.random_seed,
            )
            if self.scope == "attention_residual_add"
            else self.attention_increment
        )
        transformed_mlp = (
            apply_increment_operator(
                self.mlp_increment,
                self.mode,
                identity_keys=self.identity_keys,
                random_seed=self.random_seed,
            )
            if self.scope == "mlp_residual_add"
            else self.mlp_increment
        )
        desired = self.block_input + transformed + transformed_mlp
        rows = _torch().arange(native.shape[0], device=native.device)[:, None]
        modified = native.clone()
        modified[rows, self.positions] = desired.to(dtype=native.dtype)
        self.counts["block_output"] += 1
        self.last_summary = {
            "block": self.block,
            "scope": self.scope,
            "mode": self.mode,
            "maximum_output_delta": float(
                _torch().max(_torch().abs(desired.float() - native[rows, self.positions].float())).item()
            ),
        }
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified
