"""Operator-level RMSNorm interventions for the registered N31 boundary.

The intervention recomputes only selected token rows from the exact input seen
by one Qwen RMSNorm module.  It therefore keeps the incoming residual and all
downstream modules native; a state replacement from another path is not used.
The runtime/evaluator layer is intentionally separate from this hook primitive.
"""

from __future__ import annotations

from typing import Any, Sequence

from myrec.mechanism.transformer_instrumentation import (
    _tensor_output,
    _torch,
    resolve_qwen_backbone,
    rms_matched_random_direction,
)


RMSNORM_SCOPES = ("input_rmsnorm", "post_attention_rmsnorm")
RMSNORM_OPERATORS = ("variance_rescale", "learned_gain")
RMSNORM_MODES = (
    "identity",
    "half",
    "double",
    "sign_flip",
    "output_norm_matched_random",
)


def apply_rmsnorm_operator(
    hidden: Any,
    weight: Any,
    *,
    eps: float,
    operator: str,
    mode: str,
) -> Any:
    """Apply one registered RMSNorm operator to a selected hidden tensor.

    ``hidden`` is expected to have shape ``[..., hidden_size]``.  The identity
    mode is deliberately a tensor-preserving no-op; callers can use the native
    module output as the identity control without introducing recomputation
    roundoff.
    """

    torch = _torch()
    if operator not in RMSNORM_OPERATORS:
        raise ValueError(f"unsupported RMSNorm operator={operator}")
    if mode not in RMSNORM_MODES:
        raise ValueError(f"unsupported RMSNorm mode={mode}")
    if hidden.ndim < 2 or not hidden.is_floating_point():
        raise ValueError("RMSNorm hidden state must be a floating tensor")
    if weight.ndim != 1 or int(weight.shape[0]) != int(hidden.shape[-1]):
        raise ValueError("RMSNorm weight does not match hidden size")
    if not torch.isfinite(hidden).all() or not torch.isfinite(weight).all():
        raise ValueError("RMSNorm operator received non-finite tensors")
    if mode == "identity":
        return hidden
    if not isinstance(eps, (float, int)) or not float(eps) > 0:
        raise ValueError("RMSNorm epsilon must be positive")

    variance = hidden.float().pow(2).mean(dim=-1, keepdim=True)
    if operator == "variance_rescale":
        if mode == "half":
            variance = variance * 0.5
        elif mode == "double":
            variance = variance * 2.0
        elif mode == "sign_flip":
            # Sign is an output-direction control, not a variance operation;
            # the native variance is retained so its effect is interpretable.
            pass
        elif mode == "output_norm_matched_random":
            raise ValueError("random direction is applied after native RMSNorm")
        else:  # pragma: no cover - guarded above
            raise AssertionError(mode)
        transformed_weight = weight
    else:
        variance = variance
        if mode == "half":
            transformed_weight = weight * 0.5
        elif mode == "double":
            transformed_weight = weight * 2.0
        elif mode in {"sign_flip", "output_norm_matched_random"}:
            transformed_weight = weight
        else:  # pragma: no cover - guarded above
            raise AssertionError(mode)

    normalized = hidden.float() * torch.rsqrt(variance + float(eps))
    result = normalized.to(dtype=hidden.dtype) * transformed_weight.to(
        device=hidden.device, dtype=hidden.dtype
    )
    if mode == "sign_flip":
        result = -result
    return result


class QwenRMSNormOperatorPatch:
    """Patch one block RMSNorm's selected output rows at operator level."""

    def __init__(
        self,
        model: Any,
        block: int,
        scope: str,
        operator: str,
        mode: str,
        *,
        random_seed: int = 20260720,
    ) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("RMSNorm block must be in [0, 27]")
        if scope not in RMSNORM_SCOPES:
            raise ValueError(f"unsupported RMSNorm scope={scope}")
        if operator not in RMSNORM_OPERATORS:
            raise ValueError(f"unsupported RMSNorm operator={operator}")
        if mode not in RMSNORM_MODES:
            raise ValueError(f"unsupported RMSNorm mode={mode}")
        self.backbone = resolve_qwen_backbone(model)
        layer = self.backbone.layers[block]
        self.module = (
            layer.input_layernorm
            if scope == "input_rmsnorm"
            else layer.post_attention_layernorm
        )
        self.block = block
        self.scope = scope
        self.operator = operator
        self.mode = mode
        self.random_seed = int(random_seed)
        self.positions: Any = None
        self.identity_keys: Sequence[Sequence[str]] | None = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self.handle: Any = None

    def __enter__(self) -> "QwenRMSNormOperatorPatch":
        if self.handle is not None:
            raise RuntimeError("RMSNorm operator patch is already active")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self._clear()

    def arm(
        self,
        positions: Any,
        *,
        identity_keys: Sequence[Sequence[str]],
        sequence_length: int,
    ) -> None:
        if self.handle is None or self.positions is not None:
            raise RuntimeError("RMSNorm operator patch cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("RMSNorm positions must have shape [batch, positive_count]")
        if len(identity_keys) != int(positions.shape[0]) or any(
            len(row) != int(positions.shape[1]) for row in identity_keys
        ):
            raise ValueError("RMSNorm identity keys do not align with positions")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("RMSNorm sequence length must be positive")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= sequence_length:
            raise ValueError("RMSNorm position is outside sequence")
        self.positions = positions
        self.identity_keys = identity_keys
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None or self.fire_count != 1:
            raise RuntimeError(
                f"RMSNorm registered module fired {self.fire_count} times; expected one"
            )
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _clear(self) -> None:
        self.positions = None
        self.identity_keys = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}

    def _hook(self, module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if self.positions is None or self.identity_keys is None:
            raise RuntimeError("RMSNorm hook fired while unarmed")
        if not inputs:
            raise RuntimeError("RMSNorm module received no input")
        native = _tensor_output(output, f"block_{self.block:02d}.{self.scope}")
        hidden = inputs[0]
        if hidden.ndim < 3 or native.shape != hidden.shape:
            raise ValueError("RMSNorm input/output shape mismatch")
        if hidden.device != self.positions.device:
            raise ValueError("RMSNorm positions and hidden state use different devices")
        rows = _torch().arange(hidden.shape[0], device=hidden.device)[:, None]
        selected_hidden = hidden[rows, self.positions]
        selected_native = native[rows, self.positions]
        if self.mode == "identity":
            selected = selected_native
        elif self.mode == "output_norm_matched_random":
            selected = rms_matched_random_direction(
                selected_native,
                seed=self.random_seed,
                identity_keys=self.identity_keys,
            )
        else:
            eps = float(getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6)))
            selected = apply_rmsnorm_operator(
                selected_hidden,
                module.weight,
                eps=eps,
                operator=self.operator,
                mode=self.mode,
            )
        if selected.shape != selected_native.shape or not _torch().isfinite(selected).all():
            raise FloatingPointError("RMSNorm operator produced invalid selected rows")
        modified = native.clone()
        modified[rows, self.positions] = selected.to(dtype=native.dtype)
        self.fire_count += 1
        self.last_summary = {
            "block": self.block,
            "scope": self.scope,
            "operator": self.operator,
            "mode": self.mode,
            "rows": int(hidden.shape[0]),
            "positions_per_row": int(self.positions.shape[1]),
            "maximum_output_delta": float(
                _torch().max(_torch().abs(selected.float() - selected_native.float())).item()
            ),
        }
        return modified
