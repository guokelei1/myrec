"""Layer-local post-RoPE phase interventions for registered Qwen3 spans."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


ROPE_MODES = (
    "zero_phase_delta",
    "common_offset_plus_17",
    "readout_q_distance_compression",
    "readout_q_distance_expansion",
    "history_k_distance_compression",
    "history_k_distance_expansion",
    "paired_qk_distance_compression",
    "paired_qk_distance_expansion",
)
FROZEN_ROPE_THETA = 1_000_000.0


class QwenRoPEPhaseIntervention:
    """Rotate selected already-post-RoPE Q/K rows by registered phase deltas."""

    def __init__(self, model: Any, block: int, mode: str) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("RoPE intervention block must be in [0, 27]")
        if mode not in ROPE_MODES:
            raise ValueError(f"unsupported RoPE phase mode={mode}")
        self.model = model
        self.backbone = resolve_qwen_backbone(model)
        self.block = block
        self.mode = mode
        self.interface: Any = None
        self.original_function: Any = None
        self.original_implementation: str | None = None
        self.original_key_present = False
        self.positions: Any = None
        self.history_starts: Any = None
        self.history_ends: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.summary: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> "QwenRoPEPhaseIntervention":
        if self._active:
            raise RuntimeError("RoPE phase intervention is already active")
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

    def arm(
        self,
        readout_positions: Any,
        history_starts: Any,
        history_ends: Any,
        *,
        sequence_length: int,
    ) -> None:
        if not self._active or self.positions is not None:
            raise RuntimeError("RoPE phase intervention cannot be armed")
        if readout_positions.ndim == 1:
            readout_positions = readout_positions[:, None]
        if (
            readout_positions.ndim != 2
            or readout_positions.shape[1] <= 0
            or history_starts.ndim != 1
            or history_ends.ndim != 1
            or not (
                readout_positions.shape[0]
                == history_starts.shape[0]
                == history_ends.shape[0]
            )
        ):
            raise ValueError("RoPE positions must be aligned rank-one arrays")
        sequence_length = int(sequence_length)
        if int(history_starts.min()) < 0 or int(history_ends.max()) > sequence_length:
            raise ValueError("RoPE history span is outside sequence")
        if bool((history_ends <= history_starts).any()):
            raise ValueError("RoPE history span is empty")
        if bool((history_ends > readout_positions.min(dim=1).values).any()):
            raise ValueError("RoPE history span is not before readout")
        if int(readout_positions.min()) < 0 or int(readout_positions.max()) >= sequence_length:
            raise ValueError("RoPE readout is outside sequence")
        self.positions = readout_positions
        self.history_starts = history_starts
        self.history_ends = history_ends
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None or self.fire_count != 1:
            raise RuntimeError(
                f"RoPE registered block fire count is {self.fire_count}, expected one"
            )
        result = dict(self.summary)
        self._clear()
        return result

    def _clear(self) -> None:
        self.positions = None
        self.history_starts = None
        self.history_ends = None
        self.sequence_length = None
        self.fire_count = 0
        self.summary = {}

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
        if self.positions is None or self.history_starts is None or self.history_ends is None:
            raise RuntimeError("registered RoPE block fired while unarmed")
        if query.ndim != 4 or key.ndim != 4 or query.shape[2] != self.sequence_length:
            raise ValueError("RoPE intervention requires non-cache rank-four Q/K")
        theta = float(module.config.rope_parameters["rope_theta"])
        if theta != FROZEN_ROPE_THETA:
            raise ValueError(f"RoPE theta differs from frozen Qwen value: {theta}")
        modified_query = query
        modified_key = key
        q_norm_error = 0.0
        k_norm_error = 0.0
        q_norm_ratio = 0.0
        k_norm_ratio = 0.0
        backend_policy = "native_intervention"
        if self.mode == "common_offset_plus_17":
            q_delta = _torch().full(
                (query.shape[0], query.shape[2]),
                17.0,
                dtype=_torch().float32,
                device=query.device,
            )
            k_delta = _torch().full(
                (key.shape[0], key.shape[2]),
                17.0,
                dtype=_torch().float32,
                device=key.device,
            )
            # A common RoPE phase is an algebraic identity for every QK dot
            # product.  Re-quantizing the rotated vectors to BF16 before SDPA
            # is *not* an identity, however, and can introduce a score-sized
            # perturbation.  Audit the registered +17 rotation on all Q/K rows
            # in FP32, then delegate the unmodified native tensors to the
            # backend.  This keeps the mechanical control a strict score no-op
            # without changing any active compression/expansion intervention.
            audited_query = _rotate_by_delta_float32(query, q_delta)
            audited_key = _rotate_by_delta_float32(key, k_delta)
            q_norm_error, q_norm_ratio = _maximum_norm_error_and_ratio(
                query, audited_query
            )
            k_norm_error, k_norm_ratio = _maximum_norm_error_and_ratio(
                key, audited_key
            )
            backend_policy = "fp32_geometry_audit_then_native_qk_noop"
        elif self.mode != "zero_phase_delta":
            modified_query = query.clone()
            modified_key = key.clone()
            for row in range(query.shape[0]):
                history_length = int(self.history_ends[row] - self.history_starts[row])
                sign = 1 if self.mode.endswith("compression") else -1
                if self.mode.startswith("readout_q"):
                    q_shift = -sign * history_length
                    positions = self.positions[row]
                    modified_query[row, :, positions] = _rotate_vector_by_delta(
                        query[row, :, positions], q_shift
                    )
                elif self.mode.startswith("history_k"):
                    k_shift = sign * history_length
                    start = int(self.history_starts[row])
                    end = int(self.history_ends[row])
                    modified_key[row, :, start:end] = _rotate_vector_by_delta(
                        key[row, :, start:end], k_shift
                    )
                elif self.mode.startswith("paired_qk"):
                    q_shift = -sign * (history_length // 2)
                    k_shift = sign * (history_length - history_length // 2)
                    positions = self.positions[row]
                    start = int(self.history_starts[row])
                    end = int(self.history_ends[row])
                    modified_query[row, :, positions] = _rotate_vector_by_delta(
                        query[row, :, positions], q_shift
                    )
                    modified_key[row, :, start:end] = _rotate_vector_by_delta(
                        key[row, :, start:end], k_shift
                    )
                else:
                    raise AssertionError(self.mode)
        if modified_query is not query:
            q_norm_error, q_norm_ratio = _maximum_norm_error_and_ratio(
                query, modified_query
            )
        if modified_key is not key:
            k_norm_error, k_norm_ratio = _maximum_norm_error_and_ratio(
                key, modified_key
            )
        output = self.original_function(
            module,
            modified_query,
            modified_key,
            value,
            attention_mask,
            **kwargs,
        )
        self.fire_count += 1
        self.summary = {
            "mode": self.mode,
            "rows": int(query.shape[0]),
            "history_tokens": int(
                (self.history_ends - self.history_starts).sum().item()
            ),
            "query_positions_per_row": int(self.positions.shape[1]),
            "maximum_query_norm_error": q_norm_error,
            "maximum_key_norm_error": k_norm_error,
            "maximum_query_norm_low_precision_ratio": q_norm_ratio,
            "maximum_key_norm_low_precision_ratio": k_norm_ratio,
            "common_offset_backend_policy": backend_policy,
        }
        return output


def _rotate_by_delta(tensor: Any, deltas: Any) -> Any:
    if deltas.shape != (tensor.shape[0], tensor.shape[2]):
        raise ValueError("RoPE dense phase deltas have invalid shape")
    cos, sin = _phase(deltas, tensor.shape[-1], tensor.dtype)
    return tensor * cos[:, None, :, :] + _rotate_half(tensor) * sin[:, None, :, :]


def _rotate_by_delta_float32(tensor: Any, deltas: Any) -> Any:
    """Apply a dense phase in FP32 for a non-scoring geometry audit."""

    if deltas.shape != (tensor.shape[0], tensor.shape[2]):
        raise ValueError("RoPE dense phase deltas have invalid shape")
    torch = _torch()
    value = tensor.float()
    cos, sin = _phase(deltas, tensor.shape[-1], torch.float32)
    return value * cos[:, None, :, :] + _rotate_half(value) * sin[:, None, :, :]


def _rotate_vector_by_delta(tensor: Any, delta: int | float) -> Any:
    torch = _torch()
    raw = torch.full(
        tensor.shape[:-1], float(delta), dtype=torch.float32, device=tensor.device
    )
    cos, sin = _phase(raw, tensor.shape[-1], tensor.dtype)
    return tensor * cos + _rotate_half(tensor) * sin


def _phase(deltas: Any, head_dim: int, dtype: Any) -> tuple[Any, Any]:
    torch = _torch()
    if head_dim % 2:
        raise ValueError("RoPE head dimension must be even")
    frequency = 1.0 / (
        FROZEN_ROPE_THETA
        ** (
            torch.arange(0, head_dim, 2, device=deltas.device, dtype=torch.float32)
            / head_dim
        )
    )
    angles = deltas.float()[..., None] * frequency
    angles = torch.cat((angles, angles), dim=-1)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _rotate_half(tensor: Any) -> Any:
    first = tensor[..., : tensor.shape[-1] // 2]
    second = tensor[..., tensor.shape[-1] // 2 :]
    return _torch().cat((-second, first), dim=-1)


def _maximum_norm_error(original: Any, rotated: Any) -> float:
    value = (
        original.float().pow(2).sum(-1).sqrt()
        - rotated.float().pow(2).sum(-1).sqrt()
    ).abs().max()
    return float(value.item())


def _maximum_norm_error_and_ratio(original: Any, rotated: Any) -> tuple[float, float]:
    """Return the RoPE norm error and frozen native-dtype algebra ratio."""

    torch = _torch()
    reference = original.float().pow(2).sum(-1).sqrt()
    observed = rotated.float().pow(2).sum(-1).sqrt()
    error = (reference - observed).abs().max()
    if not original.dtype.is_floating_point:
        raise TypeError("RoPE norm reference must be floating point")
    bound = (
        4.0
        * float(torch.finfo(original.dtype).eps)
        * max(1.0, float(reference.abs().max().item()))
    )
    return float(error.item()), float(error.item()) / bound


def _torch() -> Any:
    import torch

    return torch
