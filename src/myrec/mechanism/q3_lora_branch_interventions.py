"""Complete Q3 LoRA branch interventions for the inactive N19 boundary.

The patch changes only the evaluated ``(alpha/r) B(A(x))`` contribution of one
q_proj or v_proj module.  It intentionally does not alter LoRA parameters,
the base projection, the other adapter, or any downstream attention operator.
It is a primitive for a future qrels-blind scorer, not a training method.
"""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


LORA_BRANCH_COMPONENTS = ("q", "v")
LORA_BRANCH_MODES = (
    "identity",
    "zero",
    "scale_half",
    "scale_double",
    "sign_flip",
    "output_norm_matched_random",
)
_MODE_SCALE = {
    "zero": 0.0,
    "scale_half": 0.5,
    "scale_double": 2.0,
    "sign_flip": -1.0,
}


class QwenQ3LoraBranchPatch:
    """Scale or replace one complete q/v LoRA contribution at fixed rows."""

    def __init__(self, model: Any, block: int, component: str, mode: str) -> None:
        block = int(block)
        component = str(component)
        mode = str(mode)
        if not 0 <= block < 28:
            raise ValueError("Q3 LoRA branch block must be in [0, 27]")
        if component not in LORA_BRANCH_COMPONENTS:
            raise ValueError(f"unsupported Q3 LoRA branch component={component}")
        if mode not in LORA_BRANCH_MODES:
            raise ValueError(f"unsupported Q3 LoRA branch mode={mode}")
        backbone = resolve_qwen_backbone(model)
        self.block = block
        self.component = component
        self.mode = mode
        self.module = getattr(backbone.layers[block].self_attn, f"{component}_proj")
        self.positions: Any = None
        self.history_starts: Any = None
        self.history_ends: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.handle: Any = None
        self.last_summary: dict[str, Any] = {}
        self.adapter_name: str | None = None

    def __enter__(self) -> "QwenQ3LoraBranchPatch":
        if self.handle is not None:
            raise RuntimeError("Q3 LoRA branch patch is already active")
        self.adapter_name = self._resolve_adapter_name()
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.adapter_name = None
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
            raise RuntimeError("Q3 LoRA branch patch cannot be armed")
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
            raise ValueError("Q3 LoRA positions and spans are not batch-aligned")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("Q3 LoRA sequence length must be positive")
        if int(readout_positions.min()) < 0 or int(readout_positions.max()) >= sequence_length:
            raise ValueError("Q3 LoRA readout position is outside sequence")
        if int(history_starts.min()) < 0 or int(history_ends.max()) > sequence_length:
            raise ValueError("Q3 LoRA history span is outside sequence")
        if bool((history_ends <= history_starts).any()):
            raise ValueError("Q3 LoRA history span is empty")
        if bool((history_ends > readout_positions.min(dim=1).values).any()):
            raise ValueError("Q3 LoRA history span is not before readout")
        self.positions = readout_positions
        self.history_starts = history_starts
        self.history_ends = history_ends
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None:
            raise RuntimeError("Q3 LoRA branch patch is not armed")
        if self.fire_count != 1:
            raise RuntimeError(f"Q3 LoRA branch hook fired {self.fire_count} times")
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _hook(self, _module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if self.positions is None or self.history_starts is None or self.history_ends is None:
            raise RuntimeError("Q3 LoRA branch hook fired while unarmed")
        if self.adapter_name is None or not inputs or output.ndim != 3:
            raise ValueError("Q3 LoRA branch expects one tensor input and rank-three output")
        hidden = inputs[0]
        if hidden.ndim != 3 or hidden.shape[:2] != output.shape[:2]:
            raise ValueError("Q3 LoRA branch input/output shape mismatch")
        if self.sequence_length != int(output.shape[1]):
            raise ValueError("Q3 LoRA branch sequence length drifted")
        self.fire_count += 1
        if self.mode == "identity":
            self.last_summary = {
                "component": self.component,
                "mode": self.mode,
                "adapter_name": self.adapter_name,
                "selected_positions": self._selected_count(),
                "maximum_applied_delta": 0.0,
                "native_recomposition_max_abs_error": None,
            }
            return output

        torch = _torch()
        base_layer = getattr(self.module, "base_layer", None)
        lora_a = _mapping_value(getattr(self.module, "lora_A", {}), self.adapter_name)
        lora_b = _mapping_value(getattr(self.module, "lora_B", {}), self.adapter_name)
        dropout = _mapping_value(getattr(self.module, "lora_dropout", {}), self.adapter_name)
        scaling = _mapping_value(getattr(self.module, "scaling", {}), self.adapter_name)
        if base_layer is None or lora_a is None or lora_b is None or dropout is None or scaling is None:
            raise TypeError("registered Q3 module is not an unmerged PEFT LoRA linear")
        if bool(getattr(self.module, "merged", False)) or bool(getattr(self.module, "disable_adapters", False)):
            raise RuntimeError("Q3 LoRA branch requires an active unmerged adapter")
        base = base_layer(hidden)
        adapter_input = hidden
        cast_input = getattr(self.module, "_cast_input_dtype", None)
        if cast_input is not None:
            adapter_input = cast_input(adapter_input, lora_a.weight.dtype)
        adapter = lora_b(lora_a(dropout(adapter_input))) * float(scaling)
        native_recomposition = (base + adapter).to(output.dtype)
        reconstruction_error = float((native_recomposition.float() - output.float()).abs().max().item())
        adapter_fp = adapter.float()
        if self.mode == "zero":
            transformed = torch.zeros_like(adapter_fp)
        elif self.mode == "output_norm_matched_random":
            generator = torch.Generator(device=adapter.device)
            generator.manual_seed(20260720 + self.block * 10 + (0 if self.component == "q" else 1))
            transformed = torch.randn(adapter.shape, generator=generator, device=adapter.device, dtype=torch.float32)
            transformed = self._norm_match(transformed, adapter_fp)
        else:
            transformed = adapter_fp * float(_MODE_SCALE[self.mode])
        modified = output.clone()
        rows = torch.arange(output.shape[0], device=output.device)
        if self.component == "q":
            positions = self.positions.to(output.device)
            modified[rows[:, None], positions] = (
                base.float()[rows[:, None], positions] + transformed[rows[:, None], positions]
            ).to(output.dtype)
            selected_count = int(positions.numel())
        else:
            starts = self.history_starts.to(output.device)
            ends = self.history_ends.to(output.device)
            selected_count = 0
            for row in range(output.shape[0]):
                start, end = int(starts[row]), int(ends[row])
                modified[row, start:end] = (base.float()[row, start:end] + transformed[row, start:end]).to(output.dtype)
                selected_count += end - start
        delta = modified.float() - output.float()
        self.last_summary = {
            "component": self.component,
            "mode": self.mode,
            "adapter_name": self.adapter_name,
            "selected_positions": selected_count,
            "maximum_applied_delta": float(delta.abs().max().item()),
            "native_recomposition_max_abs_error": reconstruction_error,
        }
        return modified

    @staticmethod
    def _norm_match(random_value: Any, reference: Any) -> Any:
        torch = _torch()
        ref_norm = reference.float().norm(dim=-1, keepdim=True)
        random_norm = random_value.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        return random_value * (ref_norm / random_norm)

    def _resolve_adapter_name(self) -> str:
        names = list(getattr(self.module, "active_adapters", ()))
        if len(names) != 1:
            raise ValueError(f"Q3 branch requires exactly one active adapter, observed {names}")
        name = str(names[0])
        if name not in getattr(self.module, "lora_A", {}):
            raise ValueError(f"active adapter {name!r} has no lora_A factor")
        return name

    def _selected_count(self) -> int:
        if self.component == "q":
            return int(self.positions.numel())
        return int((self.history_ends - self.history_starts).sum().item())

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


def _mapping_value(container: Any, key: str) -> Any:
    """Read both PEFT ModuleDicts and ordinary scaling dictionaries."""

    try:
        return container[key]
    except (KeyError, TypeError, AttributeError):
        return None
