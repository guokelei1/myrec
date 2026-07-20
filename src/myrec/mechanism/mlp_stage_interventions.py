"""Scoped full-stage gate/up interventions for Qwen's SwiGLU MLP."""

from __future__ import annotations

from typing import Any

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


class QwenMLPStagePatch:
    """Patch selected gate_proj and/or up_proj output rows.

    The native SiLU and down projection remain in the model.  Consequently a
    gate donor tests the actual nonlinear gate path, while an up donor tests the
    value path; a joint donor preserves their native composition on the
    recipient request.
    """

    def __init__(self, model: Any, block: int) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("MLP stage block must be in [0, 27]")
        layer = resolve_qwen_backbone(model).layers[block]
        self.layer = layer
        self.positions: Any = None
        self.gate_donor: Any = None
        self.up_donor: Any = None
        self.fire_counts = {"gate": 0, "up": 0}
        self.handles: list[Any] = []

    def __enter__(self) -> "QwenMLPStagePatch":
        if self.handles:
            raise RuntimeError("MLP stage patch is already active")
        self.handles = [
            self.layer.mlp.gate_proj.register_forward_hook(
                self._gate_hook
            ),
            self.layer.mlp.up_proj.register_forward_hook(self._up_hook),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._clear()

    def arm(
        self,
        positions: Any,
        *,
        gate_donor: Any | None = None,
        up_donor: Any | None = None,
    ) -> None:
        if not self.handles or self.positions is not None:
            raise RuntimeError("MLP stage patch cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("MLP stage positions must be [batch,positive]")
        if gate_donor is None and up_donor is None:
            raise ValueError("MLP stage patch requires gate or up donor")
        for name, donor in (("gate", gate_donor), ("up", up_donor)):
            if donor is not None and (
                donor.ndim != 3 or tuple(donor.shape[:2]) != tuple(positions.shape)
            ):
                raise ValueError(f"MLP {name} donor shape differs from positions")
        self.positions = positions
        self.gate_donor = gate_donor
        self.up_donor = up_donor
        self.fire_counts = {"gate": 0, "up": 0}

    def disarm(self) -> None:
        if self.positions is None:
            raise RuntimeError("MLP stage patch is not armed")
        expected = {
            name: 1
            for name, donor in (("gate", self.gate_donor), ("up", self.up_donor))
            if donor is not None
        }
        if any(self.fire_counts[name] != count for name, count in expected.items()):
            raise RuntimeError("MLP stage patch did not fire exactly once")
        self._clear()

    def _gate_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        self.fire_counts["gate"] += 1
        return self._patch(output, self.gate_donor, "gate")

    def _up_hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        self.fire_counts["up"] += 1
        return self._patch(output, self.up_donor, "up")

    def _patch(self, tensor: Any, donor: Any | None, name: str) -> Any:
        if donor is None:
            return tensor
        if self.positions is None or tensor.ndim != 3:
            raise RuntimeError(f"MLP {name} stage hook fired while unarmed")
        if tensor.shape[0] != self.positions.shape[0] or tensor.shape[-1] != donor.shape[-1]:
            raise ValueError(f"MLP {name} stage tensor shape mismatch")
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        positions = self.positions.to(tensor.device)
        local_donor = donor.to(tensor.device, dtype=tensor.dtype)
        # Reusing the exact native rows is a true identity operator.  Avoiding
        # an otherwise needless clone/write is important for low-precision
        # kernels: the write path can perturb downstream accumulation even
        # when the donor values are bit-identical to the native output.
        native = tensor[rows, positions]
        if _torch().equal(native, local_donor):
            return tensor
        modified = tensor.clone()
        modified[rows, positions] = local_donor
        return modified

    def _clear(self) -> None:
        self.positions = None
        self.gate_donor = None
        self.up_donor = None
        self.fire_counts = {"gate": 0, "up": 0}


def _torch() -> Any:
    import torch

    return torch
