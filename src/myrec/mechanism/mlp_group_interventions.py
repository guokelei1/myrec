"""Frozen SwiGLU feature groups and group-local Qwen MLP interventions."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


MLP_GROUPS = 16
MLP_GROUP_SEED = 20_260_718


def frozen_mlp_groups(intermediate_size: int) -> tuple[tuple[int, ...], ...]:
    """Assign every SwiGLU dimension to one of 16 near-equal hash groups."""

    intermediate_size = int(intermediate_size)
    if intermediate_size <= 0 or intermediate_size < MLP_GROUPS:
        raise ValueError("MLP intermediate size is too small for frozen groups")
    ordered = sorted(
        range(intermediate_size),
        key=lambda index: (
            hashlib.sha256(
                f"{MLP_GROUP_SEED}\0{index}".encode("utf-8")
            ).hexdigest(),
            index,
        ),
    )
    groups = tuple(
        tuple(sorted(ordered[group::MLP_GROUPS])) for group in range(MLP_GROUPS)
    )
    flattened = [index for group in groups for index in group]
    if sorted(flattened) != list(range(intermediate_size)):
        raise AssertionError("frozen MLP groups do not partition dimensions")
    if max(map(len, groups)) - min(map(len, groups)) > 1:
        raise AssertionError("frozen MLP groups are imbalanced")
    return groups


class QwenMLPGroupCapture:
    """Capture selected-token SwiGLU products before down projection."""

    def __init__(self, model: Any, block: int) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("MLP capture block must be in [0, 27]")
        self.layer = resolve_qwen_backbone(model).layers[block]
        self.positions: Any = None
        self.values: Any = None
        self.fire_count = 0
        self.handle: Any = None

    def __enter__(self) -> "QwenMLPGroupCapture":
        self.handle = self.layer.mlp.down_proj.register_forward_pre_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.positions = None
        self.values = None
        self.fire_count = 0

    def arm(self, positions: Any) -> None:
        if self.positions is not None or positions.ndim != 2:
            raise ValueError("MLP capture positions must be unarmed [batch,count]")
        self.positions = positions
        self.values = None
        self.fire_count = 0

    def disarm(self) -> Any:
        if self.fire_count != 1 or self.values is None:
            raise RuntimeError("MLP capture did not fire exactly once")
        values = self.values
        self.positions = None
        self.values = None
        self.fire_count = 0
        return values

    def _hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.positions is None or not inputs:
            raise RuntimeError("MLP capture fired while unarmed")
        tensor = inputs[0]
        if tensor.ndim != 3 or tensor.shape[0] != self.positions.shape[0]:
            raise ValueError("MLP SwiGLU product has unexpected shape")
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        positions = self.positions.to(tensor.device)
        self.values = tensor[rows, positions].detach()
        self.fire_count += 1


class QwenMLPGroupPatch:
    """Replace registered SwiGLU groups at selected token positions."""

    def __init__(
        self,
        model: Any,
        block: int,
        group_ids: Sequence[int],
    ) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("MLP patch block must be in [0, 27]")
        self.layer = resolve_qwen_backbone(model).layers[block]
        intermediate = int(self.layer.mlp.down_proj.in_features)
        groups = frozen_mlp_groups(intermediate)
        normalized = tuple(sorted(set(int(value) for value in group_ids)))
        if not normalized or any(not 0 <= value < MLP_GROUPS for value in normalized):
            raise ValueError("MLP patch group IDs are invalid")
        indices = sorted(index for group in normalized for index in groups[group])
        self.group_ids = normalized
        self.indices = _torch().tensor(indices, dtype=_torch().long)
        self.positions: Any = None
        self.donor: Any = None
        self.fire_count = 0
        self.handle: Any = None

    def __enter__(self) -> "QwenMLPGroupPatch":
        self.handle = self.layer.mlp.down_proj.register_forward_pre_hook(
            self._hook
        )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.positions = None
        self.donor = None
        self.fire_count = 0

    def arm(self, positions: Any, donor: Any) -> None:
        if self.positions is not None:
            raise RuntimeError("MLP group patch is already armed")
        if positions.ndim != 2 or donor.ndim != 3 or donor.shape[:2] != positions.shape:
            raise ValueError("MLP group donor arrays are misaligned")
        self.positions = positions
        self.donor = donor
        self.fire_count = 0

    def disarm(self) -> None:
        if self.positions is None or self.fire_count != 1:
            raise RuntimeError("MLP group patch did not fire exactly once")
        self.positions = None
        self.donor = None
        self.fire_count = 0

    def _hook(self, _module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if self.positions is None or self.donor is None or not inputs:
            raise RuntimeError("MLP group patch fired while unarmed")
        tensor = inputs[0]
        if tensor.ndim != 3 or tensor.shape[0] != self.positions.shape[0]:
            raise ValueError("MLP group patch tensor shape mismatch")
        if self.donor.shape[-1] != tensor.shape[-1]:
            raise ValueError("MLP group donor intermediate size mismatch")
        positions = self.positions.to(tensor.device)
        indices = self.indices.to(tensor.device)
        modified = tensor.clone()
        donor = self.donor.to(tensor.device, dtype=tensor.dtype)
        for row in range(tensor.shape[0]):
            for column in range(positions.shape[1]):
                position = int(positions[row, column])
                modified[row, position, indices] = donor[row, column, indices]
        self.fire_count += 1
        return (modified, *inputs[1:])


def exact_permutation_recomposition(
    product: Any, down_weight: Any, permutation: Any
) -> tuple[Any, Any, Any]:
    """Return original and permutation/inverse-column down-projection outputs."""

    torch = _torch()
    permutation = permutation.to(product.device, dtype=torch.long)
    if permutation.ndim != 1 or len(permutation) != product.shape[-1]:
        raise ValueError("MLP permutation has invalid shape")
    if sorted(permutation.cpu().tolist()) != list(range(product.shape[-1])):
        raise ValueError("MLP permutation is not bijective")
    original = torch.nn.functional.linear(product, down_weight)
    permuted_product = product[..., permutation]
    permuted_weight = down_weight[:, permutation]
    recomposed = torch.nn.functional.linear(permuted_product, permuted_weight)
    return original, recomposed, permuted_product


def _torch() -> Any:
    import torch

    return torch
