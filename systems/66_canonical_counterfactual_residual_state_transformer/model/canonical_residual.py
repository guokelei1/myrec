"""C66 zero-parameter canonical serialization wrapper around C65."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C65_MODEL_PATH = (
    REPO_ROOT
    / "systems/65_counterfactual_residual_state_transformer/model/counterfactual_residual.py"
)


def _load_c65() -> Any:
    name = "c66_runtime_c65_counterfactual_residual"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, C65_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("C66 cannot load frozen C65 model")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C65 = _load_c65()
MODES = C65.MODES
CounterfactualResidualOutput = C65.CounterfactualResidualOutput
counterfactual_training_loss = C65.counterfactual_training_loss
CounterfactualResidualStateTransformer = C65.CounterfactualResidualStateTransformer


class CanonicalCounterfactualResidualStateTransformer(nn.Module):
    """Sort by stable item key for every branch, then restore caller order."""

    CANDIDATE_NAMES = (
        "candidate_input_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.inner = CounterfactualResidualStateTransformer(**kwargs)

    @property
    def mode(self) -> str:
        return self.inner.mode

    @property
    def backbone(self) -> nn.Module:
        return self.inner.backbone

    @property
    def trainable_last_lm_layers(self) -> int:
        return self.inner.trainable_last_lm_layers

    @property
    def null_reference_stop_gradient(self) -> bool:
        return self.inner.null_reference_stop_gradient

    def parameter_count(self) -> int:
        return self.inner.parameter_count()

    def trainable_parameter_count(self) -> int:
        return self.inner.trainable_parameter_count()

    def backbone_trainable_names(self) -> list[str]:
        return self.inner.backbone_trainable_names()

    def backbone_layers(self) -> list[nn.Module]:
        return self.inner.backbone_layers()

    @staticmethod
    def _gather(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        index = order
        while index.ndim < value.ndim:
            index = index.unsqueeze(-1)
        return torch.gather(value, 1, index.expand(*value.shape[:1], order.shape[1], *value.shape[2:]))

    @staticmethod
    def _orders(
        candidate_keys: torch.Tensor, candidate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_keys.shape != candidate_mask.shape:
            raise ValueError("C66 candidate key shape differs")
        if candidate_keys.dtype != torch.int64:
            raise ValueError("C66 candidate keys must be int64")
        for row in range(candidate_keys.shape[0]):
            keys = candidate_keys[row, candidate_mask[row].bool()]
            if len(torch.unique(keys)) != len(keys):
                raise ValueError("C66 valid candidate keys must be unique")
        sentinel = torch.iinfo(torch.int64).max
        sort_keys = torch.where(
            candidate_mask.bool(), candidate_keys, torch.full_like(candidate_keys, sentinel)
        )
        order = torch.argsort(sort_keys, dim=1, stable=True)
        inverse = torch.argsort(order, dim=1, stable=True)
        return order, inverse

    def forward(
        self, *, candidate_keys: torch.Tensor, **values: torch.Tensor
    ) -> CounterfactualResidualOutput:
        order, inverse = self._orders(candidate_keys, values["candidate_mask"])
        canonical = dict(values)
        for name in self.CANDIDATE_NAMES:
            canonical[name] = self._gather(values[name], order)
        output = self.inner(**canonical)
        restored = {
            name: self._gather(getattr(output, name), inverse)
            for name in (
                "scores",
                "correction",
                "wrong_scores",
                "wrong_correction",
                "raw_correction",
                "raw_wrong_correction",
                "state_residual",
                "wrong_state_residual",
            )
        }
        return CounterfactualResidualOutput(
            **restored, active_request=output.active_request
        )
