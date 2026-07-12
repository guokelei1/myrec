"""C65 internal factual-minus-NULL candidate-state ranking core."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
from torch import nn
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C64_MODEL_PATH = (
    REPO_ROOT
    / "systems/64_end_to_end_lm_representation_probe/model/adaptive_joint_ranker.py"
)


def _load_c64_module() -> Any:
    name = "c65_runtime_c64_adaptive_joint_ranker"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, C64_MODEL_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("C65 cannot load the frozen C64 ranking core")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


C64 = _load_c64_module()
AdaptiveJointLMRanker = C64.AdaptiveJointLMRanker


MODES = (
    "hidden_residual_wrong_neutral",
    "hidden_residual_no_wrong",
    "ordinary_factual_wrong_neutral",
    "logit_difference_wrong_neutral",
)


@dataclass(frozen=True)
class CounterfactualResidualOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    wrong_scores: torch.Tensor
    wrong_correction: torch.Tensor
    raw_correction: torch.Tensor
    raw_wrong_correction: torch.Tensor
    state_residual: torch.Tensor
    wrong_state_residual: torch.Tensor
    active_request: torch.Tensor


def counterfactual_training_loss(
    output: CounterfactualResidualOutput,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    correction_l2_weight: float,
    wrong_neutrality_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(dim=-1) > 0
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1)
    row = -(target * log_probability.masked_fill(~mask, 0.0)).sum(dim=-1)
    ranking = row[valid].mean() if bool(valid.any()) else logits.sum() * 0.0
    active = output.active_request[:, None] & mask
    if bool(active.any()):
        energy = output.correction[active].square().mean()
        wrong_neutral = output.wrong_correction[active].square().mean()
    else:
        energy = output.correction.sum() * 0.0
        wrong_neutral = output.wrong_correction.sum() * 0.0
    total = (
        ranking
        + float(correction_l2_weight) * energy
        + float(wrong_neutrality_weight) * wrong_neutral
    )
    return total, {
        "ranking": ranking,
        "correction_l2": energy,
        "wrong_neutrality": wrong_neutral,
    }


class CounterfactualResidualStateTransformer(nn.Module):
    """Score only a shared LM's factual-minus-NULL internal state."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        mode: str,
        trainable_last_lm_layers: int,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        dropout: float,
        max_history: int,
        zero_initial_output: bool,
        null_reference_stop_gradient: bool,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C65 mode: {mode}")
        if not null_reference_stop_gradient:
            raise ValueError("C65 proposal requires stopped NULL reference")
        self.mode = mode
        self.null_reference_stop_gradient = bool(null_reference_stop_gradient)
        self.core = AdaptiveJointLMRanker(
            backbone=backbone,
            mode="adaptive_history_lm",
            trainable_last_lm_layers=trainable_last_lm_layers,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            heads=heads,
            layers=layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            max_history=max_history,
            zero_initial_output=zero_initial_output,
        )
        self.residual_norm = nn.LayerNorm(hidden_dim)

    @property
    def backbone(self) -> nn.Module:
        return self.core.backbone

    @property
    def trainable_last_lm_layers(self) -> int:
        return self.core.trainable_last_lm_layers

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def backbone_trainable_names(self) -> list[str]:
        return self.core.backbone_trainable_names()

    def backbone_layers(self) -> list[nn.Module]:
        return self.core._backbone_layers()

    @staticmethod
    def _core_kwargs(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        names = (
            "query_input_ids",
            "query_attention_mask",
            "query_content_mask",
            "candidate_input_ids",
            "candidate_attention_mask",
            "candidate_content_mask",
            "history_input_ids",
            "history_attention_mask",
            "history_content_mask",
            "history_event_mask",
            "candidate_mask",
            "base_scores",
            "item_only_scores",
            "repeat_request",
            "query_present",
        )
        return {name: values[name] for name in names}

    @staticmethod
    def _replace_history(
        values: Mapping[str, torch.Tensor],
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = dict(values)
        output["history_input_ids"] = input_ids
        output["history_attention_mask"] = attention_mask
        output["history_content_mask"] = content_mask
        output["history_event_mask"] = event_mask
        return output

    @staticmethod
    def _center(
        raw: torch.Tensor, candidate_mask: torch.Tensor
    ) -> torch.Tensor:
        weight = candidate_mask.to(raw.dtype)
        mean = (raw * weight).sum(-1, keepdim=True) / weight.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        return (raw - mean) * weight

    def _raw_from_states(
        self,
        factual_state: torch.Tensor,
        null_state: torch.Tensor,
        factual_logit: torch.Tensor,
        null_logit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reference = null_state.detach()
        residual = factual_state - reference
        if self.mode in {
            "hidden_residual_wrong_neutral",
            "hidden_residual_no_wrong",
        }:
            raw = self.core.output_head(self.residual_norm(residual)).squeeze(-1)
        elif self.mode == "ordinary_factual_wrong_neutral":
            raw = factual_logit
        else:
            raw = factual_logit - null_logit.detach()
        return raw, residual

    def forward(
        self,
        *,
        wrong_history_input_ids: torch.Tensor,
        wrong_history_attention_mask: torch.Tensor,
        wrong_history_content_mask: torch.Tensor,
        wrong_history_event_mask: torch.Tensor,
        **values: torch.Tensor,
    ) -> CounterfactualResidualOutput:
        factual = self.core(**self._core_kwargs(values))
        null_values = dict(values)
        null_values["history_event_mask"] = torch.zeros_like(values["history_event_mask"])
        with torch.no_grad():
            null = self.core(**self._core_kwargs(null_values))
        wrong_values = self._replace_history(
            values,
            input_ids=wrong_history_input_ids,
            attention_mask=wrong_history_attention_mask,
            content_mask=wrong_history_content_mask,
            event_mask=wrong_history_event_mask,
        )
        wrong = self.core(**self._core_kwargs(wrong_values))
        raw, residual = self._raw_from_states(
            factual.candidate_state,
            null.candidate_state,
            factual.raw_correction,
            null.raw_correction,
        )
        raw_wrong, wrong_residual = self._raw_from_states(
            wrong.candidate_state,
            null.candidate_state,
            wrong.raw_correction,
            null.raw_correction,
        )
        candidate_mask = values["candidate_mask"].bool()
        history_present = values["history_event_mask"].any(-1)
        active = (
            history_present
            & values["query_present"].bool()
            & ~values["repeat_request"].bool()
        )
        correction = self._center(raw, candidate_mask) * active[:, None].to(raw.dtype)
        wrong_correction = self._center(raw_wrong, candidate_mask) * active[:, None].to(
            raw_wrong.dtype
        )
        scores = values["base_scores"].float() + correction
        wrong_scores = values["base_scores"].float() + wrong_correction
        repeat = values["repeat_request"][:, None].bool()
        scores = torch.where(repeat, values["item_only_scores"].float(), scores)
        wrong_scores = torch.where(
            repeat, values["item_only_scores"].float(), wrong_scores
        )
        scores = scores.masked_fill(~candidate_mask, 0.0)
        wrong_scores = wrong_scores.masked_fill(~candidate_mask, 0.0)
        return CounterfactualResidualOutput(
            scores=scores,
            correction=correction,
            wrong_scores=wrong_scores,
            wrong_correction=wrong_correction,
            raw_correction=raw,
            raw_wrong_correction=raw_wrong,
            state_residual=residual,
            wrong_state_residual=wrong_residual,
            active_request=active,
        )
