"""Shared-LM paired-prefix ranking operator for C04."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModel


class LoRALinear(nn.Module):
    """A frozen linear projection plus an ordinary static LoRA update."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad = False
        self.lora_a = nn.Linear(base.in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = float(alpha) / float(rank)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(self.dropout(inputs))) * self.scale


def _load_d2_encoder(backbone: nn.Module, checkpoint_path: str | Path | None) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state", checkpoint)
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in state.items()
        if key.startswith("encoder.")
    }
    missing, unexpected = backbone.load_state_dict(encoder_state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected D2 encoder keys: {unexpected[:5]}")
    # Pooler parameters are absent from some sentence-transformer checkpoints.
    allowed_missing = {key for key in missing if key.startswith("pooler.")}
    if set(missing) != allowed_missing:
        raise ValueError(f"missing D2 encoder keys: {missing[:5]}")


def _inject_lora(backbone: nn.Module, rank: int, alpha: float, dropout: float, layers: int) -> list[str]:
    encoder_layers = backbone.encoder.layer
    start = max(len(encoder_layers) - int(layers), 0)
    replaced = []
    for index in range(start, len(encoder_layers)):
        attention = encoder_layers[index].attention.self
        for name in ("query", "value"):
            base = getattr(attention, name)
            setattr(attention, name, LoRALinear(base, rank, alpha, dropout))
            replaced.append(f"encoder.layer.{index}.attention.self.{name}")
    return replaced


class PrefixDeltaRanker(nn.Module):
    """Compact masked-LM ranker with a shared factual/null prefix operator.

    For a candidate pool, let `h` and `n` be candidate logits under the factual
    history and null-history prefixes from this same model.  C04 computes
    `d=h-n`, centers it across candidates, removes the component parallel to the
    centered null logit vector, bounds the remaining order-changing tangent,
    and returns `n+d_tangent`.  This is not a mix with an external D2p score.
    """

    VALID_MODES = {
        "paired_delta",
        "paired_no_tangent",
        "single_pass",
        "concat_head",
        "static_lora",
        "identity_shortcut",
    }

    def __init__(self, config: dict[str, Any], mode: str = "paired_delta") -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown model mode: {mode}")
        self.mode = mode
        model_cfg = config["model"]
        self.backbone = AutoModel.from_pretrained(
            model_cfg["backbone"],
            local_files_only=bool(model_cfg["local_files_only"]),
        )
        _load_d2_encoder(self.backbone, model_cfg.get("d2_initialization_checkpoint"))
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.lora_modules = _inject_lora(
            self.backbone,
            rank=int(model_cfg["lora_rank"]),
            alpha=float(model_cfg["lora_alpha"]),
            dropout=float(model_cfg["lora_dropout"]),
            layers=int(model_cfg["lora_layers"]),
        )
        hidden = int(self.backbone.config.hidden_size)
        self.score_head = nn.Linear(hidden, 1)
        self.identity_log_scale = nn.Parameter(torch.tensor(-1.0))
        self.delta_clip = float(model_cfg["delta_clip"])
        self.tangent_epsilon = float(model_cfg["tangent_epsilon"])

    def score(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self.backbone(**inputs)
        return self.score_head(output.last_hidden_state[:, 0, :].float()).squeeze(-1)

    @staticmethod
    def _masked_center(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype)
        mean = (values * weights).sum(dim=-1, keepdim=True) / weights.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        return (values - mean) * weights

    def tangent_delta(
        self,
        factual: torch.Tensor,
        null: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw_delta = (factual - null) * history_present.to(factual.dtype).unsqueeze(-1)
        centered_delta = self._masked_center(raw_delta, candidate_mask)
        centered_null = self._masked_center(null, candidate_mask)
        numerator = (centered_delta * centered_null).sum(dim=-1, keepdim=True)
        denominator = centered_null.square().sum(dim=-1, keepdim=True).clamp_min(
            self.tangent_epsilon
        )
        tangent = centered_delta - numerator / denominator * centered_null
        tangent = tangent * candidate_mask.to(tangent.dtype)
        bounded = self.delta_clip * torch.tanh(tangent / self.delta_clip)
        # Elementwise bounding can reintroduce a mean/base-parallel component;
        # project once more so the emitted delta remains in the order tangent.
        bounded = self._masked_center(bounded, candidate_mask)
        bounded_numerator = (bounded * centered_null).sum(dim=-1, keepdim=True)
        bounded = bounded - bounded_numerator / denominator * centered_null
        bounded = bounded * candidate_mask.to(bounded.dtype)
        bounded = bounded * history_present.to(bounded.dtype).unsqueeze(-1)
        return bounded, raw_delta

    def combine(
        self,
        factual: torch.Tensor,
        null: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_present: torch.Tensor,
        exact_repeat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tangent, raw_delta = self.tangent_delta(
            factual, null, candidate_mask, history_present
        )
        if self.mode == "paired_delta":
            final = null + tangent
        elif self.mode == "paired_no_tangent":
            final = null + self._masked_center(raw_delta, candidate_mask)
        elif self.mode in {"single_pass", "concat_head"}:
            final = factual
        elif self.mode == "static_lora":
            final = null
        else:
            if exact_repeat is None:
                raise ValueError("identity_shortcut requires exact_repeat")
            scale = torch.nn.functional.softplus(self.identity_log_scale)
            final = null + scale * exact_repeat.to(null.dtype)
        floor = -torch.finfo(final.dtype).max
        final = final.masked_fill(~candidate_mask, floor)
        return {
            "final": final,
            "factual": factual,
            "null": null,
            "raw_delta": raw_delta,
            "tangent_delta": tangent,
        }

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
