"""Causally Authenticated Mediation Transformer and matched controls."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


PRIMARY = "authenticated_mediation"
MODES = (
    PRIMARY,
    "unauthenticated_mediation",
    "authenticated_factual_only",
    "authenticated_random_init_same_transformer",
)


class AuthenticatedMediationTransformer(nn.Module):
    def __init__(self, snapshot: str, *, mode: str, correction_cap: float) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unexpected C29 mode: {mode}")
        self.mode = mode
        self.correction_cap = float(correction_cap)
        if self.correction_cap <= 0:
            raise ValueError("C29 correction cap must be positive")
        if mode == "authenticated_random_init_same_transformer":
            config = AutoConfig.from_pretrained(snapshot, local_files_only=True)
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(snapshot, local_files_only=True)
        for module in self.encoder.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        self.head = nn.Linear(int(self.encoder.config.hidden_size), 1, bias=False)
        nn.init.zeros_(self.head.weight)

    @property
    def uses_authentication(self) -> bool:
        return self.mode != "unauthenticated_mediation"

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        state = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.head(state).squeeze(-1)

    def correction_from_paired_logits(self, logits: torch.Tensor) -> torch.Tensor:
        paired = logits.reshape(-1, 2)
        raw = paired[:, 0] if self.mode == "authenticated_factual_only" else paired[:, 0] - paired[:, 1]
        return self.correction_cap * torch.tanh(raw)

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )

    def identity(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "uses_authentication": self.uses_authentication,
            "correction_cap": self.correction_cap,
            "encoder_model_type": str(self.encoder.config.model_type),
            "hidden_size": int(self.encoder.config.hidden_size),
            "parameters": self.parameter_count(),
            "trainable_parameters": self.parameter_count(trainable_only=True),
            "head_initialized_exact_zero": bool(torch.count_nonzero(self.head.weight) == 0),
            "dropout_disabled": all(
                module.p == 0.0 for module in self.encoder.modules() if isinstance(module, nn.Dropout)
            ),
        }
