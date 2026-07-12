"""Compact causal-prefix semantic predictor used by C49."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class PrequentialSemanticTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        width: int,
        heads: int,
        layers: int,
        ff_multiplier: int,
        max_history: int,
        temperature: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.max_history = int(max_history)
        self.temperature = float(temperature)
        self.item_projection = nn.Linear(input_dim, width)
        self.read_token = nn.Parameter(torch.empty(width))
        self.position = nn.Parameter(torch.empty(max_history, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff_multiplier * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(width)
        nn.init.normal_(self.read_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def encode_items(self, values: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_projection(values), dim=-1, eps=1e-6)

    def predict_next(self, prefixes: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if prefixes.ndim != 3 or mask.shape != prefixes.shape[:2]:
            raise ValueError("C49 prefix/mask shape differs")
        if prefixes.shape[1] > self.max_history:
            raise ValueError("C49 prefix exceeds maximum history")
        batch, length, _ = prefixes.shape
        projected = self.item_projection(prefixes) + self.position[:length][None]
        read = self.read_token[None, None].expand(batch, 1, -1)
        tokens = torch.cat((read, projected), dim=1)
        padding = torch.cat(
            (torch.zeros(batch, 1, dtype=torch.bool, device=mask.device), ~mask),
            dim=1,
        )
        state = self.output_norm(self.transformer(tokens, src_key_padding_mask=padding)[:, 0])
        state = F.normalize(state, dim=-1, eps=1e-6)
        return torch.where(mask.any(1, keepdim=True), state, torch.zeros_like(state))

    def contrastive_logits(
        self, prefixes: torch.Tensor, mask: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return self.predict_next(prefixes, mask) @ self.encode_items(targets).T / self.temperature
