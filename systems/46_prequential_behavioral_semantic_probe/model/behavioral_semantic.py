from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BehavioralSemanticTransformer(nn.Module):
    """Content-initialized Transformer used only as a behavioral signal probe."""

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

    def encode_history(
        self, history: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        if history.ndim != 3 or history_mask.shape != history.shape[:2]:
            raise ValueError("C46 history/mask shape differs")
        if history.shape[1] > self.max_history:
            raise ValueError("C46 history exceeds maximum")
        batch, length, _ = history.shape
        projected = self.item_projection(history)
        projected = projected + self.position[:length][None]
        read = self.read_token[None, None].expand(batch, 1, -1)
        values = torch.cat((read, projected), dim=1)
        key_padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=history.device),
                ~history_mask,
            ),
            dim=1,
        )
        state = self.output_norm(self.transformer(values, src_key_padding_mask=key_padding)[:, 0])
        state = F.normalize(state, dim=-1, eps=1e-6)
        return torch.where(history_mask.any(1, keepdim=True), state, torch.zeros_like(state))

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.item_projection(items), dim=-1, eps=1e-6)

    def score(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        profile = self.encode_history(history, history_mask)
        items = self.encode_items(candidates)
        score = torch.einsum("bd,bcd->bc", profile, items) / self.temperature
        return torch.where(history_mask.any(1, keepdim=True), score, torch.zeros_like(score))
