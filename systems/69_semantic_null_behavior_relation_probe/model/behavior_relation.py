from __future__ import annotations

import torch
from torch import Tensor, nn


class BehaviorRelationTransformer(nn.Module):
    """Three-token Transformer with an exactly anchored pair interaction."""

    def __init__(
        self,
        *,
        input_dim: int,
        width: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        dropout: float,
        score_bound: float,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.score_bound = float(score_bound)
        self.input_projection = nn.Linear(input_dim, width)
        self.relation_token = nn.Parameter(torch.empty(width))
        self.role = nn.Parameter(torch.empty(3, width))
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(width)
        self.score_head = nn.Linear(width, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.relation_token, std=0.02)
        nn.init.normal_(self.role, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def raw_score(self, source: Tensor, target: Tensor) -> Tensor:
        if source.shape != target.shape or source.ndim != 2:
            raise ValueError("source and target must have equal [B,D] shapes")
        batch = source.shape[0]
        relation = self.relation_token[None].expand(batch, -1)
        projected = self.input_projection(torch.stack((source, target), dim=1))
        tokens = torch.cat((relation[:, None], projected), dim=1) + self.role[None]
        hidden = self.transformer(tokens)
        raw = self.score_head(self.output_norm(hidden[:, 0])).squeeze(-1)
        return self.score_bound * torch.tanh(raw / self.score_bound)

    def anchored_score(self, source: Tensor, target: Tensor) -> Tensor:
        zero_source = torch.zeros_like(source)
        zero_target = torch.zeros_like(target)
        # Evaluate in one batch so dropout/randomness cannot differ across the
        # four terms. Dropout is nevertheless locked to zero.
        sources = torch.cat((source, zero_source, source, zero_source), dim=0)
        targets = torch.cat((target, target, zero_target, zero_target), dim=0)
        f_hc, f_0c, f_h0, f_00 = self.raw_score(sources, targets).chunk(4)
        interaction = (f_hc - f_h0) - (f_0c - f_00)
        active = source.ne(0).any(dim=1) & target.ne(0).any(dim=1)
        return torch.where(active, interaction, torch.zeros_like(interaction))
