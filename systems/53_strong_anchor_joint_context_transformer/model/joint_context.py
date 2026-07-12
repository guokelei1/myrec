"""Generic strong-anchor joint candidate/history Transformer for C53."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class JointContextOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    active_request: torch.Tensor


class StrongAnchorJointContextTransformer(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        dropout: float,
        max_history: int,
    ) -> None:
        super().__init__()
        if hidden_dim % heads or min(input_dim, hidden_dim, layers, max_history) <= 0:
            raise ValueError("C53 dimensions differ")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_history = int(max_history)
        self.content_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.base_projection = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.query_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_type = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_position = nn.Parameter(torch.zeros(max_history, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=heads, dim_feedforward=ffn_dim,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for value in (self.query_type, self.history_type, self.candidate_type, self.history_position):
            nn.init.normal_(value, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    @staticmethod
    def attention_mask(
        history_slots: int,
        candidate_slots: int,
        *,
        independent_candidates: bool,
        device: torch.device,
    ) -> torch.Tensor:
        context = 1 + history_slots
        length = context + candidate_slots
        mask = torch.zeros(length, length, dtype=torch.bool, device=device)
        mask[:context, context:] = True
        if independent_candidates:
            mask[context:, context:] = True
            positions = torch.arange(candidate_slots, device=device)
            mask[context + positions, context + positions] = False
        return mask

    def forward(
        self,
        *,
        query: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        independent_candidates: bool = False,
    ) -> JointContextOutput:
        if query.ndim != 2 or history.ndim != 3 or candidates.ndim != 3:
            raise ValueError("C53 input ranks differ")
        if history.shape[:2] != history_mask.shape or candidates.shape[:2] != candidate_mask.shape:
            raise ValueError("C53 masks differ")
        if base_scores.shape != candidate_mask.shape:
            raise ValueError("C53 base score shape differs")
        if query.shape[0] != history.shape[0] or query.shape[0] != candidates.shape[0]:
            raise ValueError("C53 batch differs")
        if query.shape[-1] != self.input_dim or history.shape[-1] != self.input_dim or candidates.shape[-1] != self.input_dim:
            raise ValueError("C53 input width differs")
        if history.shape[1] > self.max_history:
            raise ValueError("C53 history exceeds maximum")
        active = history_mask.any(dim=-1)
        q = self.content_projection(F.normalize(query.float(), dim=-1, eps=1e-6))
        h = self.content_projection(F.normalize(history.float(), dim=-1, eps=1e-6))
        c = self.content_projection(F.normalize(candidates.float(), dim=-1, eps=1e-6))
        q = q + self.query_type
        h = h + self.history_type + self.history_position[: history.shape[1]][None]
        c = c + self.candidate_type + self.base_projection(base_scores[..., None].float())
        sequence = torch.cat((q[:, None], h, c), dim=1)
        padding = torch.cat(
            (
                torch.zeros(len(query), 1, dtype=torch.bool, device=query.device),
                ~history_mask.bool(),
                ~candidate_mask.bool(),
            ),
            dim=1,
        )
        mask = self.attention_mask(
            history.shape[1], candidates.shape[1],
            independent_candidates=independent_candidates, device=query.device,
        )
        encoded = self.transformer(sequence, mask=mask, src_key_padding_mask=padding)
        candidate_state = encoded[:, 1 + history.shape[1] :]
        raw = self.output(self.output_norm(candidate_state)).squeeze(-1)
        weights = candidate_mask.to(raw.dtype)
        mean = (raw * weights).sum(dim=-1, keepdim=True) / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        correction = (raw - mean) * weights * active[:, None].to(raw.dtype)
        scores = (base_scores + correction).masked_fill(~candidate_mask, 0.0)
        return JointContextOutput(
            scores=scores, correction=correction, raw_correction=raw,
            active_request=active,
        )

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())
