"""History-only value transport through candidate competition."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("history_carrier", "independent_carrier", "factual_carrier", "raw_candidate")


@dataclass(frozen=True)
class CarrierOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    carrier: torch.Tensor
    list_message: torch.Tensor
    active_request: torch.Tensor


class HistoryCarrierCompetitionTransformer(nn.Module):
    """Candidate list attention whose value stream contains only history delta."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
        max_history: int,
    ) -> None:
        super().__init__()
        if hidden_dim % heads or min(input_dim, hidden_dim, ffn_dim, max_history) <= 0:
            raise ValueError("C54 dimensions differ")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_history = int(max_history)
        self.content_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.base_projection = nn.Sequential(
            nn.Linear(1, hidden_dim, bias=False), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        self.query_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_type = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_position = nn.Parameter(torch.zeros(max_history, hidden_dim))
        # Bias-free attention is binding: a zero carrier must remain zero.
        self.context_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True,
        )
        self.list_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True,
        )
        self.carrier_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False), nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for value in (
            self.query_type, self.history_type, self.candidate_type,
            self.history_position,
        ):
            nn.init.normal_(value, std=0.02)
        nn.init.normal_(self.output.weight, std=0.02)

    @staticmethod
    def independent_mask(candidate_slots: int, device: torch.device) -> torch.Tensor:
        return ~torch.eye(candidate_slots, dtype=torch.bool, device=device)

    def forward(
        self,
        *,
        query: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        mode: str = "history_carrier",
    ) -> CarrierOutput:
        if mode not in MODES:
            raise ValueError(f"unknown C54 mode: {mode}")
        if query.ndim != 2 or history.ndim != 3 or candidates.ndim != 3:
            raise ValueError("C54 input ranks differ")
        if history.shape[:2] != history_mask.shape or candidates.shape[:2] != candidate_mask.shape:
            raise ValueError("C54 masks differ")
        if base_scores.shape != candidate_mask.shape:
            raise ValueError("C54 base score shape differs")
        if query.shape[0] != history.shape[0] or query.shape[0] != candidates.shape[0]:
            raise ValueError("C54 batch differs")
        if query.shape[-1] != self.input_dim or history.shape[-1] != self.input_dim or candidates.shape[-1] != self.input_dim:
            raise ValueError("C54 input width differs")
        if history.shape[1] > self.max_history:
            raise ValueError("C54 history exceeds maximum")

        active = history_mask.any(dim=-1)
        q = self.content_projection(F.normalize(query.float(), dim=-1, eps=1e-6))
        h = self.content_projection(F.normalize(history.float(), dim=-1, eps=1e-6))
        c = self.content_projection(F.normalize(candidates.float(), dim=-1, eps=1e-6))
        q = q + self.query_type
        h = h + self.history_type + self.history_position[: history.shape[1]][None]
        c = c + self.candidate_type + self.base_projection(base_scores[..., None].float())

        context = torch.cat((q[:, None], h), dim=1)
        factual_padding = torch.cat(
            (torch.zeros(len(q), 1, dtype=torch.bool, device=q.device), ~history_mask.bool()),
            dim=1,
        )
        null_padding = factual_padding.clone()
        null_padding[:, 1:] = True
        factual, _ = self.context_attention(
            c, context, context, key_padding_mask=factual_padding, need_weights=False,
        )
        null, _ = self.context_attention(
            c, context, context, key_padding_mask=null_padding, need_weights=False,
        )
        carrier = factual - null

        if mode in {"history_carrier", "independent_carrier"}:
            value, local = carrier, carrier
        elif mode == "factual_carrier":
            value, local = factual, factual
        else:
            value, local = c, c
        attention_mask = (
            self.independent_mask(c.shape[1], c.device)
            if mode == "independent_carrier" else None
        )
        list_message, _ = self.list_attention(
            c, c, value, attn_mask=attention_mask,
            # With a diagonal mask, padded queries would otherwise have every
            # key masked and produce NaN.  Valid queries can read only their
            # own valid slot, so omitting key padding is exact for them.
            key_padding_mask=(None if mode == "independent_carrier" else ~candidate_mask.bool()),
            need_weights=False,
        )
        state = self.carrier_norm(local + list_message)
        state = state + self.ffn(state)
        raw = self.output(self.output_norm(state)).squeeze(-1)
        weights = candidate_mask.to(raw.dtype)
        mean = (raw * weights).sum(dim=-1, keepdim=True) / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        correction = (raw - mean) * weights * active[:, None].to(raw.dtype)
        scores = (base_scores + correction).masked_fill(~candidate_mask, 0.0)
        return CarrierOutput(
            scores=scores, correction=correction, raw_correction=raw,
            carrier=carrier, list_message=list_message, active_request=active,
        )

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())
