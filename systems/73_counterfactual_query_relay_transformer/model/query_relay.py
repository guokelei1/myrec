"""C73 counterfactual query-relay attention primitive."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


PRIMARY = "counterfactual_query_relay"
MODES = (
    PRIMARY,
    "late_state_difference",
    "pooled_query_relay",
    "factual_query_relay",
)


@dataclass(frozen=True)
class RelayOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    relay_state: torch.Tensor
    factual_query_state: torch.Tensor
    null_query_state: torch.Tensor
    active_request: torch.Tensor


def listwise_loss(
    output: RelayOutput, labels: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(-1) > 0
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    row = -(target * log_probability).sum(-1)
    if not bool(valid.any()):
        return output.scores.sum() * 0.0
    return row[valid].mean()


class CounterfactualQueryRelayTransformer(nn.Module):
    """Route personalized information through factual-minus-NULL query attention."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        max_history: int,
        mode: str,
        dropout: float,
        correction_cap: float,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C73 mode: {mode}")
        if hidden_dim % heads:
            raise ValueError("C73 hidden dimension must divide heads")
        if min(input_dim, hidden_dim, ffn_dim, max_history) <= 0:
            raise ValueError("C73 dimensions must be positive")
        if correction_cap <= 0:
            raise ValueError("C73 correction cap must be positive")
        self.mode = mode
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_history = int(max_history)
        self.correction_cap = float(correction_cap)

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_type = nn.Parameter(torch.empty(hidden_dim))
        self.candidate_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_position = nn.Parameter(torch.empty(max_history, hidden_dim))
        self.query_history_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.candidate_relay_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, batch_first=True
        )
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.relay_ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.relay_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for value in (
            self.query_type,
            self.history_type,
            self.candidate_type,
            self.history_position,
        ):
            nn.init.normal_(value, std=0.02)
        nn.init.normal_(self.output_head.weight, std=0.02)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @staticmethod
    def _safe_history_mask(mask: torch.Tensor) -> torch.Tensor:
        safe = mask.bool().clone()
        empty = ~safe.any(-1)
        if bool(empty.any()):
            safe[empty, 0] = True
        return safe

    def _query_trajectories(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        query_present: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q0 = self.input_projection(query) + self.query_type
        h = self.input_projection(history)
        h = h + self.history_type + self.history_position[: history.shape[1]][None]
        safe = self._safe_history_mask(history_mask)
        cross, _ = self.query_history_attention(
            q0, h, h, key_padding_mask=~safe, need_weights=False
        )
        active = history_mask.bool().any(-1) & query_present.bool()
        qh = self.query_norm(q0 + cross * active[:, None, None].to(cross.dtype))
        q0 = self.query_norm(q0)
        return qh, q0, h, active

    def _candidate_relay(
        self,
        candidates: torch.Tensor,
        qh: torch.Tensor,
        q0: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "pooled_query_relay":
            pooled_factual = qh.mean(dim=1)
            pooled_null = q0.mean(dim=1)
            factual_context = torch.stack(
                (pooled_factual, -pooled_factual), dim=1
            )
            null_context = torch.stack((pooled_null, -pooled_null), dim=1)
            factual, _ = self.candidate_relay_attention(
                candidates,
                factual_context,
                factual_context,
                need_weights=False,
            )
            null, _ = self.candidate_relay_attention(
                candidates, null_context, null_context, need_weights=False
            )
            return factual - null

        if self.mode == "late_state_difference":
            query_delta = (qh - q0).mean(dim=1, keepdim=True)
            factual_history = history + query_delta
            factual_context = torch.cat((q0, factual_history), dim=1)
            context_mask = torch.cat(
                (
                    torch.ones(
                        q0.shape[:2], dtype=torch.bool, device=q0.device
                    ),
                    history_mask.bool(),
                ),
                dim=1,
            )
            factual, _ = self.candidate_relay_attention(
                candidates,
                factual_context,
                factual_context,
                key_padding_mask=~context_mask,
                need_weights=False,
            )
            null, _ = self.candidate_relay_attention(
                candidates, q0, q0, need_weights=False
            )
            return factual - null

        factual, _ = self.candidate_relay_attention(
            candidates, qh, qh, need_weights=False
        )
        if self.mode == "factual_query_relay":
            return factual
        null, _ = self.candidate_relay_attention(
            candidates, q0, q0, need_weights=False
        )
        return factual - null

    def forward(
        self,
        *,
        query_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
        query_present: torch.Tensor,
    ) -> RelayOutput:
        if query_tokens.ndim != 3 or query_tokens.shape[-1] != self.input_dim:
            raise ValueError("C73 query token shape differs")
        if history_tokens.ndim != 3 or history_tokens.shape[-1] != self.input_dim:
            raise ValueError("C73 history token shape differs")
        if candidate_tokens.ndim != 3 or candidate_tokens.shape[-1] != self.input_dim:
            raise ValueError("C73 candidate token shape differs")
        if history_tokens.shape[:2] != history_mask.shape:
            raise ValueError("C73 history mask shape differs")
        if candidate_tokens.shape[:2] != candidate_mask.shape:
            raise ValueError("C73 candidate mask shape differs")
        if history_tokens.shape[1] > self.max_history:
            raise ValueError("C73 history exceeds maximum")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != base_scores.shape:
            raise ValueError("C73 score shape differs")

        qh, q0, history, has_evidence = self._query_trajectories(
            query_tokens, history_tokens, history_mask, query_present
        )
        candidates = self.input_projection(candidate_tokens) + self.candidate_type
        relay = self._candidate_relay(
            candidates, qh, q0, history, history_mask
        )
        relay_state = self.relay_norm(relay + self.relay_ffn(relay))
        raw = self.output_head(relay_state).squeeze(-1)
        bounded = self.correction_cap * torch.tanh(raw)
        weight = candidate_mask.to(bounded.dtype)
        mean = (bounded * weight).sum(-1, keepdim=True) / weight.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        active = has_evidence & ~repeat_request.bool()
        correction = (
            (bounded - mean)
            * weight
            * active[:, None].to(bounded.dtype)
        )
        scores = base_scores.float() + correction.float()
        scores = torch.where(
            repeat_request[:, None].bool(), item_only_scores.float(), scores
        )
        scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        return RelayOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            relay_state=relay_state,
            factual_query_state=qh,
            null_query_state=q0,
            active_request=active,
        )
