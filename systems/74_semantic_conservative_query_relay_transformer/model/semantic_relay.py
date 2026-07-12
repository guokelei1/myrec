"""C74 semantic-conservative two-hop query-relay Transformer."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


PRIMARY = "semantic_conservative_relay"
MODES = (
    PRIMARY,
    "coupled_value_relay",
    "pooled_semantic_relay",
    "factual_semantic_relay",
)


@dataclass(frozen=True)
class SemanticRelayOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    history_attention: torch.Tensor
    factual_candidate_attention: torch.Tensor
    null_candidate_attention: torch.Tensor
    factual_query_state: torch.Tensor
    active_request: torch.Tensor


def listwise_loss(
    output: SemanticRelayOutput, labels: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(-1) > 0
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    row = -(target * log_probability).sum(-1)
    return row[valid].mean() if bool(valid.any()) else output.scores.sum() * 0.0


class _ResidualRoute(nn.Module):
    def __init__(self, dim: int, rank: int, init_std: float) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=init_std)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value + self.up(self.down(value)), dim=-1, eps=1e-6)


class SemanticConservativeQueryRelayTransformer(nn.Module):
    """Learn routing and chronology while preserving raw LM-semantic values."""

    def __init__(
        self,
        *,
        dim: int,
        route_rank: int,
        max_history: int,
        mode: str,
        temperature: float,
        profile_scale: float,
        correction_scale: float,
        route_init_std: float,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C74 mode: {mode}")
        if min(dim, route_rank, max_history) <= 0:
            raise ValueError("C74 dimensions must be positive")
        if min(temperature, profile_scale, correction_scale, route_init_std) <= 0:
            raise ValueError("C74 scales must be positive")
        self.dim = int(dim)
        self.max_history = int(max_history)
        self.mode = mode
        self.temperature = float(temperature)
        self.profile_scale = float(profile_scale)
        self.correction_scale = float(correction_scale)
        self.history_route = _ResidualRoute(dim, route_rank, route_init_std)
        self.candidate_route = _ResidualRoute(dim, route_rank, route_init_std)
        self.chronology_bias = nn.Parameter(torch.zeros(max_history))

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def _history_to_query(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        active: torch.Tensor,
        *,
        coupled: bool,
        pooled: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        route_query = self.history_route(query)
        route_history = self.history_route(history)
        logits = torch.einsum("bqd,bhd->bqh", route_query, route_history)
        logits = logits / self.temperature
        logits = logits + self.chronology_bias[: history.shape[1]][None, None]
        logits = logits.masked_fill(~history_mask[:, None].bool(), -1e9)
        attention = torch.softmax(logits, dim=-1)
        raw_query = F.normalize(query, dim=-1, eps=1e-6)
        raw_history = F.normalize(history, dim=-1, eps=1e-6)
        if coupled:
            carrier_query = route_query
            carrier_history = route_history
        else:
            carrier_query = raw_query
            carrier_history = raw_history
        profile = torch.einsum("bqh,bhd->bqd", attention, carrier_history)
        if pooled:
            profile = profile.mean(dim=1, keepdim=True).expand_as(profile)
        factual = F.normalize(
            carrier_query
            + self.profile_scale
            * profile
            * active[:, None, None].to(profile.dtype),
            dim=-1,
            eps=1e-6,
        )
        return factual, attention

    def _candidate_energy(
        self,
        candidates: torch.Tensor,
        factual_query: torch.Tensor,
        null_query: torch.Tensor,
        *,
        coupled: bool,
        factual_only: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_candidates = F.normalize(candidates, dim=-1, eps=1e-6)
        raw_factual = F.normalize(factual_query, dim=-1, eps=1e-6)
        raw_null = F.normalize(null_query, dim=-1, eps=1e-6)
        route_candidates = self.candidate_route(raw_candidates)
        route_factual = self.candidate_route(raw_factual)
        route_null = self.candidate_route(raw_null)
        factual_logits = torch.einsum(
            "bcd,bqd->bcq", route_candidates, route_factual
        ) / self.temperature
        null_logits = torch.einsum(
            "bcd,bqd->bcq", route_candidates, route_null
        ) / self.temperature
        factual_attention = torch.softmax(factual_logits, dim=-1)
        null_attention = torch.softmax(null_logits, dim=-1)
        if coupled:
            carrier_candidates = route_candidates
            carrier_factual = route_factual
            carrier_null = route_null
        else:
            carrier_candidates = raw_candidates
            carrier_factual = raw_factual
            carrier_null = raw_null
        factual_similarity = torch.einsum(
            "bcd,bqd->bcq", carrier_candidates, carrier_factual
        )
        null_similarity = torch.einsum(
            "bcd,bqd->bcq", carrier_candidates, carrier_null
        )
        factual_energy = (factual_attention * factual_similarity).sum(-1)
        null_energy = (null_attention * null_similarity).sum(-1)
        raw = factual_energy if factual_only else factual_energy - null_energy
        return raw, factual_attention, null_attention

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
    ) -> SemanticRelayOutput:
        if query_tokens.ndim != 3 or query_tokens.shape[-1] != self.dim:
            raise ValueError("C74 query shape differs")
        if history_tokens.ndim != 3 or history_tokens.shape[-1] != self.dim:
            raise ValueError("C74 history shape differs")
        if candidate_tokens.ndim != 3 or candidate_tokens.shape[-1] != self.dim:
            raise ValueError("C74 candidate shape differs")
        if history_tokens.shape[:2] != history_mask.shape:
            raise ValueError("C74 history mask differs")
        if candidate_tokens.shape[:2] != candidate_mask.shape:
            raise ValueError("C74 candidate mask differs")
        if history_tokens.shape[1] > self.max_history:
            raise ValueError("C74 history exceeds maximum")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != base_scores.shape:
            raise ValueError("C74 score shape differs")
        active = history_mask.bool().any(-1) & query_present.bool()
        coupled = self.mode == "coupled_value_relay"
        pooled = self.mode == "pooled_semantic_relay"
        factual_only = self.mode == "factual_semantic_relay"
        raw_null = F.normalize(query_tokens, dim=-1, eps=1e-6)
        factual_query, history_attention = self._history_to_query(
            query_tokens,
            history_tokens,
            history_mask,
            active,
            coupled=coupled,
            pooled=pooled,
        )
        raw, factual_attention, null_attention = self._candidate_energy(
            candidate_tokens,
            factual_query,
            raw_null if not coupled else self.history_route(raw_null),
            coupled=coupled,
            factual_only=factual_only,
        )
        bounded = self.correction_scale * torch.tanh(raw)
        weight = candidate_mask.to(bounded.dtype)
        mean = (bounded * weight).sum(-1, keepdim=True) / weight.sum(
            -1, keepdim=True
        ).clamp_min(1.0)
        active = active & ~repeat_request.bool()
        correction = (
            (bounded - mean) * weight * active[:, None].to(bounded.dtype)
        )
        scores = base_scores.float() + correction.float()
        scores = torch.where(
            repeat_request[:, None].bool(), item_only_scores.float(), scores
        )
        scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        return SemanticRelayOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            history_attention=history_attention,
            factual_candidate_attention=factual_attention,
            null_candidate_attention=null_attention,
            factual_query_state=factual_query,
            active_request=active,
        )
