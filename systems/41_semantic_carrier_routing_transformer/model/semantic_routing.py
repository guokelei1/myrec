"""Learned history routing with an immutable LM-semantic content carrier."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


SEMANTIC_ROUTING = "semantic_routing"
SINGLE_WIDE_ROUTING = "single_wide_routing"
ASYMMETRIC_ROUTING = "asymmetric_routing"
COUPLED_CONTENT = "coupled_content"
MODES = (
    SEMANTIC_ROUTING,
    SINGLE_WIDE_ROUTING,
    ASYMMETRIC_ROUTING,
    COUPLED_CONTENT,
)


def fixed_semantic_correction(
    query: torch.Tensor,
    history: torch.Tensor,
    candidates: torch.Tensor,
    *,
    temperature: float,
    profile_scale: float,
    correction_scale: float,
) -> torch.Tensor:
    """Parameter-free semantic-attention transport control."""

    if len(history) == 0:
        return candidates.new_zeros(len(candidates))
    query = F.normalize(query, dim=-1, eps=1e-6)
    history = F.normalize(history, dim=-1, eps=1e-6)
    candidates = F.normalize(candidates, dim=-1, eps=1e-6)
    attention = torch.softmax(history.mv(query) / temperature, dim=0)
    profile = torch.einsum("j,jd->d", attention, history)
    transported = F.normalize(
        query + profile_scale * profile, dim=-1, eps=1e-6
    )
    return correction_scale * (
        candidates.mv(transported) - candidates.mv(query)
    )


class SemanticCarrierRoutingTransformer(nn.Module):
    """Route in learned metrics while carrying values in frozen LM space."""

    def __init__(
        self,
        *,
        dim: int,
        heads: int,
        rank: int,
        temperature: float,
        profile_scale: float,
        correction_scale: float,
        seed: int,
        mode: str,
        init_std: float = 0.01,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C41 mode: {mode}")
        if min(dim, heads, rank) <= 0:
            raise ValueError("C41 dimensions must be positive")
        if min(temperature, profile_scale, correction_scale, init_std) <= 0:
            raise ValueError("C41 scales must be positive")
        self.dim = int(dim)
        self.heads = int(heads)
        self.rank = int(rank)
        self.temperature = float(temperature)
        self.profile_scale = float(profile_scale)
        self.correction_scale = float(correction_scale)
        self.mode = mode
        self.down = nn.Parameter(torch.empty(self.heads, self.rank, self.dim))
        self.up = nn.Parameter(torch.empty(self.heads, self.dim, self.rank))
        generator = torch.Generator().manual_seed(int(seed))
        nn.init.normal_(self.down, mean=0.0, std=init_std, generator=generator)
        nn.init.normal_(self.up, mean=0.0, std=init_std, generator=generator)

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _adapt_all(self, value: torch.Tensor) -> torch.Tensor:
        low = torch.einsum("hrd,...d->h...r", self.down, value)
        residual = torch.einsum("hdr,h...r->h...d", self.up, low)
        return F.normalize(value.unsqueeze(0) + residual, dim=-1, eps=1e-6)

    def _adapt_wide(self, value: torch.Tensor) -> torch.Tensor:
        low = torch.einsum("hrd,...d->h...r", self.down, value)
        residual = torch.einsum("hdr,h...r->...d", self.up, low)
        return F.normalize(value + residual, dim=-1, eps=1e-6)

    def components(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, Any]:
        raw_query = F.normalize(query, dim=-1, eps=1e-6)
        raw_history = F.normalize(history, dim=-1, eps=1e-6)
        raw_candidates = F.normalize(candidates, dim=-1, eps=1e-6)

        if self.mode == SINGLE_WIDE_ROUTING:
            route_query = self._adapt_wide(raw_query).unsqueeze(0)
            route_history = self._adapt_wide(raw_history).unsqueeze(0)
            carrier_query = raw_query.unsqueeze(0)
            carrier_history = raw_history.unsqueeze(0)
            carrier_candidates = raw_candidates.unsqueeze(0)
            assignment = torch.zeros(1, dtype=torch.long, device=query.device)
        else:
            all_query = self._adapt_all(raw_query)
            all_history = self._adapt_all(raw_history)
            route_query = all_query
            if self.mode == ASYMMETRIC_ROUTING:
                assignment = torch.roll(
                    torch.arange(self.heads, device=query.device), shifts=-1
                )
                route_history = all_history[assignment]
            else:
                assignment = torch.arange(self.heads, device=query.device)
                route_history = all_history
            if self.mode == COUPLED_CONTENT:
                carrier_query = all_query
                carrier_history = all_history
                carrier_candidates = self._adapt_all(raw_candidates)
            else:
                carrier_query = raw_query.expand(self.heads, -1)
                carrier_history = raw_history.unsqueeze(0).expand(
                    self.heads, -1, -1
                )
                carrier_candidates = raw_candidates.unsqueeze(0).expand(
                    self.heads, -1, -1
                )

        logits = torch.einsum(
            "hd,hjd->hj", route_query, route_history
        ) / self.temperature
        attention = torch.softmax(logits, dim=-1)
        profile = torch.einsum("hj,hjd->hd", attention, carrier_history)
        transported = F.normalize(
            carrier_query + self.profile_scale * profile,
            dim=-1,
            eps=1e-6,
        )
        factual = torch.einsum(
            "hcd,hd->hc", carrier_candidates, transported
        )
        null = torch.einsum("hcd,hd->hc", carrier_candidates, carrier_query)
        per_head = factual - null
        return {
            "attention": attention,
            "profile": profile,
            "carrier_history": carrier_history,
            "transported_query": transported,
            "per_head_correction": per_head,
            "correction": self.correction_scale * per_head.mean(dim=0),
            "route_assignment": assignment,
            "semantic_carrier_exact": self.mode != COUPLED_CONTENT,
        }

    def forward(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
        *,
        query_present: bool = True,
        repeat_present: bool = False,
    ) -> torch.Tensor:
        if (
            query.ndim != 1
            or history.ndim != 2
            or candidates.ndim != 2
            or query.shape[-1] != self.dim
            or history.shape[-1] != self.dim
            or candidates.shape[-1] != self.dim
        ):
            raise ValueError("C41 embedding shape differs")
        if not query_present or repeat_present or len(history) == 0:
            return candidates.new_zeros(len(candidates))
        return self.components(query, history, candidates)["correction"]
