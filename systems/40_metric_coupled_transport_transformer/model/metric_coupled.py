"""Metric-coupled query/history transport with matched loop reductions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MULTIHEAD_COUPLED = "multihead_coupled"
SINGLE_WIDE_COUPLED = "single_wide_coupled"
SELECTION_ONLY = "selection_only"
SHIFTED_LOOP = "shifted_loop"
MODES = (
    MULTIHEAD_COUPLED,
    SINGLE_WIDE_COUPLED,
    SELECTION_ONLY,
    SHIFTED_LOOP,
)


class MetricCoupledTransportTransformer(nn.Module):
    """One semantic metric map closes each head's evidence loop.

    All modes own identical parameter tensors. Only the information-flow
    wiring differs, so paired initialization and capacity are exact.
    """

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
            raise ValueError(f"unknown C40 mode: {mode}")
        if min(dim, heads, rank) <= 0:
            raise ValueError("C40 dimensions must be positive")
        if temperature <= 0 or profile_scale <= 0 or correction_scale <= 0:
            raise ValueError("C40 scales must be positive")
        if init_std <= 0:
            raise ValueError("C40 init_std must be positive")
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

    def _transport_score(
        self,
        query_state: torch.Tensor,
        history_state: torch.Tensor,
        candidate_state: torch.Tensor,
        attention_query: torch.Tensor,
        attention_history: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = torch.einsum(
            "hd,hjd->hj", attention_query, attention_history
        ) / self.temperature
        attention = torch.softmax(logits, dim=-1)
        profile = torch.einsum("hj,hjd->hd", attention, history_state)
        transported = F.normalize(
            query_state + self.profile_scale * profile,
            dim=-1,
            eps=1e-6,
        )
        factual = torch.einsum("hcd,hd->hc", candidate_state, transported)
        null = torch.einsum("hcd,hd->hc", candidate_state, query_state)
        per_head = factual - null
        return per_head.mean(dim=0), {
            "attention": attention,
            "profile": profile,
            "transported_query": transported,
            "per_head_correction": per_head,
        }

    def components(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, Any]:
        query = F.normalize(query, dim=-1, eps=1e-6)
        history = F.normalize(history, dim=-1, eps=1e-6)
        candidates = F.normalize(candidates, dim=-1, eps=1e-6)

        if self.mode == SINGLE_WIDE_COUPLED:
            q = self._adapt_wide(query).unsqueeze(0)
            h = self._adapt_wide(history).unsqueeze(0)
            c = self._adapt_wide(candidates).unsqueeze(0)
            correction, state = self._transport_score(q, h, c, q, h)
            assignment = torch.zeros(1, dtype=torch.long, device=query.device)
        else:
            selected_q = self._adapt_all(query)
            selected_h = self._adapt_all(history)
            if self.mode == SELECTION_ONLY:
                q = query.expand(self.heads, -1)
                h = history.unsqueeze(0).expand(self.heads, -1, -1)
                c = candidates.unsqueeze(0).expand(self.heads, -1, -1)
                assignment = torch.full(
                    (self.heads,), -1, dtype=torch.long, device=query.device
                )
            elif self.mode == SHIFTED_LOOP:
                assignment = torch.roll(
                    torch.arange(self.heads, device=query.device), shifts=-1
                )
                q = selected_q[assignment]
                h = selected_h[assignment]
                c = self._adapt_all(candidates)[assignment]
            else:
                q = selected_q
                h = selected_h
                c = self._adapt_all(candidates)
                assignment = torch.arange(self.heads, device=query.device)
            correction, state = self._transport_score(
                q, h, c, selected_q, selected_h
            )

        return {
            **state,
            "correction": self.correction_scale * correction,
            "loop_assignment": assignment,
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
            raise ValueError("C40 embedding shape differs")
        if not query_present or repeat_present or len(history) == 0:
            return candidates.new_zeros(len(candidates))
        return self.components(query, history, candidates)["correction"]
