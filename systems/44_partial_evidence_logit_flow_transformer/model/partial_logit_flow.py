"""Candidate-conservative partial evidence flow for listwise ranking."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PARTIAL_LOGIT_FLOW = "partial_logit_flow"
FORCED_LOGIT_FLOW = "forced_logit_flow"
PARTIAL_VECTOR_WRITE = "partial_vector_write"
GLOBAL_VECTOR_WRITE = "global_vector_write"
MODES = (
    PARTIAL_LOGIT_FLOW,
    FORCED_LOGIT_FLOW,
    PARTIAL_VECTOR_WRITE,
    GLOBAL_VECTOR_WRITE,
)


class PartialEvidenceLogitFlowTransformer(nn.Module):
    """Allocate each history event across candidates and a null sink.

    The primary converts partial assignment mass directly into a centered
    candidate-logit flow. No pooled history vector or learned scalar head sits
    between evidence allocation and ranking.
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
            raise ValueError(f"unknown C44 mode: {mode}")
        if min(dim, heads, rank) <= 0:
            raise ValueError("C44 dimensions must be positive")
        if min(temperature, profile_scale, correction_scale, init_std) <= 0:
            raise ValueError("C44 scales must be positive")
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
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _adapt_all(self, value: torch.Tensor) -> torch.Tensor:
        low = torch.einsum("hrd,...d->h...r", self.down, value)
        residual = torch.einsum("hdr,h...r->h...d", self.up, low)
        return F.normalize(value.unsqueeze(0) + residual, dim=-1, eps=1e-6)

    @staticmethod
    def _center(values: torch.Tensor) -> torch.Tensor:
        return values - values.mean(dim=-1, keepdim=True)

    def components(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, Any]:
        query = F.normalize(query, dim=-1, eps=1e-6)
        history = F.normalize(history, dim=-1, eps=1e-6)
        candidates = F.normalize(candidates, dim=-1, eps=1e-6)
        q = self._adapt_all(query)
        h = self._adapt_all(history)
        c = self._adapt_all(candidates)
        event_transport = F.normalize(
            q[:, None, :] + self.profile_scale * h,
            dim=-1,
            eps=1e-6,
        )
        factual = torch.einsum("hcd,hjd->hcj", c, event_transport)
        null_score = torch.einsum("hcd,hd->hc", c, q)
        surplus = factual - null_score[:, :, None]

        candidate_mass: torch.Tensor | None = None
        null_mass: torch.Tensor | None = None
        if self.mode in (PARTIAL_LOGIT_FLOW, PARTIAL_VECTOR_WRITE):
            logits = surplus / self.temperature
            allocation = torch.softmax(
                torch.cat(
                    [logits, logits.new_zeros(self.heads, 1, len(history))], dim=1
                ),
                dim=1,
            )
            candidate_mass = allocation[:, :-1, :]
            null_mass = allocation[:, -1, :]
        elif self.mode == FORCED_LOGIT_FLOW:
            candidate_mass = torch.softmax(surplus / self.temperature, dim=1)
            null_mass = surplus.new_zeros(self.heads, len(history))

        if self.mode in (PARTIAL_LOGIT_FLOW, FORCED_LOGIT_FLOW):
            assert candidate_mass is not None
            per_head = self._center(candidate_mass.mean(dim=-1))
        elif self.mode == PARTIAL_VECTOR_WRITE:
            assert candidate_mass is not None
            displacement = torch.einsum("hcj,hjd->hcd", candidate_mass, h)
            displacement = displacement / (1.0 + candidate_mass.sum(dim=-1, keepdim=True))
            transported = F.normalize(
                q[:, None, :] + self.profile_scale * displacement,
                dim=-1,
                eps=1e-6,
            )
            per_head = self._center(
                torch.einsum("hcd,hcd->hc", c, transported) - null_score
            )
        else:
            profile = h.mean(dim=1)
            transported = F.normalize(
                q + self.profile_scale * profile,
                dim=-1,
                eps=1e-6,
            )
            per_head = self._center(
                torch.einsum("hcd,hd->hc", c, transported) - null_score
            )

        correction = self.correction_scale * per_head.mean(dim=0)
        return {
            "adapted_query": q,
            "adapted_history": h,
            "adapted_candidates": c,
            "event_transport": event_transport,
            "surplus": surplus,
            "candidate_mass": candidate_mass,
            "null_mass": null_mass,
            "per_head_correction": per_head,
            "correction": correction,
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
            raise ValueError("C44 embedding shape differs")
        if not query_present or repeat_present or len(history) == 0:
            return candidates.new_zeros(len(candidates))
        return self.components(query, history, candidates)["correction"]
