"""Candidate-axis barycentric residual transport and exact reductions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


PRIMARY = "barycentric_residual_transport"
GLOBAL_CONTROL = "global_tangent_transport"
UNCENTERED_CONTROL = "uncentered_additive_transport"
RELATIVE_ONLY_CONTROL = "relative_surplus_only"
MODES = (PRIMARY, GLOBAL_CONTROL, UNCENTERED_CONTROL, RELATIVE_ONLY_CONTROL)


class LowRankBarycentricResidualTransport(nn.Module):
    """Shared LM adapter plus a mean-preserving candidate contrast."""

    def __init__(
        self,
        *,
        dim: int,
        rank: int,
        temperature: float,
        profile_scale: float,
        correction_scale: float,
        seed: int,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C37 mode: {mode}")
        self.dim = int(dim)
        self.rank = int(rank)
        self.temperature = float(temperature)
        self.profile_scale = float(profile_scale)
        self.correction_scale = float(correction_scale)
        self.mode = mode
        self.down = nn.Linear(self.dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.dim, bias=False)
        generator = torch.Generator().manual_seed(int(seed))
        nn.init.normal_(self.down.weight, std=0.02, generator=generator)
        nn.init.zeros_(self.up.weight)

    def adapt(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value + self.up(self.down(value)), dim=-1, eps=1e-6)

    @staticmethod
    def tangent(value: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        if value.ndim == 1:
            return value - torch.dot(value, base) * base
        return value - value.mv(base)[:, None] * base[None, :]

    def relative_transport(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tangent = self.tangent(history, query)
        candidate_tangent = self.tangent(candidates, query)
        cosine = F.normalize(candidate_tangent, dim=-1, eps=1e-6) @ F.normalize(
            history_tangent, dim=-1, eps=1e-6
        ).T
        support = F.relu(cosine - cosine.mean(dim=0, keepdim=True))
        support_mass = support.sum(dim=1)
        admitted = support_mass > 0
        raw = support @ history_tangent / (1.0 + support_mass[:, None])
        centered = torch.zeros_like(raw)
        if bool(admitted.any()):
            centered[admitted] = raw[admitted] - raw[admitted].mean(
                dim=0, keepdim=True
            )
        return raw, centered, support, admitted

    def global_transport(
        self, query: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(history.mv(query) / self.temperature, dim=0)
        profile = (weights[:, None] * history).sum(dim=0)
        return self.tangent(profile, query), weights

    def transport_components(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, Any]:
        adapted_query = self.adapt(F.normalize(query, dim=-1, eps=1e-6))
        adapted_history = self.adapt(F.normalize(history, dim=-1, eps=1e-6))
        adapted_candidates = self.adapt(F.normalize(candidates, dim=-1, eps=1e-6))
        global_write, global_weights = self.global_transport(
            adapted_query, adapted_history
        )
        raw, centered, support, admitted = self.relative_transport(
            adapted_query, adapted_history, adapted_candidates
        )
        if self.mode == GLOBAL_CONTROL:
            residual = torch.zeros_like(raw)
            displacement = global_write[None, :].expand_as(raw)
        elif self.mode == UNCENTERED_CONTROL:
            residual = raw
            displacement = global_write[None, :] + residual
        elif self.mode == RELATIVE_ONLY_CONTROL:
            residual = raw
            displacement = residual
        else:
            residual = centered
            displacement = global_write[None, :] + residual
        transported = F.normalize(
            adapted_query[None, :] + self.profile_scale * displacement,
            dim=-1,
            eps=1e-6,
        )
        return {
            "adapted_query": adapted_query,
            "adapted_candidates": adapted_candidates,
            "global_write": global_write,
            "global_weights": global_weights,
            "raw_relative": raw,
            "centered_relative": centered,
            "support": support,
            "admitted": admitted,
            "residual": residual,
            "displacement": displacement,
            "transported": transported,
        }

    def transport_state(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.transport_components(query, history, candidates)
        return (
            state["adapted_query"],
            state["adapted_candidates"],
            state["displacement"],
            state["transported"],
            state["support"],
        )

    def forward(
        self, query: torch.Tensor, history: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        if (
            query.ndim != 1
            or history.ndim != 2
            or candidates.ndim != 2
            or query.shape[0] != self.dim
            or history.shape[1] != self.dim
            or candidates.shape[1] != self.dim
        ):
            raise ValueError("C37 embedding shape differs")
        if len(history) == 0:
            return torch.zeros(
                len(candidates), device=candidates.device, dtype=candidates.dtype
            )
        state = self.transport_components(query, history, candidates)
        return self.correction_scale * (
            (state["adapted_candidates"] * state["transported"]).sum(dim=1)
            - state["adapted_candidates"].mv(state["adapted_query"])
        )

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class FrozenBGEBarycentricResidualRanker(nn.Module):
    """End-to-end LM wrapper; cached states are an exact gate optimization."""

    def __init__(
        self, snapshot: str, attention: LowRankBarycentricResidualTransport
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(snapshot, local_files_only=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.attention = attention

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            state = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state[:, 0]
        return F.normalize(state.float(), dim=-1, eps=1e-6)

    def forward(
        self,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        history_input_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query = self.encode(query_input_ids, query_attention_mask)
        history = (
            self.encode(history_input_ids, history_attention_mask)
            if len(history_input_ids)
            else query.new_empty((0, self.attention.dim))
        )
        candidates = self.encode(candidate_input_ids, candidate_attention_mask)
        return self.attention(query[0], history, candidates)
