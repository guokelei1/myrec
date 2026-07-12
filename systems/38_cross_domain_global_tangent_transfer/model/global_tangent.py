"""Candidate-shared global tangent transfer and exact reductions."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


PRIMARY = "query_attended_tangent"
UNPROJECTED = "query_attended_unprojected"
MEAN_TANGENT = "mean_history_tangent"
MODES = (PRIMARY, UNPROJECTED, MEAN_TANGENT)


class LowRankGlobalTangentTransfer(nn.Module):
    """Shared low-rank LM-state adapter with one global history write."""

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
            raise ValueError(f"unknown C38 mode: {mode}")
        if dim <= 0 or rank <= 0:
            raise ValueError("dim and rank must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
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
        return value - torch.dot(value, base) * base

    def transport_components(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, Any]:
        adapted_query = self.adapt(F.normalize(query, dim=-1, eps=1e-6))
        adapted_history = self.adapt(F.normalize(history, dim=-1, eps=1e-6))
        adapted_candidates = self.adapt(F.normalize(candidates, dim=-1, eps=1e-6))
        if self.mode == MEAN_TANGENT:
            weights = history.new_full((len(history),), 1.0 / len(history))
        else:
            weights = torch.softmax(
                adapted_history.mv(adapted_query) / self.temperature,
                dim=0,
            )
        profile = (weights[:, None] * adapted_history).sum(dim=0)
        if self.mode == UNPROJECTED:
            displacement = profile
        else:
            displacement = self.tangent(profile, adapted_query)
        transported_query = F.normalize(
            adapted_query + self.profile_scale * displacement,
            dim=-1,
            eps=1e-6,
        )
        return {
            "adapted_query": adapted_query,
            "adapted_history": adapted_history,
            "adapted_candidates": adapted_candidates,
            "weights": weights,
            "profile": profile,
            "displacement": displacement,
            "transported_query": transported_query,
        }

    def forward(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
        *,
        query_present: bool = True,
    ) -> torch.Tensor:
        if (
            query.ndim != 1
            or history.ndim != 2
            or candidates.ndim != 2
            or query.shape[0] != self.dim
            or history.shape[1] != self.dim
            or candidates.shape[1] != self.dim
        ):
            raise ValueError("C38 embedding shape differs")
        if len(history) == 0 or not query_present:
            return torch.zeros(
                len(candidates),
                device=candidates.device,
                dtype=candidates.dtype,
            )
        state = self.transport_components(query, history, candidates)
        transported = state["adapted_candidates"].mv(state["transported_query"])
        base = state["adapted_candidates"].mv(state["adapted_query"])
        return self.correction_scale * (transported - base)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class FrozenBGEGlobalTangentRanker(nn.Module):
    """Transformer wrapper; cached hidden states are an exact gate optimization."""

    def __init__(
        self,
        snapshot: str,
        transport: LowRankGlobalTangentTransfer,
    ) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(snapshot, local_files_only=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.transport = transport

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            state = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
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
            else query.new_empty((0, self.transport.dim))
        )
        candidates = self.encode(candidate_input_ids, candidate_attention_mask)
        return self.transport(query[0], history, candidates)
