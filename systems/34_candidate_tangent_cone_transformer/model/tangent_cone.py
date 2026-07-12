"""Candidate-specific tangent-cone attention and matched reductions."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


PRIMARY = "candidate_tangent_cone"
TARGET_CONTROL = "standard_target_attention"
GLOBAL_CONTROL = "global_tangent_transport"
MODES = (PRIMARY, TARGET_CONTROL, GLOBAL_CONTROL)


class LowRankTangentConeAttention(nn.Module):
    """One shared LM-space adapter with three parameter-identical read laws.

    The primary maps candidates and authenticated events into the tangent
    space at the adapted query.  An event may update a candidate only when its
    tangent direction has positive cosine with that candidate direction.  The
    denominator contains one fixed null unit, so all non-positive evidence is
    an exact no-write rather than a forced softmax assignment.
    """

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
            raise ValueError(f"unknown C34 mode: {mode}")
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

    def _candidate_cone_transport(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tangent = self.tangent(history, query)
        candidate_tangent = self.tangent(candidates, query)
        history_direction = F.normalize(history_tangent, dim=-1, eps=1e-6)
        candidate_direction = F.normalize(candidate_tangent, dim=-1, eps=1e-6)
        cosine = candidate_direction @ history_direction.T
        support = F.relu(cosine)
        denominator = 1.0 + support.sum(dim=1, keepdim=True)
        displacement = support @ history_tangent / denominator
        transported = F.normalize(
            query[None, :] + self.profile_scale * displacement,
            dim=-1,
            eps=1e-6,
        )
        return displacement, transported, support

    def _standard_target_transport(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_tangent = self.tangent(history, query)
        candidate_tangent = self.tangent(candidates, query)
        history_direction = F.normalize(history_tangent, dim=-1, eps=1e-6)
        candidate_direction = F.normalize(candidate_tangent, dim=-1, eps=1e-6)
        cosine = candidate_direction @ history_direction.T
        weights = torch.softmax(cosine / self.temperature, dim=1)
        displacement = weights @ history_tangent
        transported = F.normalize(
            query[None, :] + self.profile_scale * displacement,
            dim=-1,
            eps=1e-6,
        )
        return displacement, transported, weights

    def _global_transport(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weights = torch.softmax(history.mv(query) / self.temperature, dim=0)
        profile = (weights[:, None] * history).sum(dim=0)
        displacement = self.tangent(profile, query)
        transported_one = F.normalize(
            query + self.profile_scale * displacement, dim=-1, eps=1e-6
        )
        displacement_rows = displacement[None, :].expand(len(candidates), -1)
        transported = transported_one[None, :].expand(len(candidates), -1)
        return displacement_rows, transported, weights[None, :].expand(len(candidates), -1)

    def transport_state(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        adapted_query = self.adapt(F.normalize(query, dim=-1, eps=1e-6))
        adapted_history = self.adapt(F.normalize(history, dim=-1, eps=1e-6))
        adapted_candidates = self.adapt(F.normalize(candidates, dim=-1, eps=1e-6))
        if self.mode == PRIMARY:
            displacement, transported, mass = self._candidate_cone_transport(
                adapted_query, adapted_history, adapted_candidates
            )
        elif self.mode == TARGET_CONTROL:
            displacement, transported, mass = self._standard_target_transport(
                adapted_query, adapted_history, adapted_candidates
            )
        else:
            displacement, transported, mass = self._global_transport(
                adapted_query, adapted_history, adapted_candidates
            )
        return adapted_query, adapted_candidates, displacement, transported, mass

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
            raise ValueError("C34 embedding shape differs")
        if len(history) == 0:
            return torch.zeros(
                len(candidates), device=candidates.device, dtype=candidates.dtype
            )
        adapted_query, adapted_candidates, _, transported, _ = self.transport_state(
            query, history, candidates
        )
        factual = (adapted_candidates * transported).sum(dim=1)
        null = adapted_candidates.mv(adapted_query)
        return self.correction_scale * (factual - null)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class FrozenBGETangentConeRanker(nn.Module):
    """End-to-end execution wrapper; cached states are an exact gate shortcut."""

    def __init__(self, snapshot: str, attention: LowRankTangentConeAttention) -> None:
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
