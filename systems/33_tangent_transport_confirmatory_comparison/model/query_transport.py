"""Capacity-matched tangent and unprojected query transports."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


PRIMARY = "tangent_transport_confirmation"
CONTROL = "unprojected_query_transport"
MODES = (PRIMARY, CONTROL)


class LowRankQueryTransport(nn.Module):
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
            raise ValueError(f"unknown C33 mode: {mode}")
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

    @property
    def uses_tangent_projection(self) -> bool:
        return self.mode == PRIMARY

    def adapt(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value + self.up(self.down(value)), dim=-1, eps=1e-6)

    def transport_state(
        self, query: torch.Tensor, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_query = F.normalize(query, dim=-1, eps=1e-6)
        raw_history = F.normalize(history, dim=-1, eps=1e-6)
        weights = torch.softmax(raw_history.mv(raw_query) / self.temperature, dim=0)
        adapted_query = self.adapt(raw_query)
        adapted_history = self.adapt(raw_history)
        profile = (weights[:, None] * adapted_history).sum(0)
        if self.uses_tangent_projection:
            displacement = profile - torch.dot(profile, adapted_query) * adapted_query
        else:
            displacement = profile
        transported = F.normalize(
            adapted_query + self.profile_scale * displacement, dim=-1, eps=1e-6
        )
        return adapted_query, displacement, transported

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
            raise ValueError("C33 embedding shape differs")
        if len(history) == 0:
            return torch.zeros(
                len(candidates), device=candidates.device, dtype=candidates.dtype
            )
        adapted_query, _, transported = self.transport_state(query, history)
        adapted_candidates = self.adapt(
            F.normalize(candidates, dim=-1, eps=1e-6)
        )
        return self.correction_scale * (
            adapted_candidates.mv(transported) - adapted_candidates.mv(adapted_query)
        )

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class FrozenBGETransportRanker(nn.Module):
    def __init__(self, snapshot: str, transport: LowRankQueryTransport) -> None:
        super().__init__()
        self.encoder = AutoModel.from_pretrained(snapshot, local_files_only=True)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.transport = transport

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
            else query.new_empty((0, self.transport.dim))
        )
        candidates = self.encode(candidate_input_ids, candidate_attention_mask)
        return self.transport(query[0], history, candidates)
