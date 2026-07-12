"""Candidate-relative, eventwise score-halfspace value projection."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


EVENTWISE_HALFSPACE = "eventwise_halfspace"
EVENTWISE_RAW = "eventwise_raw"
POSTPOOL_HALFSPACE = "postpool_halfspace"
RAY_ONLY = "ray_only"
GLOBAL_ONLY = "global_only"
MODES = (
    EVENTWISE_HALFSPACE,
    EVENTWISE_RAW,
    POSTPOOL_HALFSPACE,
    RAY_ONLY,
    GLOBAL_ONLY,
)


def project_to_score_halfspace(
    values: torch.Tensor,
    normals: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Euclidean projection onto ``{u: <normal,u> >= 0}``."""

    if values.shape != normals.shape:
        values, normals = torch.broadcast_tensors(values, normals)
    dot = (values * normals).sum(dim=-1, keepdim=True)
    norm_squared = normals.square().sum(dim=-1, keepdim=True)
    return values + F.relu(-dot) * normals / (norm_squared + eps)


def score_ray(
    values: torch.Tensor,
    normals: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Keep only the positive score-normal component of a value."""

    if values.shape != normals.shape:
        values, normals = torch.broadcast_tensors(values, normals)
    dot = (values * normals).sum(dim=-1, keepdim=True)
    norm_squared = normals.square().sum(dim=-1, keepdim=True)
    return F.relu(dot) * normals / (norm_squared + eps)


class HalfspaceCertifiedValueTransformer(nn.Module):
    """Minimal multi-head Transformer block for the C39 design gate."""

    def __init__(
        self,
        *,
        dim: int,
        inner_dim: int,
        heads: int,
        ffn_dim: int,
        temperature: float,
        global_scale: float,
        candidate_scale: float,
        seed: int,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C39 mode: {mode}")
        if dim <= 0 or inner_dim <= 0 or ffn_dim <= 0 or heads <= 0:
            raise ValueError("C39 dimensions must be positive")
        if inner_dim % heads:
            raise ValueError("C39 inner_dim must be divisible by heads")
        if temperature <= 0:
            raise ValueError("C39 temperature must be positive")
        self.dim = int(dim)
        self.inner_dim = int(inner_dim)
        self.heads = int(heads)
        self.head_dim = self.inner_dim // self.heads
        self.ffn_dim = int(ffn_dim)
        self.temperature = float(temperature)
        self.global_scale = float(global_scale)
        self.candidate_scale = float(candidate_scale)
        self.mode = mode

        self.q_proj = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.k_proj = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.v_proj = nn.Linear(self.dim, self.inner_dim, bias=False)
        self.out_proj = nn.Linear(self.inner_dim, self.dim, bias=False)
        self.ffn_norm = nn.LayerNorm(self.dim)
        self.ffn_up = nn.Linear(self.dim, self.ffn_dim, bias=False)
        self.ffn_down = nn.Linear(self.ffn_dim, self.dim, bias=False)

        generator = torch.Generator().manual_seed(int(seed))
        for layer in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.ffn_up,
        ):
            nn.init.normal_(layer.weight, mean=0.0, std=0.02, generator=generator)
        nn.init.zeros_(self.ffn_down.weight)
        nn.init.ones_(self.ffn_norm.weight)
        nn.init.zeros_(self.ffn_norm.bias)

    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def _heads(self, value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*value.shape[:-1], self.heads, self.head_dim)

    def _ffn(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.ffn_down(F.gelu(self.ffn_up(self.ffn_norm(value))))

    def components(
        self,
        query: torch.Tensor,
        history: torch.Tensor,
        candidates: torch.Tensor,
    ) -> dict[str, Any]:
        query = F.normalize(query, dim=-1, eps=1e-6)
        history = F.normalize(history, dim=-1, eps=1e-6)
        candidates = F.normalize(candidates, dim=-1, eps=1e-6)

        query_head = self._heads(self.q_proj(query))
        history_key = self._heads(self.k_proj(history))
        history_value = self._heads(self.v_proj(history))
        candidate_key = self._heads(self.k_proj(candidates))
        attention_logits = torch.einsum(
            "rd,jrd->jr", query_head, history_key
        ) / (math.sqrt(self.head_dim) * self.temperature)
        attention = torch.softmax(attention_logits, dim=0)
        global_head = torch.einsum("jr,jrd->rd", attention, history_value)
        global_write = self.out_proj(global_head.reshape(self.inner_dim))

        relative_candidate_key = candidate_key - candidate_key.mean(dim=0, keepdim=True)
        support = F.relu(
            torch.einsum(
                "crd,jrd->cjr", relative_candidate_key, history_key
            )
            / math.sqrt(self.head_dim)
        )
        unnormalized_edge_weight = support * attention.unsqueeze(0)
        edge_weight = unnormalized_edge_weight / (
            1.0 + unnormalized_edge_weight.sum(dim=1, keepdim=True)
        )
        innovation = history_value - global_head.unsqueeze(0)

        # For nn.Linear, out = x @ weight.T.  Hence the head-space gradient of
        # <candidate, out_proj(x)> is candidate @ weight, split by head.
        score_normal = self._heads(candidates @ self.out_proj.weight)
        pair_innovation = innovation.unsqueeze(0).expand(
            len(candidates), -1, -1, -1
        )
        pair_normal = score_normal.unsqueeze(1).expand_as(pair_innovation)
        raw_readout = (pair_innovation * pair_normal).sum(dim=-1)

        if self.mode == EVENTWISE_HALFSPACE:
            pair_value = project_to_score_halfspace(pair_innovation, pair_normal)
            local_head = torch.einsum("cjr,cjrd->crd", edge_weight, pair_value)
        elif self.mode == EVENTWISE_RAW:
            pair_value = pair_innovation
            local_head = torch.einsum("cjr,cjrd->crd", edge_weight, pair_value)
        elif self.mode == RAY_ONLY:
            pair_value = score_ray(pair_innovation, pair_normal)
            local_head = torch.einsum("cjr,cjrd->crd", edge_weight, pair_value)
        elif self.mode == POSTPOOL_HALFSPACE:
            raw_pool = torch.einsum("cjr,cjrd->crd", edge_weight, pair_innovation)
            local_head = project_to_score_halfspace(raw_pool, score_normal)
            pair_value = pair_innovation
        else:
            pair_value = pair_innovation.new_zeros(pair_innovation.shape)
            local_head = score_normal.new_zeros(score_normal.shape)

        edge_value = edge_weight.unsqueeze(-1) * pair_value
        local_write = self.out_proj(local_head.reshape(len(candidates), self.inner_dim))
        factual_state = (
            query.unsqueeze(0)
            + self.global_scale * global_write.unsqueeze(0)
            + self.candidate_scale * local_write
        )
        null_state = query.unsqueeze(0).expand_as(factual_state)
        factual_readout = (self._ffn(factual_state) * candidates).sum(dim=-1)
        null_readout = (self._ffn(null_state) * candidates).sum(dim=-1)
        correction = factual_readout - null_readout

        projected_readout = (pair_value * pair_normal).sum(dim=-1)
        return {
            "attention": attention,
            "support": support,
            "edge_weight": edge_weight,
            "global_head": global_head,
            "global_write": global_write,
            "innovation": innovation,
            "score_normal": score_normal,
            "raw_readout": raw_readout,
            "pair_value": pair_value,
            "edge_value": edge_value,
            "projected_readout": projected_readout,
            "local_head": local_head,
            "local_write": local_write,
            "factual_state": factual_state,
            "null_state": null_state,
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
            raise ValueError("C39 embedding shape differs")
        if not query_present or repeat_present or len(history) == 0:
            return candidates.new_zeros(len(candidates))
        return self.components(query, history, candidates)["correction"]
