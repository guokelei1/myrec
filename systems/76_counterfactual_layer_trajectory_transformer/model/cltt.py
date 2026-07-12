"""Tiny executable C76 operator used by the data-free design gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn


QUERY = 0
CANDIDATE = 1
HISTORY = 2

MODES = (
    "counterfactual_trajectory",
    "final_logit_delta",
    "final_hidden_delta",
    "factual_trajectory",
    "ordinary_full",
)


class TinyTransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int) -> None:
        super().__init__()
        self.heads = int(heads)
        self.norm_attention = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(
            hidden, heads, dropout=0.0, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, states: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        normalized = self.norm_attention(states)
        blocked = ~allowed.bool()
        blocked = blocked.repeat_interleave(self.heads, dim=0)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=blocked,
            need_weights=False,
        )
        states = states + update
        return states + self.ffn(self.norm_ffn(states))


class TinyBackbone(nn.Module):
    def __init__(
        self,
        vocabulary: int,
        hidden: int,
        heads: int,
        layers: int,
        maximum_length: int,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary, hidden)
        self.position_embedding = nn.Embedding(maximum_length, hidden)
        self.segment_embedding = nn.Embedding(3, hidden)
        self.layers = nn.ModuleList(
            [TinyTransformerBlock(hidden, heads) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        allowed: torch.Tensor,
    ) -> list[torch.Tensor]:
        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        states = (
            self.token_embedding(token_ids.long())
            + self.position_embedding(positions).unsqueeze(0)
            + self.segment_embedding(segment_ids.long())
        )
        hidden_states = []
        for layer in self.layers:
            states = layer(states, allowed)
            hidden_states.append(self.final_norm(states))
        return hidden_states


class TrajectoryEncoder(nn.Module):
    def __init__(self, hidden: int, heads: int, layers: int, backbone_layers: int) -> None:
        super().__init__()
        self.read_token = nn.Parameter(torch.zeros(1, 1, hidden))
        nn.init.normal_(self.read_token, std=0.02)
        self.layer_embedding = nn.Embedding(backbone_layers, hidden)
        self.role_embedding = nn.Embedding(3, hidden)
        block = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=heads,
            dim_feedforward=hidden * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, num_layers=layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, trajectory: torch.Tensor) -> torch.Tensor:
        # trajectory: [batch*candidates, backbone_layers, 3, hidden]
        batch, layers, roles, hidden = trajectory.shape
        layer_ids = torch.arange(layers, device=trajectory.device).view(1, layers, 1)
        role_ids = torch.arange(roles, device=trajectory.device).view(1, 1, roles)
        states = (
            trajectory
            + self.layer_embedding(layer_ids)
            + self.role_embedding(role_ids)
        ).reshape(batch, layers * roles, hidden)
        read = self.read_token.expand(batch, -1, -1)
        output = self.encoder(torch.cat((read, states), dim=1))
        return self.norm(output[:, 0])


@dataclass
class ForwardDiagnostics:
    scores: torch.Tensor
    trajectory_rms: torch.Tensor
    early_trajectory_rms: torch.Tensor
    factual_cut_max_abs: torch.Tensor


def full_and_cut_masks(
    segment_ids: torch.Tensor, history_present: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return full and bidirectional history-cut masks."""

    batch, length = segment_ids.shape
    full = torch.ones((batch, length, length), dtype=torch.bool, device=segment_ids.device)
    query_segment = segment_ids.unsqueeze(2)
    key_segment = segment_ids.unsqueeze(1)
    q_history = query_segment.eq(HISTORY)
    k_history = key_segment.eq(HISTORY)
    cross_history = q_history ^ k_history
    cut = full & ~cross_history
    present = history_present.bool().view(batch, 1, 1)
    return torch.where(present, full, cut), cut


def carrier_scaled_difference(
    factual: torch.Tensor, cut: torch.Tensor, epsilon: float = 1e-6
) -> torch.Tensor:
    scale = torch.sqrt(
        0.5 * (factual.float().square() + cut.float().square()).mean(-1, keepdim=True)
        + epsilon
    )
    return (factual.float() - cut.float()) / scale


def carrier_scaled_state(states: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    scale = torch.sqrt(states.float().square().mean(-1, keepdim=True) + epsilon)
    return states.float() / scale


def segment_pool(states: torch.Tensor, segments: torch.Tensor) -> torch.Tensor:
    values = []
    for role in (QUERY, CANDIDATE, HISTORY):
        mask = segments.eq(role).unsqueeze(-1)
        values.append(
            (states * mask).sum(1) / mask.sum(1).clamp_min(1)
        )
    return torch.stack(values, dim=1)


class CounterfactualLayerTrajectoryTransformer(nn.Module):
    """C76 ranker and capacity-matched degenerations."""

    def __init__(
        self,
        *,
        mode: str,
        vocabulary_size: int,
        hidden_size: int,
        attention_heads: int,
        backbone_layers: int,
        trajectory_layers: int,
        trajectory_heads: int,
        maximum_length: int,
        correction_bound: float,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C76 mode: {mode}")
        self.mode = mode
        self.backbone_layers = int(backbone_layers)
        self.correction_bound = float(correction_bound)
        self.backbone = TinyBackbone(
            vocabulary_size,
            hidden_size,
            attention_heads,
            backbone_layers,
            maximum_length,
        )
        self.trajectory = TrajectoryEncoder(
            hidden_size,
            trajectory_heads,
            trajectory_layers,
            backbone_layers,
        )
        self.final_state_anchor = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.eye_(self.final_state_anchor.weight)
        self.logit_probe = nn.Linear(hidden_size, 1, bias=False)
        self.logit_lift = nn.Linear(1, hidden_size, bias=False)
        self.score_head = nn.Linear(hidden_size, 1, bias=False)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @staticmethod
    def _trajectory_tokens(
        hidden: Iterable[torch.Tensor], segments: torch.Tensor
    ) -> torch.Tensor:
        return torch.stack([segment_pool(value, segments) for value in hidden], dim=1)

    def forward(
        self,
        token_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        base_scores: torch.Tensor,
        history_present: torch.Tensor,
        repeat_present: torch.Tensor,
        repeat_scores: torch.Tensor,
    ) -> ForwardDiagnostics:
        batch, candidates, length = token_ids.shape
        flat_tokens = token_ids.reshape(batch * candidates, length)
        flat_segments = segment_ids.reshape(batch * candidates, length)
        flat_history = history_present[:, None].expand(-1, candidates).reshape(-1)
        full_mask, cut_mask = full_and_cut_masks(flat_segments, flat_history)
        factual = self.backbone(flat_tokens, flat_segments, full_mask)
        cut = self.backbone(flat_tokens, flat_segments, cut_mask)
        differences = [carrier_scaled_difference(left, right) for left, right in zip(factual, cut)]
        factual_states = [carrier_scaled_state(value) for value in factual]

        if self.mode == "counterfactual_trajectory":
            trajectory = self._trajectory_tokens(differences, flat_segments)
            final_candidate = segment_pool(differences[-1], flat_segments)[:, CANDIDATE]
            trajectory[:, -1, CANDIDATE] = trajectory[:, -1, CANDIDATE] + self.final_state_anchor(final_candidate)
        elif self.mode == "final_hidden_delta":
            final = segment_pool(differences[-1], flat_segments)
            trajectory = final[:, None].expand(-1, self.backbone_layers, -1, -1).clone()
        elif self.mode == "final_logit_delta":
            scalar = self.logit_probe(factual[-1][:, 0]) - self.logit_probe(cut[-1][:, 0])
            lifted = self.logit_lift(scalar)
            trajectory = lifted[:, None, None].expand(-1, self.backbone_layers, 3, -1).clone()
        elif self.mode == "factual_trajectory":
            trajectory = self._trajectory_tokens(factual_states, flat_segments)
        else:
            final = segment_pool(factual_states[-1], flat_segments)
            trajectory = final[:, None].expand(-1, self.backbone_layers, -1, -1).clone()

        encoded = self.trajectory(trajectory)
        correction = self.correction_bound * torch.tanh(self.score_head(encoded)).reshape(batch, candidates)
        correction = correction - correction.mean(-1, keepdim=True)
        personalized = base_scores.detach() + correction
        scores = torch.where(
            history_present.bool().unsqueeze(1), personalized, base_scores.detach()
        )
        scores = torch.where(
            repeat_present.bool().unsqueeze(1), repeat_scores.detach(), scores
        )
        stacked = torch.stack(differences, dim=1)
        return ForwardDiagnostics(
            scores=scores,
            trajectory_rms=stacked.square().mean().sqrt(),
            early_trajectory_rms=stacked[:, :-1].square().mean().sqrt(),
            factual_cut_max_abs=stacked.abs().max(),
        )
