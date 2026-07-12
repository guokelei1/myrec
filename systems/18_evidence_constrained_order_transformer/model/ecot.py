"""Evidence-conditioned order projection inside a Transformer ranker."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    denominator = weights.sum(dim=dim, keepdim=True).clamp_min(1.0)
    return (values * weights).sum(dim=dim, keepdim=True) / denominator


def project_two_group_order(
    proposal: torch.Tensor,
    anchor: torch.Tensor,
    repeat_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    *,
    bisection_steps: int = 48,
) -> torch.Tensor:
    """Project proposal onto repeat-delta >= nonrepeat-delta inequalities.

    For ``delta = score - anchor`` the feasible set is

    ``delta_i >= delta_k`` for every repeated ``i`` and non-repeated ``k``.

    The Euclidean projection clamps low repeated deltas upward and high
    non-repeated deltas downward at a shared threshold.  The threshold is the
    unique root (or any point in the zero interval) at which total upward and
    downward corrections match, preserving the candidate mean exactly up to
    floating-point bisection error.
    """

    if proposal.shape != anchor.shape or proposal.ndim != 2:
        raise ValueError("proposal and anchor must have identical [batch,candidate] shape")
    if repeat_mask.shape != proposal.shape:
        raise ValueError("repeat_mask shape mismatch")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(repeat_mask, dtype=torch.bool)
    if candidate_mask.shape != proposal.shape:
        raise ValueError("candidate_mask shape mismatch")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")

    valid = candidate_mask.bool()
    repeated = repeat_mask.bool() & valid
    nonrepeated = (~repeat_mask.bool()) & valid
    constrained = repeated.any(dim=1) & nonrepeated.any(dim=1)
    delta = proposal - anchor

    positive_inf = torch.full_like(delta, torch.inf)
    negative_inf = torch.full_like(delta, -torch.inf)
    low = torch.where(valid, delta, positive_inf).amin(dim=1)
    high = torch.where(valid, delta, negative_inf).amax(dim=1)

    # Bisection remains entirely tensorized.  For already feasible rows the
    # root lies in a flat interval and the final clamp is the identity.
    for _ in range(bisection_steps):
        threshold = (low + high) * 0.5
        upward = torch.where(
            repeated,
            (threshold.unsqueeze(1) - delta).clamp_min(0.0),
            torch.zeros_like(delta),
        ).sum(dim=1)
        downward = torch.where(
            nonrepeated,
            (delta - threshold.unsqueeze(1)).clamp_min(0.0),
            torch.zeros_like(delta),
        ).sum(dim=1)
        function = upward - downward
        high = torch.where(function > 0, threshold, high)
        low = torch.where(function <= 0, threshold, low)

    threshold = ((low + high) * 0.5).unsqueeze(1)
    projected_delta = torch.where(repeated, torch.maximum(delta, threshold), delta)
    projected_delta = torch.where(
        nonrepeated, torch.minimum(projected_delta, threshold), projected_delta
    )
    projected_delta = torch.where(valid, projected_delta, delta)
    output = anchor + projected_delta
    return torch.where(constrained.unsqueeze(1), output, proposal)


def protected_margin_violation(
    scores: torch.Tensor,
    anchor: torch.Tensor,
    repeat_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return per-request maximum protected-margin violation."""

    if candidate_mask is None:
        candidate_mask = torch.ones_like(repeat_mask, dtype=torch.bool)
    repeated = repeat_mask.bool() & candidate_mask.bool()
    nonrepeated = (~repeat_mask.bool()) & candidate_mask.bool()
    delta = scores - anchor
    min_repeat = torch.where(
        repeated, delta, torch.full_like(delta, torch.inf)
    ).amin(dim=1)
    max_nonrepeat = torch.where(
        nonrepeated, delta, torch.full_like(delta, -torch.inf)
    ).amax(dim=1)
    constrained = repeated.any(dim=1) & nonrepeated.any(dim=1)
    violation = (max_nonrepeat - min_repeat).clamp_min(0.0)
    return torch.where(constrained, violation, torch.zeros_like(violation))


def soft_constraint_penalty(
    scores: torch.Tensor,
    anchor: torch.Tensor,
    repeat_mask: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    violation = protected_margin_violation(scores, anchor, repeat_mask, candidate_mask)
    constrained = repeat_mask.any(dim=1)
    if not bool(constrained.any()):
        return scores.sum() * 0.0
    return violation[constrained].square().mean()


@dataclass(frozen=True)
class ECOTOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    anchor_scores: torch.Tensor
    proposal_scores: torch.Tensor
    raw_transfer: torch.Tensor


class ECOTRanker(nn.Module):
    """Small LLM4Rec-style Transformer with an internal order projection."""

    VALID_MODES = frozenset({"projection", "direct", "soft_penalty"})

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        nhead: int,
        layers: int,
        ffn_dim: int,
        history_slots: int,
        dropout: float,
        proposal_radius: float,
        repeat_bonus: float,
        projection_bisection_steps: int,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"unknown mode: {mode}")
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.mode = mode
        self.proposal_radius = float(proposal_radius)
        self.repeat_bonus = float(repeat_bonus)
        self.bisection_steps = int(projection_bisection_steps)
        if self.bisection_steps < 32:
            raise ValueError("projection_bisection_steps must be at least 32")
        self.history_slots = int(history_slots)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.query_type = nn.Parameter(torch.empty(d_model))
        self.history_type = nn.Parameter(torch.empty(d_model))
        self.candidate_type = nn.Parameter(torch.empty(d_model))
        self.history_position = nn.Parameter(torch.empty(history_slots, d_model))

        base_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.base_transformer = nn.TransformerEncoder(
            base_layer, num_layers=max(1, layers // 2), norm=nn.LayerNorm(d_model)
        )
        history_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(
            history_layer, num_layers=layers, norm=nn.LayerNorm(d_model)
        )
        self.base_head = nn.Linear(d_model, 1)
        self.transfer_head = nn.Sequential(
            nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1)
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.query_type, std=0.02)
        nn.init.normal_(self.history_type, std=0.02)
        nn.init.normal_(self.candidate_type, std=0.02)
        nn.init.normal_(self.history_position, std=0.02)
        for module in (self.base_head, self.transfer_head[-1]):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def clone_with_mode(self, mode: str) -> "ECOTRanker":
        if mode not in self.VALID_MODES:
            raise ValueError(mode)
        cloned = copy.deepcopy(self)
        cloned.mode = mode
        return cloned

    def query_only(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_mask is None:
            candidate_mask = torch.ones(
                candidates.shape[:2], dtype=torch.bool, device=candidates.device
            )
        query_token = self.input_projection(query).unsqueeze(1) + self.query_type
        candidate_tokens = self.input_projection(candidates) + self.candidate_type
        tokens = torch.cat((query_token, candidate_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(query.shape[0], 1, dtype=torch.bool, device=query.device),
                ~candidate_mask.bool(),
            ),
            dim=1,
        )
        encoded = self.base_transformer(tokens, src_key_padding_mask=padding)
        candidate_states = encoded[:, 1:]
        base_scores = self.base_head(candidate_states).squeeze(-1)
        base_scores = base_scores.masked_fill(~candidate_mask.bool(), -1e4)
        return base_scores, candidate_states

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        repeat_mask: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> ECOTOutput:
        active_mode = self.mode if mode is None else mode
        if active_mode not in self.VALID_MODES:
            raise ValueError(active_mode)
        batch, candidate_count, _ = candidates.shape
        if history.shape[1] != self.history_slots:
            raise ValueError("history slot count mismatch")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch, candidate_count, dtype=torch.bool, device=candidates.device
            )
        base_scores, candidate_states = self.query_only(query, candidates, candidate_mask)
        history_present = history_mask.bool().any(dim=1)

        query_token = self.input_projection(query).unsqueeze(1) + self.query_type
        history_tokens = (
            self.input_projection(history)
            + self.history_type
            + self.history_position.unsqueeze(0)
        )
        candidate_tokens = candidate_states + self.candidate_type
        tokens = torch.cat((query_token, history_tokens, candidate_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=query.device),
                ~history_mask.bool(),
                ~candidate_mask.bool(),
            ),
            dim=1,
        )
        encoded = self.history_transformer(tokens, src_key_padding_mask=padding)
        history_candidate_states = encoded[:, 1 + self.history_slots :]
        raw_transfer = self.transfer_head(history_candidate_states).squeeze(-1)
        bounded = self.proposal_radius * torch.tanh(raw_transfer)
        bounded = bounded - _masked_mean(bounded, candidate_mask.bool(), dim=1)
        bounded = bounded.masked_fill(~candidate_mask.bool(), 0.0)

        anchor = base_scores + self.repeat_bonus * repeat_mask.to(base_scores.dtype)
        proposal = anchor + bounded
        if active_mode == "projection":
            personalized = project_two_group_order(
                proposal,
                anchor,
                repeat_mask,
                candidate_mask,
                bisection_steps=self.bisection_steps,
            )
        else:
            personalized = proposal
        scores = torch.where(history_present.unsqueeze(1), personalized, base_scores)
        anchor = torch.where(history_present.unsqueeze(1), anchor, base_scores)
        proposal = torch.where(history_present.unsqueeze(1), proposal, base_scores)
        raw_transfer = torch.where(
            history_present.unsqueeze(1), raw_transfer, torch.zeros_like(raw_transfer)
        )
        return ECOTOutput(scores, base_scores, anchor, proposal, raw_transfer)
