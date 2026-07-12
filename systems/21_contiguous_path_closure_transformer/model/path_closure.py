"""Candidate-centred path-closure operator for the C21 signal gate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "contiguous_path",
    "one_step",
    "unordered_pair",
    "endpoint_only",
    "pooled_history",
)


@dataclass(frozen=True)
class PathClosureOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    deltas: torch.Tensor
    evidence: torch.Tensor
    has_evidence: torch.Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class PathClosureProbe(nn.Module):
    """Measure directed contiguous-path evidence on frozen input states.

    This deliberately is not the proposed Transformer.  All modes share the
    same parameters; only their history evidence algebra differs.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        projection_dim: int,
        max_history: int,
        max_horizon: int,
        evidence_temperature: float,
        score_delta_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C21 mode: {mode}")
        if input_dim <= 0 or projection_dim <= 1:
            raise ValueError("invalid C21 projection shape")
        if max_history < 2 or not 1 <= max_horizon < max_history:
            raise ValueError("invalid C21 path horizon")
        if evidence_temperature <= 0.0 or score_delta_max <= 0.0:
            raise ValueError("invalid C21 evidence scale")
        self.input_dim = int(input_dim)
        self.projection_dim = int(projection_dim)
        self.max_history = int(max_history)
        self.max_horizon = int(max_horizon)
        self.evidence_temperature = float(evidence_temperature)
        self.score_delta_max = float(score_delta_max)
        self.mode = mode

        self.state_projection = nn.Linear(input_dim, projection_dim, bias=False)
        self.relation_projection = nn.Linear(projection_dim, projection_dim, bias=False)
        self.anchor_projection = nn.Linear(projection_dim, projection_dim, bias=False)
        self.direction_log_gain = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.anchor_log_gain = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.evidence_log_gain = nn.Parameter(torch.tensor(_inverse_softplus(1.0)))
        self.residual_logit = nn.Parameter(torch.tensor(-2.0))

        starts: list[int] = []
        ends: list[int] = []
        lengths: list[int] = []
        for start in range(max_history):
            for length in range(1, max_horizon + 1):
                end = start + length
                if end < max_history:
                    starts.append(start)
                    ends.append(end)
                    lengths.append(length)
        self.register_buffer("segment_starts", torch.tensor(starts, dtype=torch.long), persistent=False)
        self.register_buffer("segment_ends", torch.tensor(ends, dtype=torch.long), persistent=False)
        self.register_buffer("segment_lengths", torch.tensor(lengths, dtype=torch.long), persistent=False)

    def _unit(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value, p=2.0, dim=-1, eps=1e-8)

    @staticmethod
    def _masked_logmeanexp(
        logits: torch.Tensor,
        mask: torch.Tensor,
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.ndim != 3 or mask.ndim != 2:
            raise ValueError("C21 pooling expects logits [B,C,K] and mask [B,K]")
        expanded = mask[:, None, :]
        valid = mask.any(dim=-1)
        safe = (logits / temperature).masked_fill(~expanded, -torch.inf)
        pooled = temperature * (
            torch.logsumexp(safe, dim=-1)
            - mask.sum(dim=-1).clamp_min(1).to(logits.dtype).log()[:, None]
        )
        pooled = torch.where(valid[:, None], pooled, torch.zeros_like(pooled))
        return pooled, valid

    def _path_evidence(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        history_slots = history.shape[1]
        available = self.segment_ends < history_slots
        starts_index = self.segment_starts[available]
        ends_index = self.segment_ends[available]
        lengths = self.segment_lengths[available]
        if not len(starts_index):
            return candidates.new_zeros(candidates.shape[:2]), history_mask.new_zeros(history_mask.shape[0])
        starts = history[:, starts_index]
        ends = history[:, ends_index]
        segment_mask = history_mask[:, starts_index] & history_mask[:, ends_index]
        if self.mode == "one_step":
            segment_mask = segment_mask & lengths.eq(1)[None, :]

        relation = self._unit(
            self.relation_projection(candidates - query[:, None, :])
        )
        path = self._unit(self.relation_projection(ends - starts))
        direction = torch.einsum("bcd,bsd->bcs", relation, path)

        anchor_query = self._unit(self.anchor_projection(query))
        anchor_candidate = self._unit(self.anchor_projection(candidates))
        anchor_start = self._unit(self.anchor_projection(starts))
        anchor_end = self._unit(self.anchor_projection(ends))
        query_start = torch.einsum("bd,bsd->bs", anchor_query, anchor_start)
        query_end = torch.einsum("bd,bsd->bs", anchor_query, anchor_end)
        candidate_end = torch.einsum("bcd,bsd->bcs", anchor_candidate, anchor_end)
        candidate_start = torch.einsum("bcd,bsd->bcs", anchor_candidate, anchor_start)
        oriented_anchor = 0.5 * (query_start[:, None, :] + candidate_end)

        direction_gain = F.softplus(self.direction_log_gain)
        anchor_gain = F.softplus(self.anchor_log_gain)
        if self.mode == "unordered_pair":
            reverse_anchor = 0.5 * (query_end[:, None, :] + candidate_start)
            logits = direction_gain * direction.abs() + anchor_gain * torch.maximum(
                oriented_anchor, reverse_anchor
            )
        elif self.mode == "endpoint_only":
            logits = anchor_gain * oriented_anchor
        else:
            logits = direction_gain * direction + anchor_gain * oriented_anchor
        return self._masked_logmeanexp(logits, segment_mask, self.evidence_temperature)

    def _pooled_evidence(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        relation = self._unit(
            self.relation_projection(candidates - query[:, None, :])
        )
        history_relation = self._unit(
            self.relation_projection(history - query[:, None, :])
        )
        directional = torch.einsum("bcd,bhd->bch", relation, history_relation)
        anchor_query = self._unit(self.anchor_projection(query))
        anchor_candidate = self._unit(self.anchor_projection(candidates))
        anchor_history = self._unit(self.anchor_projection(history))
        query_history = torch.einsum("bd,bhd->bh", anchor_query, anchor_history)
        candidate_history = torch.einsum("bcd,bhd->bch", anchor_candidate, anchor_history)
        logits = F.softplus(self.direction_log_gain) * directional + F.softplus(
            self.anchor_log_gain
        ) * 0.5 * (query_history[:, None, :] + candidate_history)
        return self._masked_logmeanexp(logits, history_mask, self.evidence_temperature)

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        base_scores: torch.Tensor,
        query_present: torch.Tensor | None = None,
    ) -> PathClosureOutput:
        if query.ndim != 2 or candidates.ndim != 3 or history.ndim != 3:
            raise ValueError("C21 states must have shapes [B,D], [B,C,D], [B,H,D]")
        if candidates.shape[:2] != candidate_mask.shape or history.shape[:2] != history_mask.shape:
            raise ValueError("C21 state/mask shape mismatch")
        if base_scores.shape != candidate_mask.shape or query.shape[0] != candidates.shape[0]:
            raise ValueError("C21 score/batch shape mismatch")
        if query.shape[-1] != self.input_dim or candidates.shape[-1] != self.input_dim or history.shape[-1] != self.input_dim:
            raise ValueError("C21 input dimension changed")
        candidate_mask = candidate_mask.bool()
        history_mask = history_mask.bool()
        if query_present is None:
            query_present = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)
        if query_present.shape != (query.shape[0],):
            raise ValueError("C21 query-present mask must have shape [B]")

        projected_query = self.state_projection(query)
        projected_candidates = self.state_projection(candidates)
        projected_history = self.state_projection(history)
        if self.mode == "pooled_history":
            evidence, has_evidence = self._pooled_evidence(
                projected_query, projected_candidates, projected_history, history_mask
            )
        else:
            evidence, has_evidence = self._path_evidence(
                projected_query, projected_candidates, projected_history, history_mask
            )
        evidence = evidence.masked_fill(~candidate_mask, 0.0)
        count = candidate_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(evidence.dtype)
        evidence_mean = evidence.sum(dim=-1, keepdim=True) / count
        centred_evidence = (evidence - evidence_mean).masked_fill(~candidate_mask, 0.0)
        raw_delta = torch.tanh(F.softplus(self.evidence_log_gain) * centred_evidence)
        raw_mean = (raw_delta * candidate_mask.to(raw_delta.dtype)).sum(dim=-1, keepdim=True) / count
        centred_delta = (raw_delta - raw_mean).masked_fill(~candidate_mask, 0.0)
        write_mask = (has_evidence & query_present.bool())[:, None].to(centred_delta.dtype)
        deltas = (
            0.5
            * self.score_delta_max
            * torch.sigmoid(self.residual_logit)
            * centred_delta
            * write_mask
        )
        scores = base_scores + deltas
        return PathClosureOutput(
            scores=scores,
            base_scores=base_scores,
            deltas=deltas,
            evidence=evidence,
            has_evidence=has_evidence & query_present.bool(),
        )
