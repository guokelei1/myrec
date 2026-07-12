"""Conservative one-sided transport on the strong-base order graph."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


TRANSPORT_MODES = ("one_sided", "signed", "hard")


@dataclass(frozen=True)
class EdgeTransportOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    transport: torch.Tensor
    base_gap: torch.Tensor


class BaseOrderEdgeTransportTransformer(nn.Module):
    """Use history only to open capacity-bounded adjacent base-order edges."""

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def forward(
        self,
        *,
        base_scores: torch.Tensor,
        evidence: torch.Tensor,
        candidate_mask: torch.Tensor,
        canonical_order: torch.Tensor,
        mode: str = "one_sided",
    ) -> EdgeTransportOutput:
        if mode not in TRANSPORT_MODES:
            raise ValueError(f"unknown C60 transport mode: {mode}")
        if base_scores.shape != evidence.shape or base_scores.shape != candidate_mask.shape:
            raise ValueError("C60 score/evidence/mask shape differs")
        if canonical_order.shape != base_scores.shape:
            raise ValueError("C60 canonical-order shape differs")
        batch, candidates = base_scores.shape
        correction = torch.zeros_like(base_scores, dtype=torch.float64)
        transport = torch.zeros(
            (batch, max(0, candidates - 1)), dtype=torch.float64, device=base_scores.device
        )
        base_gap = torch.zeros_like(transport)
        for row in range(batch):
            count = int(candidate_mask[row].sum())
            if count <= 1:
                continue
            order = canonical_order[row, :count].long()
            if not torch.equal(torch.sort(order).values, torch.where(candidate_mask[row])[0]):
                raise ValueError("C60 canonical order is not a candidate permutation")
            score = base_scores[row, order].double()
            value = evidence[row, order].double()
            gap = score[:-1] - score[1:]
            if bool((gap < -1e-12).any()):
                raise ValueError("C60 canonical base order is not descending")
            gap = gap.clamp_min(0.0)
            direction = value[1:] - value[:-1]
            baseline = torch.sigmoid(-gap)
            conditioned = torch.sigmoid(-gap + direction)
            improvement = conditioned - baseline
            if mode == "one_sided":
                rate = (improvement / (1.0 - baseline).clamp_min(1e-12)).clamp(0.0, 1.0)
            elif mode == "signed":
                denominator = torch.where(
                    improvement >= 0,
                    1.0 - baseline,
                    baseline,
                ).clamp_min(1e-12)
                rate = (improvement / denominator).clamp(-1.0, 1.0)
            else:
                rate = (direction > gap).to(torch.float64)
            flow = rate * gap
            sorted_correction = torch.zeros_like(score)
            sorted_correction[:-1] -= flow
            sorted_correction[1:] += flow
            correction[row, order] = sorted_correction
            transport[row, : count - 1] = flow
            base_gap[row, : count - 1] = gap
        scores = base_scores.double() + correction
        scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        correction = correction.masked_fill(~candidate_mask.bool(), 0.0)
        return EdgeTransportOutput(
            scores=scores,
            correction=correction,
            transport=transport,
            base_gap=base_gap,
        )
