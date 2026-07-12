"""Fixed-direction semantic candidate-budget Transformer for C59.

The contextual token states are produced by the registered frozen BGE
Transformer.  This module changes the ranking interaction: every history
event allocates a finite probability budget over the exchangeable candidate
set and a NULL sink.  No learned value or output projection remains.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "candidate_budget",
    "slot_budget_no_null",
    "history_softmax",
    "pooled_history",
    "raw_query",
)


@dataclass(frozen=True)
class SemanticBudgetOutput:
    scores: Mapping[str, torch.Tensor]
    corrections: Mapping[str, torch.Tensor]
    features: Mapping[str, torch.Tensor]
    triadic_logits: torch.Tensor
    primary_allocation: torch.Tensor
    null_allocation: torch.Tensor
    active_request: torch.Tensor


class SemanticCandidateBudgetTransformer(nn.Module):
    """Compose frozen Transformer tokens through a fixed candidate budget."""

    def __init__(self, *, null_logit: float = 0.0, epsilon: float = 1e-6) -> None:
        super().__init__()
        if epsilon <= 0:
            raise ValueError("C59 epsilon must be positive")
        self.null_logit = float(null_logit)
        self.epsilon = float(epsilon)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @staticmethod
    def _validate(
        *,
        query_tokens: torch.Tensor,
        query_token_mask: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_tokens: torch.Tensor,
        history_token_mask: torch.Tensor,
        event_weights: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
    ) -> None:
        batch, candidates = candidate_mask.shape
        if query_tokens.shape[:-1] != query_token_mask.shape:
            raise ValueError("C59 query token shape differs")
        if candidate_tokens.shape[:-1] != candidate_token_mask.shape:
            raise ValueError("C59 candidate token shape differs")
        if history_tokens.shape[:-1] != history_token_mask.shape:
            raise ValueError("C59 history token shape differs")
        if candidate_tokens.shape[:2] != (batch, candidates):
            raise ValueError("C59 candidate batch shape differs")
        if history_tokens.shape[:2] != event_weights.shape:
            raise ValueError("C59 event weight shape differs")
        if query_tokens.shape[0] != batch or history_tokens.shape[0] != batch:
            raise ValueError("C59 request batch shape differs")
        if query_tokens.shape[-1] != candidate_tokens.shape[-1] or query_tokens.shape[-1] != history_tokens.shape[-1]:
            raise ValueError("C59 contextual dimensions differ")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != candidate_mask.shape:
            raise ValueError("C59 anchor score shape differs")
        if repeat_request.shape != (batch,):
            raise ValueError("C59 repeat mask shape differs")

    def _zscore(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Canonical float64 population standardization over a candidate set."""
        output = torch.zeros_like(values, dtype=torch.float32)
        for row in range(len(values)):
            positions = torch.where(mask[row].bool())[0]
            if not len(positions):
                continue
            selected = values[row, positions].double()
            ordered = torch.sort(selected).values
            mean = ordered.sum(dtype=torch.float64) / len(ordered)
            centered = ordered - mean
            variance = (centered * centered).sum(dtype=torch.float64) / len(ordered)
            scale = torch.sqrt(variance)
            if bool(torch.isfinite(scale)) and float(scale) > self.epsilon:
                output[row, positions] = ((selected - mean) / scale).float()
        return output

    @staticmethod
    def _symmetric_softmax(logits: torch.Tensor) -> torch.Tensor:
        """Softmax with a canonical candidate-set denominator."""
        value = logits.double()
        ordered = torch.sort(value, dim=-1).values
        maximum = ordered[..., -1:]
        denominator = torch.exp(ordered - maximum).sum(dim=-1, keepdim=True, dtype=torch.float64)
        return (torch.exp(value - maximum) / denominator).float()

    @staticmethod
    def _raw_query_similarity(
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_token_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Mean over query tokens of their maximum candidate-token cosine.
        similarity = torch.einsum("qd,ctd->qct", query, candidates)
        similarity = similarity.masked_fill(~candidate_token_mask[None].bool(), -torch.inf)
        return similarity.max(dim=-1).values.mean(dim=0)

    @staticmethod
    def _candidate_event_similarity(
        candidates: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        # Mean over candidate tokens of their maximum event-token cosine.
        similarity = torch.einsum("ctd,md->ctm", candidates, event).max(dim=-1).values
        weight = candidate_token_mask.to(similarity.dtype)
        return (similarity * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1.0)

    def forward(
        self,
        *,
        query_tokens: torch.Tensor,
        query_token_mask: torch.Tensor,
        candidate_tokens: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_tokens: torch.Tensor,
        history_token_mask: torch.Tensor,
        event_weights: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
    ) -> SemanticBudgetOutput:
        self._validate(
            query_tokens=query_tokens,
            query_token_mask=query_token_mask,
            candidate_tokens=candidate_tokens,
            candidate_token_mask=candidate_token_mask,
            history_tokens=history_tokens,
            history_token_mask=history_token_mask,
            event_weights=event_weights,
            candidate_mask=candidate_mask,
            base_scores=base_scores,
            item_only_scores=item_only_scores,
            repeat_request=repeat_request,
        )
        query_tokens = F.normalize(query_tokens.float(), dim=-1)
        candidate_tokens = F.normalize(candidate_tokens.float(), dim=-1)
        history_tokens = F.normalize(history_tokens.float(), dim=-1)
        candidate_mask = candidate_mask.bool()
        query_token_mask = query_token_mask.bool()
        candidate_token_mask = candidate_token_mask.bool()
        history_token_mask = history_token_mask.bool()
        repeat_request = repeat_request.bool()

        batch, candidates = candidate_mask.shape
        events = history_tokens.shape[1]
        event_mask = history_token_mask.any(dim=-1)
        active = (
            query_token_mask.any(dim=-1)
            & event_mask.any(dim=-1)
            & candidate_mask.any(dim=-1)
            & ~repeat_request
        )
        triadic = query_tokens.new_zeros((batch, candidates, events), dtype=torch.float32)
        raw_query = query_tokens.new_zeros((batch, candidates), dtype=torch.float32)
        normalized_event_weights = event_weights.new_zeros(event_weights.shape, dtype=torch.float32)

        for row in range(batch):
            candidate_positions = torch.where(candidate_mask[row])[0]
            query_positions = torch.where(query_token_mask[row])[0]
            event_positions = torch.where(event_mask[row])[0]
            if not len(candidate_positions) or not len(query_positions):
                continue
            query = query_tokens[row, query_positions]
            candidate = candidate_tokens[row, candidate_positions]
            candidate_content = candidate_token_mask[row, candidate_positions]
            if bool(candidate_content.any(dim=-1).logical_not().any()):
                raise ValueError("C59 valid candidate lacks content token")
            raw_query[row, candidate_positions] = self._raw_query_similarity(
                query, candidate, candidate_content
            )
            if not len(event_positions):
                continue
            weights = event_weights[row, event_positions].float().clamp_min(0.0)
            total = weights.sum()
            if not bool(torch.isfinite(total)) or float(total) <= self.epsilon:
                raise ValueError("C59 active history lacks positive finite event weight")
            normalized_event_weights[row, event_positions] = weights / total
            for event_position in event_positions:
                token_positions = torch.where(history_token_mask[row, event_position])[0]
                event = history_tokens[row, event_position, token_positions]
                query_event = (query @ event.T).max(dim=-1).values.mean()
                candidate_event = self._candidate_event_similarity(
                    candidate, candidate_content, event
                )
                triadic[row, candidate_positions, event_position] = query_event * candidate_event

        features = {
            mode: query_tokens.new_zeros((batch, candidates), dtype=torch.float32)
            for mode in MODES
        }
        allocation = query_tokens.new_zeros((batch, candidates, events), dtype=torch.float32)
        null_allocation = query_tokens.new_zeros((batch, events), dtype=torch.float32)
        for row in torch.where(active)[0]:
            candidate_positions = torch.where(candidate_mask[row])[0]
            event_positions = torch.where(event_mask[row])[0]
            logits = triadic[row][candidate_positions][:, event_positions]
            weights = normalized_event_weights[row, event_positions]

            # Each event distributes one finite budget over candidates + NULL.
            event_logits = logits.T
            null = event_logits.new_full((len(event_positions), 1), self.null_logit)
            probability = self._symmetric_softmax(torch.cat((event_logits, null), dim=-1))
            primary_probability = probability[:, :-1]
            features["candidate_budget"][row, candidate_positions] = (
                weights[:, None] * primary_probability
            ).sum(dim=0)
            allocation[row][candidate_positions[:, None], event_positions[None, :]] = primary_probability.T
            null_allocation[row, event_positions] = probability[:, -1]

            no_null = self._symmetric_softmax(event_logits)
            features["slot_budget_no_null"][row, candidate_positions] = (
                weights[:, None] * no_null
            ).sum(dim=0)

            # Nearest target/history-attention reduction with the same logits.
            history_logits = logits + torch.log(weights.clamp_min(self.epsilon))[None, :]
            history_null = history_logits.new_full((len(candidate_positions), 1), self.null_logit)
            history_probability = torch.softmax(
                torch.cat((history_logits, history_null), dim=-1), dim=-1
            )[:, :-1]
            features["history_softmax"][row, candidate_positions] = (
                history_probability * logits
            ).sum(dim=-1)
            features["pooled_history"][row, candidate_positions] = (
                logits * weights[None, :]
            ).sum(dim=-1)
            features["raw_query"][row, candidate_positions] = raw_query[
                row, candidate_positions
            ]

        corrections = {
            mode: self._zscore(value, candidate_mask) * active[:, None].to(torch.float32)
            for mode, value in features.items()
        }
        base = base_scores.float().masked_fill(~candidate_mask, 0.0)
        item_only = item_only_scores.float().masked_fill(~candidate_mask, 0.0)
        scores = {}
        for mode in MODES:
            score = base + corrections[mode]
            score = torch.where(repeat_request[:, None], item_only, score)
            scores[mode] = score.masked_fill(~candidate_mask, 0.0)
        return SemanticBudgetOutput(
            scores=scores,
            corrections=corrections,
            features=features,
            triadic_logits=triadic,
            primary_allocation=allocation,
            null_allocation=null_allocation,
            active_request=active,
        )
