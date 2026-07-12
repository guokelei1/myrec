"""CPU reference prototype for C09 cross-view agreement attention.

This module is deliberately self contained.  It accepts token tensors and never
opens data, label, qrels, run, or candidate-manifest files.  The model is a tiny
end-to-end Transformer ranker intended for structural tests, not a production
training implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


def positive_margin_meet(left: Tensor, right: Tensor, eps: float = 1e-12) -> Tensor:
    """Return a differentiable conjunctive meet of two oriented margins.

    The result is positive iff *both* inputs are positive.  On the positive
    orthant it is the parallel sum ``left * right / (left + right)`` and is no
    larger than the weaker margin.  It is exactly zero on disagreement.
    """

    left_pos = torch.relu(left)
    right_pos = torch.relu(right)
    denominator = left_pos + right_pos
    safe_denominator = torch.where(
        denominator > 0,
        denominator + eps,
        torch.ones_like(denominator),
    )
    value = left_pos * right_pos / safe_denominator
    return torch.where(denominator > 0, value, torch.zeros_like(value))


@dataclass(frozen=True)
class AgreementDiagnostics:
    """Inspectable tensors returned by :class:`AgreementContrastAttention`."""

    margin_query_first: Tensor
    margin_candidate_first: Tensor
    agreement_strength: Tensor
    attention_weight: Tensor
    correction: Tensor


class AgreementContrastAttention(nn.Module):
    """Candidate-set attention opened only by cross-view margin agreement.

    This is a matrix-valued attention operator, not a scalar interpolation.
    History affects candidate ``i`` only through oriented pairs ``(i, j)`` for
    which both restricted views give a positive history-residual margin.
    """

    def __init__(self, d_model: int, eps: float = 1e-12) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, 1, bias=False)

    def forward(
        self,
        base_scores: Tensor,
        base_hidden: Tensor,
        query_first_scores: Tensor,
        candidate_first_scores: Tensor,
        evidence_available: Tensor,
    ) -> tuple[Tensor, AgreementDiagnostics]:
        """Apply agreement attention.

        Args:
            base_scores: Query/candidate-only logits, shape ``[B, N]``.
            base_hidden: Base candidate states, shape ``[B, N, D]``.
            query_first_scores: Restricted-view logits, shape ``[B, N]``.
            candidate_first_scores: Restricted-view logits, shape ``[B, N]``.
            evidence_available: Boolean query-and-history mask, shape ``[B]``.
        """

        if base_hidden.shape[:2] != base_scores.shape:
            raise ValueError("base hidden and score candidate axes must match")
        if base_hidden.shape[-1] != self.d_model:
            raise ValueError("unexpected hidden dimension")
        if query_first_scores.shape != base_scores.shape:
            raise ValueError("query-first scores must match base scores")
        if candidate_first_scores.shape != base_scores.shape:
            raise ValueError("candidate-first scores must match base scores")
        if evidence_available.shape != base_scores.shape[:1]:
            raise ValueError("evidence_available must have shape [batch]")

        residual_q = query_first_scores - base_scores
        residual_c = candidate_first_scores - base_scores
        margin_q = residual_q.unsqueeze(-1) - residual_q.unsqueeze(-2)
        margin_c = residual_c.unsqueeze(-1) - residual_c.unsqueeze(-2)

        strength = positive_margin_meet(margin_q, margin_c, eps=self.eps)
        candidate_count = base_scores.shape[-1]
        diagonal = torch.eye(
            candidate_count,
            dtype=torch.bool,
            device=base_scores.device,
        ).unsqueeze(0)
        strength = strength.masked_fill(diagonal, 0.0)

        # The +1 is an explicit null-attention sink.  Therefore a weakly
        # supported row stays weak, and a row with no agreed pair is exactly 0.
        weight = strength / (1.0 + strength.sum(dim=-1, keepdim=True))
        contrast = base_hidden.unsqueeze(2) - base_hidden.unsqueeze(1)
        values = self.value_projection(contrast)
        context = (weight.unsqueeze(-1) * values).sum(dim=2)
        correction = self.output_projection(context).squeeze(-1)

        available = evidence_available.to(dtype=torch.bool).unsqueeze(-1)
        scores = torch.where(available, base_scores + correction, base_scores)
        diagnostics = AgreementDiagnostics(
            margin_query_first=margin_q,
            margin_candidate_first=margin_c,
            agreement_strength=strength,
            attention_weight=weight,
            correction=torch.where(available, correction, torch.zeros_like(correction)),
        )
        return scores, diagnostics


class SharedSegmentEncoder(nn.Module):
    """One shared token Transformer for queries, history items, and candidates."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        max_tokens: int,
    ) -> None:
        super().__init__()
        self.max_tokens = max_tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_tokens, d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim != 2:
            raise ValueError("segment token ids must have shape [batch, length]")
        if token_ids.shape[1] > self.max_tokens:
            raise ValueError("segment exceeds configured max_tokens")

        valid = token_ids.ne(0)
        has_token = valid.any(dim=-1)
        safe_valid = valid.clone()
        if (~has_token).any():
            safe_valid[~has_token, 0] = True

        positions = torch.arange(token_ids.shape[1], device=token_ids.device)
        hidden = self.token_embedding(token_ids) + self.position_embedding(positions)[None]
        encoded = self.encoder(hidden, src_key_padding_mask=~safe_valid)
        weights = valid.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = self.output_norm(pooled)
        return torch.where(has_token.unsqueeze(-1), pooled, torch.zeros_like(pooled))


class CrossViewAgreementTransformer(nn.Module):
    """Minimal end-to-end shared-Transformer personalized ranker.

    View Q first lets a query mediator attend to history while remaining blind
    to every candidate.  View C first lets each candidate mediator attend to
    history while remaining blind to the query.  Both use the same attention,
    rank Transformer, and output head.  Their logits are never averaged.
    """

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 16,
        nhead: int = 4,
        dim_feedforward: int = 32,
        max_segment_tokens: int = 8,
        max_history: int = 16,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_history = max_history
        self.segment_encoder = SharedSegmentEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            max_tokens=max_segment_tokens,
        )
        self.history_position = nn.Embedding(max_history, d_model)
        self.history_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # This exact module is reused for Q->H and C->H mediation.
        self.mediator_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True,
        )
        self.rank_token = nn.Parameter(torch.zeros(d_model))
        self.null_mediator = nn.Parameter(torch.zeros(d_model))
        self.role_embedding = nn.Parameter(torch.zeros(4, d_model))
        self.rank_encoder = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.rank_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1, bias=False)
        self.agreement_attention = AgreementContrastAttention(d_model=d_model)

        nn.init.normal_(self.rank_token, std=0.02)
        nn.init.normal_(self.null_mediator, std=0.02)
        nn.init.normal_(self.role_embedding, std=0.02)

    def _encode_inputs(
        self,
        query_token_ids: Tensor,
        history_token_ids: Tensor,
        candidate_token_ids: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if history_token_ids.ndim != 3:
            raise ValueError("history ids must have shape [B, H, L]")
        if candidate_token_ids.ndim != 3:
            raise ValueError("candidate ids must have shape [B, N, L]")
        batch, history_count, history_width = history_token_ids.shape
        candidate_batch, candidate_count, candidate_width = candidate_token_ids.shape
        if candidate_batch != batch or query_token_ids.shape[0] != batch:
            raise ValueError("input batch dimensions must match")
        if history_mask.shape != (batch, history_count):
            raise ValueError("history mask must have shape [B, H]")
        if history_count > self.max_history:
            raise ValueError("history exceeds configured max_history")

        query = self.segment_encoder(query_token_ids)
        history = self.segment_encoder(history_token_ids.reshape(-1, history_width))
        history = history.reshape(batch, history_count, self.d_model)
        candidates = self.segment_encoder(candidate_token_ids.reshape(-1, candidate_width))
        candidates = candidates.reshape(batch, candidate_count, self.d_model)

        valid_history = history_mask.to(dtype=torch.bool)
        has_history = valid_history.any(dim=-1)
        safe_mask = valid_history.clone()
        safe_history = history.clone()
        if (~has_history).any():
            safe_mask[~has_history, 0] = True
            safe_history[~has_history, 0] = 0.0

        positions = torch.arange(history_count, device=history.device)
        contextual_history = safe_history + self.history_position(positions)[None]
        contextual_history = self.history_encoder(
            contextual_history,
            src_key_padding_mask=~safe_mask,
        )
        contextual_history = torch.where(
            valid_history.unsqueeze(-1),
            contextual_history,
            torch.zeros_like(contextual_history),
        )
        if (~has_history).any():
            # Keep one finite zero key for MHA; the availability mask later
            # enforces exact fallback, so this dummy cannot alter final scores.
            contextual_history = contextual_history.clone()
            contextual_history[~has_history, 0] = 0.0
        return query, contextual_history, candidates, safe_mask

    def encode_restricted_mediators(
        self,
        query_token_ids: Tensor,
        history_token_ids: Tensor,
        candidate_token_ids: Tensor,
        history_mask: Tensor,
    ) -> Dict[str, Tensor]:
        """Expose the two information-restricted mediators for barrier tests."""

        query, history, candidates, safe_history_mask = self._encode_inputs(
            query_token_ids,
            history_token_ids,
            candidate_token_ids,
            history_mask,
        )
        query_mediator, _ = self.mediator_attention(
            query=query.unsqueeze(1),
            key=history,
            value=history,
            key_padding_mask=~safe_history_mask,
            need_weights=False,
        )

        batch, candidate_count, _ = candidates.shape
        expanded_history = history[:, None].expand(-1, candidate_count, -1, -1)
        expanded_history = expanded_history.reshape(
            batch * candidate_count,
            history.shape[1],
            self.d_model,
        )
        expanded_mask = safe_history_mask[:, None].expand(-1, candidate_count, -1)
        expanded_mask = expanded_mask.reshape(batch * candidate_count, history.shape[1])
        candidate_mediator, _ = self.mediator_attention(
            query=candidates.reshape(batch * candidate_count, 1, self.d_model),
            key=expanded_history,
            value=expanded_history,
            key_padding_mask=~expanded_mask,
            need_weights=False,
        )
        candidate_mediator = candidate_mediator.reshape(
            batch,
            candidate_count,
            self.d_model,
        )
        return {
            "query": query,
            "candidates": candidates,
            "query_first_mediator": query_mediator.squeeze(1),
            "candidate_first_mediator": candidate_mediator,
        }

    def _score_candidates(
        self,
        query: Tensor,
        candidates: Tensor,
        mediator: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, candidate_count, _ = candidates.shape
        expanded_query = query[:, None].expand(-1, candidate_count, -1)
        expanded_rank = self.rank_token[None, None].expand(batch, candidate_count, -1)
        sequence = torch.stack(
            [expanded_rank, expanded_query, candidates, mediator],
            dim=2,
        )
        sequence = sequence + self.role_embedding[None, None]
        encoded = self.rank_encoder(sequence.reshape(-1, 4, self.d_model))
        rank_hidden = self.rank_norm(encoded[:, 0]).reshape(
            batch,
            candidate_count,
            self.d_model,
        )
        scores = self.score_head(rank_hidden).squeeze(-1)
        return scores, rank_hidden

    def forward(
        self,
        query_token_ids: Tensor,
        history_token_ids: Tensor,
        candidate_token_ids: Tensor,
        history_mask: Tensor,
        query_present: Optional[Tensor] = None,
    ) -> Dict[str, Tensor | AgreementDiagnostics]:
        mediators = self.encode_restricted_mediators(
            query_token_ids,
            history_token_ids,
            candidate_token_ids,
            history_mask,
        )
        query = mediators["query"]
        candidates = mediators["candidates"]
        candidate_count = candidates.shape[1]

        null_mediator = self.null_mediator[None, None].expand(
            candidates.shape[0],
            candidate_count,
            -1,
        )
        query_mediator = mediators["query_first_mediator"][:, None].expand(
            -1,
            candidate_count,
            -1,
        )
        candidate_mediator = mediators["candidate_first_mediator"]

        base_scores, base_hidden = self._score_candidates(
            query,
            candidates,
            null_mediator,
        )
        query_first_scores, _ = self._score_candidates(
            query,
            candidates,
            query_mediator,
        )
        candidate_first_scores, _ = self._score_candidates(
            query,
            candidates,
            candidate_mediator,
        )

        if query_present is None:
            query_present = query_token_ids.ne(0).any(dim=-1)
        history_present = history_mask.to(dtype=torch.bool).any(dim=-1)
        evidence_available = query_present.to(dtype=torch.bool) & history_present
        scores, diagnostics = self.agreement_attention(
            base_scores=base_scores,
            base_hidden=base_hidden,
            query_first_scores=query_first_scores,
            candidate_first_scores=candidate_first_scores,
            evidence_available=evidence_available,
        )
        return {
            "scores": scores,
            "base_scores": base_scores,
            "query_first_scores": query_first_scores,
            "candidate_first_scores": candidate_first_scores,
            "base_hidden": base_hidden,
            "evidence_available": evidence_available,
            "diagnostics": diagnostics,
        }
