"""Predictive evidence-write Transformer.

The same Transformer is reused for the history-blind candidate path and the
two masked-context prediction streams.  History can affect ranking only via a
late, bounded, candidate-centred write or the monotone exact-token channel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Mode = Literal[
    "predictive_gain",
    "paired_logit",
    "single_pass",
    "dual_stream",
    "centered_attention",
    "base",
]


@dataclass(frozen=True)
class EvidenceWriteOutput:
    scores: Tensor
    base_scores: Tensor
    token_features: Tensor
    hidden_write: Tensor
    repeat_count: Tensor
    identity_coordinate: Tensor
    identity_contribution: Tensor
    log_probs_history: Tensor
    log_probs_query_only: Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def candidate_center(values: Tensor) -> Tensor:
    """Remove the candidate-common component (candidate axis is dimension 1)."""

    if values.ndim < 2:
        raise ValueError("candidate-centred tensors must have at least two axes")
    return values - values.mean(dim=1, keepdim=True)


def bounded_zero_sum_write(raw_write: Tensor, radius: float, eps: float = 1e-8) -> Tensor:
    """Project to zero candidate sum and smoothly bound every candidate norm.

    A single per-request scale preserves exact zero sum.  The largest candidate
    norm is strictly below ``radius`` whenever it is non-zero.
    """

    centred = candidate_center(raw_write)
    max_norm = centred.norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)
    scale = radius / (radius + max_norm + eps)
    return centred * scale


def predictive_token_evidence(token_features: Tensor, candidate_embeddings: Tensor) -> Tensor:
    """Turn per-token predictive evidence into a candidate hidden innovation."""

    if token_features.shape != candidate_embeddings.shape[:-1]:
        raise ValueError("token feature and candidate-token shapes do not agree")
    return (torch.tanh(token_features).unsqueeze(-1) * candidate_embeddings).mean(dim=2)


class MonotoneEvidenceRankingHead(nn.Module):
    """Read hidden state and a reserved exact-identity evidence coordinate.

    The identity coordinate is part of the late internal evidence interface,
    not a score produced by another scorer.  Its final-logit derivative is
    ``softplus(identity_log_weight) > 0`` by construction.
    """

    def __init__(self, d_model: int, initial_identity_weight: float) -> None:
        super().__init__()
        self.hidden_weight = nn.Parameter(torch.empty(1, d_model))
        self.bias = nn.Parameter(torch.zeros(1))
        self.identity_log_weight = nn.Parameter(
            torch.tensor(_inverse_softplus(initial_identity_weight))
        )
        nn.init.xavier_uniform_(self.hidden_weight)

    @property
    def identity_weight(self) -> Tensor:
        return F.softplus(self.identity_log_weight)

    def forward(self, hidden: Tensor, identity_coordinate: Tensor) -> Tensor:
        hidden_logit = F.linear(hidden, self.hidden_weight, self.bias).squeeze(-1)
        return hidden_logit + self.identity_weight * identity_coordinate


class PredictiveEvidenceWriteTransformer(nn.Module):
    """A compact LM ranker with a late predictive-information write interface.

    ``predictive_gain`` is the proposed primitive.  Other modes keep the same
    parameters and initialization while changing only the history-write rule.
    This makes ``single_pass``, ``dual_stream``, and ``centered_attention``
    exact parameter-count controls.  ``paired_logit`` is the nearest-neighbour
    final-logit-delta control; its write matrix is intentionally bypassed.
    """

    MODES = {
        "predictive_gain",
        "paired_logit",
        "single_pass",
        "dual_stream",
        "centered_attention",
        "base",
    }

    def __init__(
        self,
        *,
        vocab_size: int,
        candidate_token_count: int,
        d_model: int = 32,
        nhead: int = 4,
        num_layers: int = 1,
        dim_feedforward: int = 64,
        max_sequence_length: int = 64,
        dropout: float = 0.0,
        max_write_norm: float = 1.0,
        initial_repeat_bonus: float = 0.25,
        mode: Mode = "predictive_gain",
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        if vocab_size < 3 or candidate_token_count < 1:
            raise ValueError("invalid vocabulary or candidate token count")
        if max_write_norm <= 0:
            raise ValueError("max_write_norm must be positive")
        self.mode = mode
        self.vocab_size = vocab_size
        self.candidate_token_count = candidate_token_count
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.max_write_norm = float(max_write_norm)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls_embedding = nn.Parameter(torch.empty(d_model))
        self.position_embedding = nn.Parameter(torch.empty(max_sequence_length + 1, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.shared_lm = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(d_model))
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))
        self.write_projection = nn.Linear(d_model, d_model, bias=False)
        self.ranking_head = MonotoneEvidenceRankingHead(d_model, initial_repeat_bonus)
        self.paired_log_scale = nn.Parameter(torch.tensor(_inverse_softplus(0.25)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        nn.init.normal_(self.cls_embedding, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.write_projection.weight)
        nn.init.xavier_uniform_(self.ranking_head.hidden_weight)
        nn.init.zeros_(self.ranking_head.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _encode(self, tokens: Tensor, token_mask: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 2 or token_mask.shape != tokens.shape:
            raise ValueError("tokens and token_mask must both be [batch, length]")
        if tokens.shape[1] > self.max_sequence_length:
            raise ValueError("sequence exceeds max_sequence_length")
        batch, length = tokens.shape
        cls = self.cls_embedding.view(1, 1, -1).expand(batch, 1, -1)
        embedded = self.token_embedding(tokens)
        hidden = torch.cat((cls, embedded), dim=1)
        hidden = hidden + self.position_embedding[: length + 1].unsqueeze(0)
        cls_mask = torch.ones((batch, 1), dtype=torch.bool, device=tokens.device)
        visible = torch.cat((cls_mask, token_mask), dim=1)
        encoded = self.shared_lm(hidden, src_key_padding_mask=~visible)
        return encoded[:, 0], encoded[:, 1:]

    def _base_candidate_states(self, query_tokens: Tensor, candidate_tokens: Tensor) -> Tensor:
        batch, candidates, token_count = candidate_tokens.shape
        if token_count != self.candidate_token_count:
            raise ValueError("candidate token count differs from the registered architecture")
        query = query_tokens[:, None, :].expand(-1, candidates, -1)
        sequences = torch.cat((query, candidate_tokens), dim=-1).reshape(batch * candidates, -1)
        visible = sequences.ne(0)
        states, _ = self._encode(sequences, visible)
        return states.reshape(batch, candidates, self.d_model)

    def _context_states(
        self,
        evidence_query_tokens: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, events, event_tokens = history_tokens.shape
        flat_history = history_tokens.reshape(batch, events * event_tokens)
        flat_mask = history_mask.unsqueeze(-1).expand(-1, -1, event_tokens).reshape(batch, -1)
        query_mask = evidence_query_tokens.ne(0)

        history_sequence = torch.cat((evidence_query_tokens, flat_history), dim=1)
        history_visible = torch.cat((query_mask, flat_mask), dim=1)
        history_context, history_encoded = self._encode(history_sequence, history_visible)

        null_history = torch.zeros_like(flat_history)
        query_only_sequence = torch.cat((evidence_query_tokens, null_history), dim=1)
        query_only_visible = torch.cat((query_mask, torch.zeros_like(flat_mask)), dim=1)
        query_context, _ = self._encode(query_only_sequence, query_only_visible)

        offset = evidence_query_tokens.shape[1]
        encoded_history = history_encoded[:, offset:]
        return history_context, query_context, encoded_history, flat_mask, flat_history

    def _candidate_log_probs(self, context: Tensor, candidate_tokens: Tensor) -> Tensor:
        logits = F.linear(context, self.token_embedding.weight, self.decoder_bias)
        log_probs = F.log_softmax(logits, dim=-1)
        batch, candidates, token_count = candidate_tokens.shape
        expanded = log_probs[:, None, :].expand(-1, candidates, -1)
        return expanded.gather(dim=-1, index=candidate_tokens.reshape(batch, candidates, token_count))

    def _repeat_channel(
        self,
        candidate_tokens: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        candidate_item = candidate_tokens[:, :, 0]
        history_item = history_tokens[:, :, 0]
        matches = candidate_item.unsqueeze(-1).eq(history_item.unsqueeze(1))
        matches = matches & history_mask.unsqueeze(1)
        count = matches.sum(dim=-1).to(candidate_tokens.dtype if candidate_tokens.is_floating_point() else torch.float32)
        identity_coordinate = torch.log1p(count)
        return count, identity_coordinate

    def forward(
        self,
        *,
        query_tokens: Tensor,
        candidate_tokens: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
        evidence_query_tokens: Tensor | None = None,
    ) -> EvidenceWriteOutput:
        if query_tokens.ndim != 2 or candidate_tokens.ndim != 3 or history_tokens.ndim != 3:
            raise ValueError("query/candidate/history ranks must be 2/3/3")
        if history_mask.shape != history_tokens.shape[:2]:
            raise ValueError("history_mask must be [batch, events]")
        if evidence_query_tokens is None:
            evidence_query_tokens = query_tokens

        base_hidden = self._base_candidate_states(query_tokens, candidate_tokens)
        zero_identity = torch.zeros(base_hidden.shape[:2], dtype=base_hidden.dtype, device=base_hidden.device)
        base_scores = self.ranking_head(base_hidden, zero_identity)
        history_context, query_context, encoded_history, flat_history_mask, _ = self._context_states(
            evidence_query_tokens, history_tokens, history_mask
        )
        log_probs_history = self._candidate_log_probs(history_context, candidate_tokens)
        log_probs_query = self._candidate_log_probs(query_context, candidate_tokens)
        log_ratio = log_probs_history - log_probs_query
        token_embeddings = self.token_embedding(candidate_tokens)

        token_features = torch.zeros_like(log_ratio)
        hidden_write = torch.zeros_like(base_hidden)
        history_scores = base_scores

        if self.mode == "predictive_gain":
            token_features = candidate_center(log_ratio)
            innovation = predictive_token_evidence(token_features, token_embeddings)
            hidden_write = bounded_zero_sum_write(
                self.write_projection(innovation), self.max_write_norm
            )
            history_scores = self.ranking_head(base_hidden + hidden_write, zero_identity)
        elif self.mode == "single_pass":
            token_features = candidate_center(log_probs_history)
            innovation = predictive_token_evidence(token_features, token_embeddings)
            hidden_write = bounded_zero_sum_write(
                self.write_projection(innovation), self.max_write_norm
            )
            history_scores = self.ranking_head(base_hidden + hidden_write, zero_identity)
        elif self.mode == "dual_stream":
            context_delta = history_context - query_context
            token_features = torch.einsum("bctd,bd->bct", token_embeddings, context_delta) / math.sqrt(self.d_model)
            token_features = candidate_center(token_features)
            innovation = predictive_token_evidence(token_features, token_embeddings)
            hidden_write = bounded_zero_sum_write(
                self.write_projection(innovation), self.max_write_norm
            )
            history_scores = self.ranking_head(base_hidden + hidden_write, zero_identity)
        elif self.mode == "centered_attention":
            attention_logits = torch.einsum("bcd,bld->bcl", base_hidden, encoded_history) / math.sqrt(self.d_model)
            attention_logits = attention_logits.masked_fill(~flat_history_mask[:, None, :], -torch.inf)
            attention = torch.softmax(attention_logits, dim=-1)
            attention = torch.nan_to_num(attention, nan=0.0)
            values = encoded_history - query_context[:, None, :]
            innovation = torch.einsum("bcl,bld->bcd", attention, values)
            hidden_write = bounded_zero_sum_write(
                self.write_projection(innovation), self.max_write_norm
            )
            history_scores = self.ranking_head(base_hidden + hidden_write, zero_identity)
        elif self.mode == "paired_logit":
            token_features = candidate_center(log_ratio)
            scalar_delta = candidate_center(log_ratio.mean(dim=-1))
            scalar_delta = torch.tanh(scalar_delta) * F.softplus(self.paired_log_scale)
            history_scores = base_scores + scalar_delta
        elif self.mode != "base":
            raise AssertionError("unreachable mode")

        repeat_count, identity_coordinate = self._repeat_channel(candidate_tokens, history_tokens, history_mask)
        if self.mode != "base":
            if self.mode == "paired_logit":
                # Paired-logit remains explicitly a nearest-neighbour control.
                history_scores = self.ranking_head(base_hidden, identity_coordinate) + scalar_delta
            else:
                history_scores = self.ranking_head(base_hidden + hidden_write, identity_coordinate)
        history_present = history_mask.any(dim=1, keepdim=True)
        scores = torch.where(history_present, history_scores, base_scores)
        identity_contribution = self.ranking_head.identity_weight * identity_coordinate
        return EvidenceWriteOutput(
            scores=scores,
            base_scores=base_scores,
            token_features=token_features,
            hidden_write=hidden_write,
            repeat_count=repeat_count,
            identity_coordinate=identity_coordinate,
            identity_contribution=identity_contribution,
            log_probs_history=log_probs_history,
            log_probs_query_only=log_probs_query,
        )
