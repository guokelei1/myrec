"""C11 Eventwise Predictive-Write Transformer.

The primitive preserves candidate-token x history-event conditional predictive
gain until a late Transformer integrator.  No history is pooled before the
candidate/event interaction in the primary mode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
import torch.nn.functional as F


Mode = Literal[
    "eventwise_predictive",
    "pooled_c10",
    "centered_attention",
    "scalar_logit",
    "eventwise_hidden",
    "base",
]


@dataclass(frozen=True)
class EventwiseWriteOutput:
    scores: Tensor
    base_scores: Tensor
    gain_matrix: Tensor
    event_innovations: Tensor
    hidden_write: Tensor
    repeat_count: Tensor
    identity_coordinate: Tensor
    identity_contribution: Tensor
    event_log_probs: Tensor
    query_log_probs: Tensor


def _inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def candidate_center(values: Tensor) -> Tensor:
    if values.ndim < 2:
        raise ValueError("candidate-centred tensors require a candidate axis")
    return values - values.mean(dim=1, keepdim=True)


def bounded_zero_sum_write(raw: Tensor, radius: float, eps: float = 1e-8) -> Tensor:
    centred = candidate_center(raw)
    max_norm = centred.norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)
    return centred * (radius / (radius + max_norm + eps))


def eventwise_token_innovations(gain: Tensor, candidate_embeddings: Tensor) -> Tensor:
    """Map [B,C,H,T] gains to [B,C,H,D] without pooling events."""

    expected = gain.shape[:2] + gain.shape[3:]
    if gain.ndim != 4 or candidate_embeddings.shape[:-1] != expected:
        raise ValueError("candidate token embeddings do not match gain tensor")
    return (
        torch.tanh(gain).unsqueeze(-1)
        * candidate_embeddings.unsqueeze(2)
    ).mean(dim=3)


def safe_masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    masked = logits.masked_fill(~mask, -torch.inf)
    probabilities = torch.softmax(masked, dim=dim)
    return torch.nan_to_num(probabilities, nan=0.0)


class MonotoneEvidenceHead(nn.Module):
    """One ranking head with an internal monotone exact-identity coordinate."""

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


class EventwisePredictiveWriteTransformer(nn.Module):
    """End-to-end Transformer ranker with a late eventwise predictive write."""

    MODES = {
        "eventwise_predictive",
        "pooled_c10",
        "centered_attention",
        "scalar_logit",
        "eventwise_hidden",
        "base",
    }

    def __init__(
        self,
        *,
        vocab_size: int,
        candidate_token_count: int,
        max_history_events: int,
        d_model: int = 32,
        nhead: int = 4,
        lm_layers: int = 1,
        integrator_layers: int = 1,
        dim_feedforward: int = 64,
        max_lm_sequence_length: int = 16,
        dropout: float = 0.0,
        max_write_norm: float = 1.0,
        initial_identity_weight: float = 0.25,
        mode: Mode = "eventwise_predictive",
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"unknown mode: {mode}")
        if candidate_token_count < 1 or max_history_events < 1:
            raise ValueError("candidate token count and history limit must be positive")
        if max_write_norm <= 0:
            raise ValueError("max_write_norm must be positive")
        self.mode = mode
        self.vocab_size = vocab_size
        self.candidate_token_count = candidate_token_count
        self.max_history_events = max_history_events
        self.d_model = d_model
        self.max_lm_sequence_length = max_lm_sequence_length
        self.max_write_norm = float(max_write_norm)

        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.cls_embedding = nn.Parameter(torch.empty(d_model))
        self.lm_position_embedding = nn.Parameter(
            torch.empty(max_lm_sequence_length + 1, d_model)
        )
        lm_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.shared_lm = nn.TransformerEncoder(
            lm_layer, num_layers=lm_layers, norm=nn.LayerNorm(d_model)
        )
        self.decoder_bias = nn.Parameter(torch.zeros(vocab_size))

        self.event_position_embedding = nn.Parameter(
            torch.empty(max_history_events, d_model)
        )
        integrator_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.late_event_integrator = nn.TransformerEncoder(
            integrator_layer,
            num_layers=integrator_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.event_pool_query = nn.Parameter(torch.empty(d_model))
        self.write_projection = nn.Linear(d_model, d_model, bias=False)
        self.ranking_head = MonotoneEvidenceHead(d_model, initial_identity_weight)
        self.scalar_log_scale = nn.Parameter(torch.tensor(_inverse_softplus(0.25)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()
        nn.init.normal_(self.cls_embedding, std=0.02)
        nn.init.normal_(self.lm_position_embedding, std=0.02)
        nn.init.normal_(self.event_position_embedding, std=0.02)
        nn.init.normal_(self.event_pool_query, std=0.02)
        nn.init.xavier_uniform_(self.write_projection.weight)
        nn.init.xavier_uniform_(self.ranking_head.hidden_weight)
        nn.init.zeros_(self.ranking_head.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def _encode(self, tokens: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        if tokens.ndim != 2 or mask.shape != tokens.shape:
            raise ValueError("tokens/mask must be [batch,length]")
        if tokens.shape[1] > self.max_lm_sequence_length:
            raise ValueError("LM sequence exceeds registered maximum")
        batch, length = tokens.shape
        cls = self.cls_embedding.view(1, 1, -1).expand(batch, 1, -1)
        hidden = torch.cat((cls, self.token_embedding(tokens)), dim=1)
        hidden = hidden + self.lm_position_embedding[: length + 1].unsqueeze(0)
        visible = torch.cat(
            (torch.ones((batch, 1), dtype=torch.bool, device=tokens.device), mask),
            dim=1,
        )
        encoded = self.shared_lm(hidden, src_key_padding_mask=~visible)
        return encoded[:, 0], encoded[:, 1:]

    def _base_states(self, query: Tensor, candidates: Tensor) -> Tensor:
        batch, count, token_count = candidates.shape
        if token_count != self.candidate_token_count:
            raise ValueError("candidate token count changed from the fingerprint")
        repeated_query = query[:, None, :].expand(-1, count, -1)
        sequences = torch.cat((repeated_query, candidates), dim=-1).reshape(batch * count, -1)
        states, _ = self._encode(sequences, sequences.ne(0))
        return states.reshape(batch, count, self.d_model)

    def _event_contexts(
        self,
        evidence_query: Tensor,
        history: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, events, _ = history.shape
        if events > self.max_history_events:
            raise ValueError("history exceeds registered maximum")
        query_only, _ = self._encode(evidence_query, evidence_query.ne(0))
        repeated_query = evidence_query[:, None, :].expand(-1, events, -1)
        sequences = torch.cat((repeated_query, history), dim=-1).reshape(batch * events, -1)
        visible = sequences.ne(0)
        event_context, _ = self._encode(sequences, visible)
        event_context = event_context.reshape(batch, events, self.d_model)
        # Masked events are made exactly query-only before any candidate interaction.
        event_context = torch.where(
            history_mask.unsqueeze(-1), event_context, query_only[:, None, :]
        )
        return query_only, event_context

    def _log_probs(self, context: Tensor) -> Tensor:
        return F.log_softmax(
            F.linear(context, self.token_embedding.weight, self.decoder_bias), dim=-1
        )

    def _gather_candidate_tokens(self, log_probs: Tensor, candidates: Tensor) -> Tensor:
        # [B,H,V] -> [B,C,H,T]
        batch, events, _ = log_probs.shape
        count, token_count = candidates.shape[1:]
        expanded = log_probs[:, None, :, :].expand(-1, count, -1, -1)
        indices = candidates[:, :, None, :].expand(-1, -1, events, -1)
        return expanded.gather(dim=-1, index=indices)

    def _identity_coordinate(
        self,
        candidates: Tensor,
        history: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        match = candidates[:, :, 0, None].eq(history[:, None, :, 0])
        count = (match & history_mask[:, None, :]).sum(dim=-1).to(torch.float32)
        return count, torch.log1p(count)

    def _late_integrate(self, innovations: Tensor, history_mask: Tensor) -> Tensor:
        batch, candidates, events, width = innovations.shape
        positions = self.event_position_embedding[:events].view(1, 1, events, width)
        sequence = (innovations + positions).reshape(batch * candidates, events, width)
        expanded_mask = history_mask[:, None, :].expand(-1, candidates, -1).reshape(
            batch * candidates, events
        )
        safe_mask = expanded_mask.clone()
        empty = ~safe_mask.any(dim=1)
        safe_mask[empty, 0] = True
        encoded = self.late_event_integrator(
            sequence, src_key_padding_mask=~safe_mask
        )
        logits = torch.einsum("bhd,d->bh", encoded, self.event_pool_query) / math.sqrt(width)
        weights = safe_masked_softmax(logits, expanded_mask, dim=1)
        pooled = torch.einsum("bh,bhd->bd", weights, encoded)
        return pooled.reshape(batch, candidates, width)

    def forward(
        self,
        *,
        query_tokens: Tensor,
        candidate_tokens: Tensor,
        history_tokens: Tensor,
        history_mask: Tensor,
        evidence_query_tokens: Tensor | None = None,
    ) -> EventwiseWriteOutput:
        if query_tokens.ndim != 2 or candidate_tokens.ndim != 3 or history_tokens.ndim != 3:
            raise ValueError("query/candidate/history tensors have invalid rank")
        if history_mask.shape != history_tokens.shape[:2]:
            raise ValueError("history_mask must be [batch,event]")
        if evidence_query_tokens is None:
            evidence_query_tokens = query_tokens

        base_hidden = self._base_states(query_tokens, candidate_tokens)
        zero_identity = torch.zeros(
            base_hidden.shape[:2], dtype=base_hidden.dtype, device=base_hidden.device
        )
        base_scores = self.ranking_head(base_hidden, zero_identity)
        query_context, event_context = self._event_contexts(
            evidence_query_tokens, history_tokens, history_mask
        )
        query_log_probs = self._log_probs(query_context)
        event_log_probs = self._log_probs(event_context)
        gathered_event = self._gather_candidate_tokens(event_log_probs, candidate_tokens)
        gathered_query = self._gather_candidate_tokens(
            query_log_probs[:, None, :], candidate_tokens
        ).squeeze(2)
        raw_gain = gathered_event - gathered_query.unsqueeze(2)
        candidate_embeddings = self.token_embedding(candidate_tokens)

        gain_matrix = torch.zeros_like(raw_gain)
        event_innovations = torch.zeros(
            *raw_gain.shape[:3], self.d_model,
            dtype=base_hidden.dtype,
            device=base_hidden.device,
        )
        hidden_write = torch.zeros_like(base_hidden)
        scalar_delta = torch.zeros_like(base_scores)

        if self.mode == "eventwise_predictive":
            gain_matrix = candidate_center(raw_gain)
            event_innovations = eventwise_token_innovations(
                gain_matrix, candidate_embeddings
            )
        elif self.mode == "pooled_c10":
            event_weight = history_mask.to(event_context.dtype)
            denominator = event_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled_context = (event_context * event_weight.unsqueeze(-1)).sum(dim=1) / denominator
            pooled_log_probs = self._log_probs(pooled_context)
            pooled_gathered = self._gather_candidate_tokens(
                pooled_log_probs[:, None, :], candidate_tokens
            ).squeeze(2)
            pooled_gain = candidate_center(pooled_gathered - gathered_query)
            gain_matrix = pooled_gain.unsqueeze(2).expand_as(raw_gain)
            event_innovations = eventwise_token_innovations(
                gain_matrix, candidate_embeddings
            )
        elif self.mode == "eventwise_hidden":
            context_delta = event_context - query_context[:, None, :]
            gain_matrix = torch.einsum(
                "bctd,bhd->bcht", candidate_embeddings, context_delta
            ) / math.sqrt(self.d_model)
            gain_matrix = candidate_center(gain_matrix)
            event_innovations = eventwise_token_innovations(
                gain_matrix, candidate_embeddings
            )
        elif self.mode == "centered_attention":
            value = event_context - query_context[:, None, :]
            similarity = torch.einsum("bcd,bhd->bch", base_hidden, value) / math.sqrt(
                self.d_model
            )
            similarity = similarity.masked_fill(~history_mask[:, None, :], -torch.inf)
            attention = safe_masked_softmax(
                similarity, history_mask[:, None, :].expand_as(similarity), dim=2
            )
            event_innovations = attention.unsqueeze(-1) * value[:, None, :, :]
            event_innovations = candidate_center(event_innovations)
        elif self.mode == "scalar_logit":
            gain_matrix = candidate_center(raw_gain)
            masked = gain_matrix * history_mask[:, None, :, None]
            denominator = history_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1)
            scalar = masked.sum(dim=(2, 3)) / (denominator.squeeze(-1) * self.candidate_token_count)
            scalar_delta = candidate_center(torch.tanh(scalar)) * F.softplus(
                self.scalar_log_scale
            )
        elif self.mode != "base":
            raise AssertionError("unreachable mode")

        if self.mode not in {"base", "scalar_logit"}:
            integrated = self._late_integrate(event_innovations, history_mask)
            hidden_write = bounded_zero_sum_write(
                self.write_projection(integrated), self.max_write_norm
            )

        repeat_count, identity_coordinate = self._identity_coordinate(
            candidate_tokens, history_tokens, history_mask
        )
        if self.mode == "base":
            history_scores = base_scores
        elif self.mode == "scalar_logit":
            history_scores = self.ranking_head(base_hidden, identity_coordinate) + scalar_delta
        else:
            history_scores = self.ranking_head(
                base_hidden + hidden_write, identity_coordinate
            )
        history_present = history_mask.any(dim=1, keepdim=True)
        scores = torch.where(history_present, history_scores, base_scores)
        return EventwiseWriteOutput(
            scores=scores,
            base_scores=base_scores,
            gain_matrix=gain_matrix,
            event_innovations=event_innovations,
            hidden_write=hidden_write,
            repeat_count=repeat_count,
            identity_coordinate=identity_coordinate,
            identity_contribution=self.ranking_head.identity_weight * identity_coordinate,
            event_log_probs=gathered_event,
            query_log_probs=gathered_query,
        )
