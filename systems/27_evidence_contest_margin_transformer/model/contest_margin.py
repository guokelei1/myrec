"""Evidence-conditioned antisymmetric candidate-contest Transformer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn


MODES = ("evidence_contest", "generic_contest", "candidate_contest", "additive_node")


@dataclass(frozen=True)
class ContestOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    anchor_scores: torch.Tensor
    active_request: torch.Tensor
    repeat_request: torch.Tensor
    pair_delta: torch.Tensor
    pair_complement_error: torch.Tensor
    pair_diagonal_error: torch.Tensor


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    mask = mask.to(torch.bool)
    masked = logits.masked_fill(~mask, -torch.inf)
    maximum = masked.amax(dim=dim, keepdim=True)
    maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    values = torch.exp(masked - maximum) * mask.to(logits.dtype)
    return values / values.sum(dim=dim, keepdim=True).clamp_min(1e-12)


class EvidenceContestMarginTransformer(nn.Module):
    def __init__(
        self,
        *,
        embedding_weight: torch.Tensor,
        padding_idx: int,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        token_layers: int,
        history_layers: int,
        ffn_dim: int,
        dropout: float,
        max_query_tokens: int,
        max_item_tokens: int,
        max_history: int,
        pair_delta_max: float,
        additive_delta_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C27 mode: {mode}")
        if hidden_dim % heads or embedding_weight.ndim != 2:
            raise ValueError("C27 hidden/head/embedding shape differs")
        if embedding_weight.shape[1] != input_dim:
            raise ValueError("C27 embedding input dimension differs")
        self.mode = mode
        self.padding_idx = int(padding_idx)
        self.max_query_tokens = int(max_query_tokens)
        self.max_item_tokens = int(max_item_tokens)
        self.max_history = int(max_history)
        self.pair_delta_max = float(pair_delta_max)
        self.additive_delta_max = float(additive_delta_max)
        self.word_embeddings = nn.Embedding.from_pretrained(
            embedding_weight.detach().clone(), freeze=True, padding_idx=padding_idx
        )
        self.token_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.position_embedding = nn.Parameter(
            torch.empty(max(max_query_tokens, max_item_tokens), hidden_dim)
        )
        token_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_transformer = nn.TransformerEncoder(
            token_block, num_layers=token_layers, enable_nested_tensor=False
        )
        self.bridge_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.read_token = nn.Parameter(torch.zeros(hidden_dim))
        history_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(
            history_block, num_layers=history_layers, enable_nested_tensor=False
        )
        self.candidate_norm = nn.LayerNorm(hidden_dim)
        self.evidence_norm = nn.LayerNorm(hidden_dim)
        self.node_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.odd_comparator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.read_token, std=0.02)
        for module in self.odd_comparator:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def _encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        prefix, length = token_ids.shape[:-1], token_ids.shape[-1]
        if length > self.position_embedding.shape[0] or token_ids.shape != attention_mask.shape:
            raise ValueError("C27 token encoder shape/length differs")
        flat_ids = token_ids.reshape(-1, length)
        flat_mask = attention_mask.reshape(-1, length).to(torch.bool)
        safe_mask = flat_mask.clone()
        empty = ~safe_mask.any(dim=-1)
        if bool(empty.any()):
            safe_mask[empty, 0] = True
        value = self.token_projection(self.word_embeddings(flat_ids))
        value = value + self.position_embedding[:length].view(1, length, -1)
        encoded = self.token_transformer(value, src_key_padding_mask=~safe_mask)
        encoded = encoded * flat_mask[:, :, None].to(encoded.dtype)
        return encoded.reshape(*prefix, length, encoded.shape[-1])

    def candidate_and_evidence_states(
        self,
        *,
        query_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_content_mask: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_content_mask: torch.Tensor,
        history_token_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        history_content_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        event_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates = candidate_mask.shape
        history = history_mask.shape[1]
        q = self._encode(query_ids, query_attention_mask)
        c = self._encode(candidate_token_ids, candidate_attention_mask)
        h = self._encode(history_token_ids, history_attention_mask)
        scale = math.sqrt(q.shape[-1])

        candidate_logits = torch.einsum("bqd,bcld->bcql", q, c) / scale
        candidate_match_mask = candidate_content_mask[:, :, None, :].expand_as(candidate_logits)
        candidate_weights = masked_softmax(candidate_logits, candidate_match_mask, dim=-1)
        candidate_match = torch.einsum("bcql,bcld->bcqd", candidate_weights, c)
        candidate_strength = (
            candidate_weights * candidate_logits.masked_fill(~candidate_match_mask, 0.0)
        ).sum(-1)

        history_logits = torch.einsum("bqd,bhld->bhql", q, h) / scale
        history_match_mask = history_content_mask[:, :, None, :].expand_as(history_logits)
        history_weights = masked_softmax(history_logits, history_match_mask, dim=-1)
        history_match = torch.einsum("bhql,bhld->bhqd", history_weights, h)
        history_strength = (
            history_weights * history_logits.masked_fill(~history_match_mask, 0.0)
        ).sum(-1)

        query_mask = query_content_mask[:, None, None, :].expand(
            batch, candidates, history, -1
        )
        pivot_logits = candidate_strength[:, :, None, :] + history_strength[:, None, :, :]
        pivot_weights = masked_softmax(pivot_logits, query_mask, dim=-1)
        q_expanded = q[:, None, None, :, :]
        c_expanded = candidate_match[:, :, None, :, :]
        h_expanded = history_match[:, None, :, :, :]
        event_values = self.bridge_ffn(
            (c_expanded - q_expanded) * (h_expanded - q_expanded)
        )
        events = (event_values * pivot_weights[..., None]).sum(dim=-2)
        events = events * event_weights[:, None, :, None].to(events.dtype)

        candidate_pivot_mask = query_content_mask[:, None, :].expand(
            batch, candidates, -1
        )
        candidate_pivot = masked_softmax(candidate_strength, candidate_pivot_mask, dim=-1)
        candidate_values = self.bridge_ffn(candidate_match - q[:, None, :, :])
        candidate_state = (candidate_values * candidate_pivot[..., None]).sum(dim=-2)
        candidate_state = self.candidate_norm(candidate_state)
        candidate_state = candidate_state * candidate_mask[:, :, None].to(candidate_state.dtype)

        event_mask = history_mask[:, None, :].expand(-1, candidates, -1)
        event_mask = event_mask & candidate_mask[:, :, None]
        flat_events = events.reshape(batch * candidates, history, -1)
        flat_mask = event_mask.reshape(batch * candidates, history)
        read = self.read_token.view(1, 1, -1).expand(batch * candidates, -1, -1)
        sequence = torch.cat((read, flat_events), dim=1)
        padding = torch.cat(
            (
                torch.zeros(batch * candidates, 1, dtype=torch.bool, device=q.device),
                ~flat_mask,
            ),
            dim=1,
        )
        evidence_state = self.history_transformer(
            sequence, src_key_padding_mask=padding
        )[:, 0].reshape(batch, candidates, -1)
        evidence_state = self.evidence_norm(evidence_state)
        evidence_state = evidence_state * candidate_mask[:, :, None].to(evidence_state.dtype)
        return candidate_state, evidence_state

    def pair_delta(self, nodes: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        difference = nodes[:, :, None, :] - nodes[:, None, :, :]
        delta = self.pair_delta_max * torch.tanh(self.odd_comparator(difference).squeeze(-1))
        pair_mask = candidate_mask[:, :, None] & candidate_mask[:, None, :]
        return delta * pair_mask.to(delta.dtype)

    @staticmethod
    def contest_scores(
        base_scores: torch.Tensor, pair_delta: torch.Tensor, candidate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        candidates = candidate_mask.shape[1]
        eye = torch.eye(candidates, dtype=torch.bool, device=base_scores.device)[None]
        pair_mask = candidate_mask[:, :, None] & candidate_mask[:, None, :] & ~eye
        base_gap = base_scores[:, :, None] - base_scores[:, None, :]
        probability = torch.sigmoid(base_gap + pair_delta)
        neutral_probability = torch.sigmoid(base_gap)
        count = pair_mask.sum(dim=-1).clamp_min(1)
        borda = (probability * pair_mask.to(probability.dtype)).sum(dim=-1) / count
        neutral_borda = (
            neutral_probability * pair_mask.to(neutral_probability.dtype)
        ).sum(dim=-1) / count
        eps = torch.finfo(borda.dtype).eps
        personalized_logit = torch.logit(borda.clamp(min=eps, max=1.0 - eps))
        neutral_logit = torch.logit(neutral_borda.clamp(min=eps, max=1.0 - eps))
        scores = base_scores + (personalized_logit - neutral_logit)
        scores = scores * candidate_mask.to(scores.dtype)
        complement = (probability + probability.transpose(1, 2) - 1.0).abs()
        complement_error = complement.masked_select(pair_mask).max() if bool(pair_mask.any()) else scores.new_zeros(())
        diagonal_error = pair_delta.diagonal(dim1=1, dim2=2).abs().max()
        return scores, complement_error, diagonal_error

    def forward(
        self,
        *,
        query_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_content_mask: torch.Tensor,
        candidate_token_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_content_mask: torch.Tensor,
        history_token_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        history_content_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        repeat_mask: torch.Tensor,
        event_weights: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        query_present: torch.Tensor | None = None,
    ) -> ContestOutput:
        batch, candidates = candidate_mask.shape
        history = history_mask.shape[1]
        if repeat_mask.shape != (batch, candidates, history):
            raise ValueError("C27 repeat mask differs")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != candidate_mask.shape:
            raise ValueError("C27 score shapes differ")
        if history > self.max_history:
            raise ValueError("C27 history exceeds registered maximum")
        if query_present is None:
            query_present = query_content_mask.any(dim=-1)
        else:
            query_present = query_present & query_content_mask.any(dim=-1)

        candidate_state, evidence_state = self.candidate_and_evidence_states(
            query_ids=query_ids,
            query_attention_mask=query_attention_mask,
            query_content_mask=query_content_mask,
            candidate_token_ids=candidate_token_ids,
            candidate_attention_mask=candidate_attention_mask,
            candidate_content_mask=candidate_content_mask,
            history_token_ids=history_token_ids,
            history_attention_mask=history_attention_mask,
            history_content_mask=history_content_mask,
            candidate_mask=candidate_mask,
            history_mask=history_mask,
            event_weights=event_weights,
        )
        evidence_nodes = self.node_ffn(candidate_state * evidence_state)
        generic_nodes = self.node_ffn(candidate_state + evidence_state)
        candidate_nodes = self.node_ffn(candidate_state)
        evidence_delta = self.pair_delta(evidence_nodes, candidate_mask)
        generic_delta = self.pair_delta(generic_nodes, candidate_mask)
        candidate_delta = self.pair_delta(candidate_nodes, candidate_mask)
        evidence_scores, complement_error, diagonal_error = self.contest_scores(
            base_scores, evidence_delta, candidate_mask
        )
        generic_scores, _, _ = self.contest_scores(base_scores, generic_delta, candidate_mask)
        candidate_scores, _, _ = self.contest_scores(base_scores, candidate_delta, candidate_mask)

        raw_additive = self.odd_comparator(evidence_nodes).squeeze(-1)
        weights = candidate_mask.to(raw_additive.dtype)
        raw_additive = (
            raw_additive
            - (raw_additive * weights).sum(dim=-1, keepdim=True)
            / weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        ) * weights
        additive_scores = base_scores + self.additive_delta_max * torch.tanh(raw_additive)

        if self.mode == "evidence_contest":
            active_scores, selected_delta = evidence_scores, evidence_delta
        elif self.mode == "generic_contest":
            active_scores, selected_delta = generic_scores, generic_delta
        elif self.mode == "candidate_contest":
            active_scores, selected_delta = candidate_scores, candidate_delta
        else:
            active_scores, selected_delta = additive_scores, evidence_delta
        compute_match = (
            evidence_scores.sum()
            + generic_scores.sum()
            + candidate_scores.sum()
            + additive_scores.sum()
        ) * 0.0
        active_scores = active_scores + compute_match

        repeat_request = (
            repeat_mask & candidate_mask[:, :, None] & history_mask[:, None, :]
        ).any(dim=(1, 2))
        active = history_mask.any(dim=-1) & ~repeat_request & query_present
        anchor = torch.where(repeat_request[:, None], item_only_scores, base_scores)
        scores = torch.where(active[:, None], active_scores, anchor)
        scores = scores.masked_fill(~candidate_mask, 0.0)
        correction = (scores - anchor) * candidate_mask.to(scores.dtype)
        return ContestOutput(
            scores=scores,
            correction=correction,
            anchor_scores=anchor,
            active_request=active,
            repeat_request=repeat_request,
            pair_delta=selected_delta,
            pair_complement_error=complement_error,
            pair_diagonal_error=diagonal_error,
        )

    def parameter_count(self, *, trainable_only: bool = False) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if not trainable_only or parameter.requires_grad
        )
