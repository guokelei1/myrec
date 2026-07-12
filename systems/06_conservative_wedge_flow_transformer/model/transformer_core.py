"""End-to-end block-sparse Transformer ranking core for C06.

The core keeps the ordinary query--candidate score path structurally blind to
history.  Ordered history may reach the final logits only through the
conservative wedge-flow layer.  Candidate segments share all parameters and
receive neither a candidate-index nor an item-ID embedding, which makes the
whole ranker equivariant to candidate permutations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .information_barrier import (
    CANDIDATE,
    HISTORY,
    PAD,
    QUERY,
    build_information_barrier_mask,
)
from .wedge_flow import ConservativeWedgeFlowProbeRanker, WedgeFlowOutput


@dataclass
class TransformerCoreOutput:
    """Outputs needed by ranking, audits, and information-flow tests."""

    scores: torch.Tensor
    base_scores: torch.Tensor
    applied_score_delta: torch.Tensor
    query_state: torch.Tensor
    candidate_states: torch.Tensor
    history_states: torch.Tensor
    token_states: torch.Tensor
    token_roles: torch.Tensor
    candidate_group_ids: torch.Tensor
    attention_mask: torch.Tensor
    wedge: WedgeFlowOutput


class _BarrierSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit per-example allow mask.

    PAD query rows intentionally have no allowed keys in the architectural
    mask.  The guarded softmax maps those empty rows to exactly zero instead of
    applying softmax to an all-``-inf`` row.
    """

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        dropout: float,
    ) -> None:
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")
        self.model_dim = int(model_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.model_dim // self.num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim)
        self.output = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_states: torch.Tensor,
        allowed: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, _ = token_states.shape
        if allowed.shape != (batch, length, length):
            raise ValueError("allowed attention mask shape mismatch")

        qkv = self.qkv(token_states).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        logits = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(
            float(self.head_dim)
        )

        allowed_heads = allowed[:, None, :, :]
        row_has_key = allowed_heads.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~allowed_heads, -torch.inf)
        # Avoid NaN for PAD rows while retaining the original all-false mask.
        logits = torch.where(row_has_key, logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(allowed_heads, weights, torch.zeros_like(weights))
        weights = self.dropout(weights)
        attended = torch.matmul(weights, value)
        attended = attended.transpose(1, 2).reshape(batch, length, self.model_dim)
        return self.output(attended)


class _BarrierTransformerLayer(nn.Module):
    """Pre-norm Transformer layer that zeros PAD states after each sublayer."""

    def __init__(
        self,
        model_dim: int,
        num_heads: int,
        *,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = _BarrierSelfAttention(
            model_dim, num_heads, dropout=dropout
        )
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, model_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        token_states: torch.Tensor,
        allowed: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> torch.Tensor:
        valid = valid_tokens[:, :, None]
        attention_update = self.attention(
            self.attention_norm(token_states), allowed
        )
        token_states = torch.where(
            valid,
            token_states + self.dropout(attention_update),
            torch.zeros_like(token_states),
        )
        ffn_update = self.ffn(self.ffn_norm(token_states))
        return torch.where(
            valid,
            token_states + self.dropout(ffn_update),
            torch.zeros_like(token_states),
        )


class BlockSparseWedgeFlowTransformerRanker(nn.Module):
    """Minimal jointly trained Transformer/LM ranking core.

    Input states are lexical/token representations, not offline ranking
    scores.  Query tokens, flattened candidate segments, and ordered history
    events are processed together under ``build_information_barrier_mask``.
    The base head reads only candidate states protected by that barrier.  The
    Hodge wedge consumes the resulting query/candidate/history states and is
    the sole history-to-final-logit path.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        *,
        num_heads: int,
        num_layers: int,
        flow_dim: int,
        max_history_positions: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        score_delta_max: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or model_dim <= 0:
            raise ValueError("input_dim and model_dim must be positive")
        if num_heads <= 0 or num_layers <= 0:
            raise ValueError("num_heads and num_layers must be positive")
        if max_history_positions <= 0:
            raise ValueError("max_history_positions must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")
        if model_dim % num_heads != 0:
            raise ValueError("model_dim must be divisible by num_heads")

        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim)
        self.max_history_positions = int(max_history_positions)
        hidden_dim = int(ffn_dim or (4 * model_dim))
        if hidden_dim <= 0:
            raise ValueError("ffn_dim must be positive")

        self.input_projection = nn.Linear(input_dim, model_dim)
        self.role_embedding = nn.Embedding(4, model_dim, padding_idx=PAD)
        # Only history receives a global/order position.  Candidate segments
        # deliberately receive no candidate-index or item-ID embedding.
        self.history_order_embedding = nn.Embedding(
            max_history_positions, model_dim
        )
        self.layers = nn.ModuleList(
            [
                _BarrierTransformerLayer(
                    model_dim,
                    num_heads,
                    ffn_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.base_head = nn.Linear(model_dim, 1)
        self.wedge_flow = ConservativeWedgeFlowProbeRanker(
            model_dim,
            flow_dim,
            score_delta_max=score_delta_max,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.normal_(self.role_embedding.weight, std=0.02)
        with torch.no_grad():
            self.role_embedding.weight[PAD].zero_()
        nn.init.normal_(self.history_order_embedding.weight, std=0.02)
        nn.init.xavier_uniform_(self.base_head.weight)
        nn.init.zeros_(self.base_head.bias)

    def forward(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_prior: torch.Tensor | None = None,
        *,
        flow_enabled: bool = True,
    ) -> TransformerCoreOutput:
        self._validate(
            query_tokens,
            candidate_tokens,
            history_tokens,
            query_mask,
            candidate_token_mask,
            history_mask,
            history_prior,
        )
        query_mask = query_mask.bool()
        candidate_token_mask = candidate_token_mask.bool()
        history_mask = history_mask.bool()
        batch, candidate_count, candidate_length, _ = candidate_tokens.shape
        query_length = query_tokens.shape[1]
        history_count = history_tokens.shape[1]

        safe_query = torch.where(
            query_mask[:, :, None], query_tokens, torch.zeros_like(query_tokens)
        )
        safe_candidates = torch.where(
            candidate_token_mask[:, :, :, None],
            candidate_tokens,
            torch.zeros_like(candidate_tokens),
        )
        safe_history = torch.where(
            history_mask[:, :, None],
            history_tokens,
            torch.zeros_like(history_tokens),
        )
        flat_candidates = safe_candidates.reshape(
            batch, candidate_count * candidate_length, self.input_dim
        )
        concatenated = torch.cat(
            [safe_query, flat_candidates, safe_history], dim=1
        )

        token_roles, candidate_group_ids = self._make_layout(
            query_mask, candidate_token_mask, history_mask
        )
        valid_tokens = token_roles != PAD
        allowed = build_information_barrier_mask(
            token_roles, candidate_group_ids
        )
        token_states = self.input_projection(concatenated)
        token_states = token_states + self.role_embedding(token_roles)
        if history_count:
            history_positions = torch.arange(
                history_count, device=history_tokens.device
            )
            history_position_states = self.history_order_embedding(
                history_positions
            )[None, :, :]
            history_slice = slice(
                query_length + candidate_count * candidate_length, None
            )
            token_states = token_states.clone()
            token_states[:, history_slice, :] = (
                token_states[:, history_slice, :] + history_position_states
            )
        token_states = torch.where(
            valid_tokens[:, :, None], token_states, torch.zeros_like(token_states)
        )

        for layer in self.layers:
            token_states = layer(token_states, allowed, valid_tokens)
        token_states = torch.where(
            valid_tokens[:, :, None],
            self.final_norm(token_states),
            torch.zeros_like(token_states),
        )

        query_states = token_states[:, :query_length, :]
        candidate_start = query_length
        candidate_stop = candidate_start + candidate_count * candidate_length
        candidate_token_states = token_states[
            :, candidate_start:candidate_stop, :
        ].reshape(batch, candidate_count, candidate_length, self.model_dim)
        history_states = token_states[:, candidate_stop:, :]

        query_state = self._masked_mean(query_states, query_mask, dim=1)
        candidate_states = self._masked_mean(
            candidate_token_states, candidate_token_mask, dim=2
        )
        candidate_mask = candidate_token_mask.any(dim=-1)

        # The final base head is jointly trained and explicitly evaluated in
        # FP32.  It is not detached when combined with the wedge update.
        with torch.autocast(device_type=token_states.device.type, enabled=False):
            base_scores = self.base_head(candidate_states.float()).squeeze(-1)
            base_scores = torch.where(
                candidate_mask,
                base_scores,
                torch.zeros_like(base_scores),
            )
        if history_prior is None:
            history_prior = torch.ones(
                batch,
                history_count,
                device=history_tokens.device,
                dtype=torch.float32,
            )
        safe_prior = torch.where(
            history_mask,
            history_prior.float(),
            torch.zeros_like(history_prior, dtype=torch.float32),
        )
        wedge = self.wedge_flow(
            query=query_state,
            candidates=candidate_states,
            history=history_states,
            candidate_mask=candidate_mask,
            history_mask=history_mask,
            history_prior=safe_prior,
            base_scores=base_scores,
        )
        history_present = history_mask.any(dim=-1)
        active = candidate_mask & history_present[:, None]
        if flow_enabled:
            applied_delta = torch.where(
                active,
                wedge.conservative_score_delta,
                torch.zeros_like(base_scores),
            )
        else:
            applied_delta = torch.zeros_like(base_scores)
        # Do not use wedge.scores here: the cheap probe intentionally detaches
        # its external base, whereas this is the jointly trained LM core.
        scores = base_scores + applied_delta
        return TransformerCoreOutput(
            scores=scores,
            base_scores=base_scores,
            applied_score_delta=applied_delta,
            query_state=query_state,
            candidate_states=candidate_states,
            history_states=history_states,
            token_states=token_states,
            token_roles=token_roles,
            candidate_group_ids=candidate_group_ids,
            attention_mask=allowed,
            wedge=wedge,
        )

    @staticmethod
    def _masked_mean(
        values: torch.Tensor,
        mask: torch.Tensor,
        *,
        dim: int,
    ) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        count = weights.sum(dim=dim).clamp_min(1.0)
        return (values * weights).sum(dim=dim) / count

    @staticmethod
    def _make_layout(
        query_mask: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidate_count, candidate_length = candidate_token_mask.shape
        device = query_mask.device
        query_roles = torch.where(
            query_mask,
            torch.full_like(query_mask, QUERY, dtype=torch.long),
            torch.full_like(query_mask, PAD, dtype=torch.long),
        )
        candidate_roles = torch.where(
            candidate_token_mask,
            torch.full_like(candidate_token_mask, CANDIDATE, dtype=torch.long),
            torch.full_like(candidate_token_mask, PAD, dtype=torch.long),
        ).reshape(batch, candidate_count * candidate_length)
        history_roles = torch.where(
            history_mask,
            torch.full_like(history_mask, HISTORY, dtype=torch.long),
            torch.full_like(history_mask, PAD, dtype=torch.long),
        )
        token_roles = torch.cat(
            [query_roles, candidate_roles, history_roles], dim=1
        )

        query_groups = torch.full_like(query_roles, -1, dtype=torch.long)
        groups = torch.arange(
            candidate_count, device=device, dtype=torch.long
        )[None, :, None].expand(batch, candidate_count, candidate_length)
        groups = torch.where(
            candidate_token_mask,
            groups,
            torch.full_like(groups, -1),
        ).reshape(batch, candidate_count * candidate_length)
        history_groups = torch.full_like(history_roles, -1, dtype=torch.long)
        candidate_group_ids = torch.cat(
            [query_groups, groups, history_groups], dim=1
        )
        return token_roles, candidate_group_ids

    def _validate(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        history_tokens: torch.Tensor,
        query_mask: torch.Tensor,
        candidate_token_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_prior: torch.Tensor | None,
    ) -> None:
        if query_tokens.ndim != 3 or query_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"query_tokens must have shape [B, Q, {self.input_dim}]"
            )
        if candidate_tokens.ndim != 4 or candidate_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                "candidate_tokens must have shape "
                f"[B, C, T, {self.input_dim}]"
            )
        if history_tokens.ndim != 3 or history_tokens.shape[-1] != self.input_dim:
            raise ValueError(
                f"history_tokens must have shape [B, H, {self.input_dim}]"
            )
        batch, query_length, _ = query_tokens.shape
        candidate_batch, candidate_count, candidate_length, _ = (
            candidate_tokens.shape
        )
        history_batch, history_count, _ = history_tokens.shape
        if candidate_count == 0 or candidate_length == 0:
            raise ValueError("at least one candidate token column is required")
        if candidate_batch != batch or history_batch != batch:
            raise ValueError("batch dimensions must match")
        if query_mask.shape != (batch, query_length):
            raise ValueError("query_mask shape mismatch")
        if candidate_token_mask.shape != (
            batch,
            candidate_count,
            candidate_length,
        ):
            raise ValueError("candidate_token_mask shape mismatch")
        if history_mask.shape != (batch, history_count):
            raise ValueError("history_mask shape mismatch")
        if history_prior is not None and history_prior.shape != (
            batch,
            history_count,
        ):
            raise ValueError("history_prior shape mismatch")
        if history_count > self.max_history_positions:
            raise ValueError("history length exceeds max_history_positions")
        q_mask = query_mask.bool()
        c_mask = candidate_token_mask.bool()
        h_mask = history_mask.bool()
        if not bool(q_mask.any(dim=-1).all().item()):
            raise ValueError("every request needs at least one query token")
        if not bool(c_mask.any(dim=(-1, -2)).all().item()):
            raise ValueError("every request needs at least one candidate token")
        if not bool(torch.isfinite(query_tokens[q_mask]).all().item()):
            raise ValueError("valid query tokens contain non-finite values")
        if not bool(torch.isfinite(candidate_tokens[c_mask]).all().item()):
            raise ValueError("valid candidate tokens contain non-finite values")
        if h_mask.any() and not bool(
            torch.isfinite(history_tokens[h_mask]).all().item()
        ):
            raise ValueError("valid history tokens contain non-finite values")
        if history_prior is not None and h_mask.any():
            valid_prior = history_prior[h_mask]
            if not bool(torch.isfinite(valid_prior).all().item()):
                raise ValueError("valid history prior contains non-finite values")
            if not bool(((valid_prior > 0) & (valid_prior <= 1)).all().item()):
                raise ValueError("valid history prior must lie in (0, 1]")
