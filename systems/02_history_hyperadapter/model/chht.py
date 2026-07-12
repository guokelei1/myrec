"""Candidate-conditioned Cayley HyperAdapter Transformer for C02.

The personalization path changes an internal FFN output map.  With no history,
the event-composed skew kernel is exactly zero and the returned scores are the
provided non-personalized base scores bit-for-bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


Variant = Literal[
    "chht",
    "static_lora",
    "output_gate",
    "mean_history",
    "history_only",
]


@dataclass
class CHHTOutput:
    scores: torch.Tensor
    residual: torch.Tensor
    core: torch.Tensor
    core_norm: torch.Tensor
    delta_hidden: torch.Tensor
    cayley_orthogonality_error: torch.Tensor


def masked_zscore(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Within-row z-score with exact zeros in padded positions."""

    if values.shape != mask.shape:
        raise ValueError(f"values/mask mismatch: {values.shape} != {mask.shape}")
    weights = mask.to(values.dtype)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=-1, keepdim=True) / count
    centered = (values - mean) * weights
    variance = centered.square().sum(dim=-1, keepdim=True) / count
    return centered / variance.sqrt().clamp_min(eps)


def multi_positive_listwise_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    """Multi-positive listwise softmax loss, averaged by request."""

    valid = candidate_mask.bool()
    positive = labels.bool() & valid
    if not torch.all(positive.any(dim=-1)):
        raise ValueError("every training request must contain a positive candidate")
    neg_inf = -torch.finfo(scores.dtype).max
    all_lse = torch.logsumexp(scores.masked_fill(~valid, neg_inf), dim=-1)
    positive_lse = torch.logsumexp(scores.masked_fill(~positive, neg_inf), dim=-1)
    return (all_lse - positive_lse).mean()


class CHHTRanker(nn.Module):
    """Compact Transformer ranker with one candidate-specific internal update."""

    VALID_VARIANTS = {
        "chht",
        "static_lora",
        "output_gate",
        "mean_history",
        "history_only",
    }

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 96,
        heads: int = 4,
        ffn_dim: int = 192,
        rank: int = 8,
        history_layers: int = 1,
        pair_layers: int = 1,
        dropout: float = 0.1,
        max_history: int = 20,
        max_skew_norm: float = 0.35,
        max_score_residual: float = 1.5,
        variant: Variant = "chht",
    ) -> None:
        super().__init__()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        if not 1 <= rank <= hidden_dim:
            raise ValueError("rank must be in [1, hidden_dim]")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.ffn_dim = int(ffn_dim)
        self.rank = int(rank)
        self.max_history = int(max_history)
        self.max_skew_norm = float(max_skew_norm)
        self.max_score_residual = float(max_score_residual)
        self.variant = variant

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_type = nn.Parameter(torch.zeros(hidden_dim))
        self.candidate_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_type = nn.Parameter(torch.zeros(hidden_dim))
        self.history_position = nn.Embedding(max_history + 1, hidden_dim)

        history_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_encoder = nn.TransformerEncoder(
            history_layer,
            num_layers=history_layers,
            enable_nested_tensor=False,
        )
        pair_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_encoder = nn.TransformerEncoder(
            pair_layer,
            num_layers=pair_layers,
            enable_nested_tensor=False,
        )

        # The modulated map is the output projection of this final FFN.
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_in = nn.Linear(hidden_dim, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, hidden_dim, bias=False)

        # Triadic rank-space coordinates.  Separate left/right maps are what
        # permit a nonzero off-diagonal skew kernel.
        self.a_h = nn.Linear(hidden_dim, rank, bias=False)
        self.a_q = nn.Linear(hidden_dim, rank, bias=False)
        self.a_c = nn.Linear(hidden_dim, rank, bias=False)
        self.a_e = nn.Linear(3, rank, bias=False)
        self.b_h = nn.Linear(hidden_dim, rank, bias=False)
        self.b_q = nn.Linear(hidden_dim, rank, bias=False)
        self.b_c = nn.Linear(hidden_dim, rank, bias=False)
        self.b_e = nn.Linear(3, rank, bias=False)
        self.rho_h = nn.Linear(hidden_dim, rank, bias=False)
        self.rho_q = nn.Linear(hidden_dim, rank, bias=False)
        self.rho_c = nn.Linear(hidden_dim, rank, bias=False)
        self.rho_feature = nn.Parameter(torch.tensor([0.1, 0.1, 0.5]))

        self.rotation_basis = nn.Parameter(torch.empty(hidden_dim, rank))
        nn.init.orthogonal_(self.rotation_basis)
        self.score_head = nn.Linear(hidden_dim, 1, bias=False)

        # Nearest-neighbor controls.  They remain small relative to the shared
        # Transformer backbone, and every instantiated variant has the same
        # total parameter count for an auditable capacity envelope.
        self.static_down = nn.Linear(hidden_dim, rank, bias=False)
        self.static_up = nn.Linear(rank, hidden_dim, bias=False)
        self.output_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.output_gate = nn.Linear(hidden_dim, 1)
        self.output_value = nn.Linear(hidden_dim, 1, bias=False)
        self.mean_history_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_history_head = nn.Linear(hidden_dim, 1, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.query_type, std=0.02)
        nn.init.normal_(self.candidate_type, std=0.02)
        nn.init.normal_(self.history_type, std=0.02)
        nn.init.normal_(self.history_position.weight, std=0.02)
        nn.init.normal_(self.score_head.weight, std=0.02)
        nn.init.normal_(self.output_value.weight, std=0.02)
        nn.init.normal_(self.mean_history_head.weight, std=0.02)
        nn.init.normal_(self.static_down.weight, std=0.02)
        nn.init.zeros_(self.static_up.weight)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        history_embeddings: torch.Tensor,
        base_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_event_weight: torch.Tensor,
        repeat_mask: torch.Tensor,
        *,
        variant: Variant | None = None,
    ) -> CHHTOutput:
        variant = variant or self.variant
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown variant: {variant}")
        self._validate_shapes(
            query_embeddings,
            candidate_embeddings,
            history_embeddings,
            base_scores,
            candidate_mask,
            history_mask,
            history_event_weight,
            repeat_mask,
        )
        batch, candidates, _ = candidate_embeddings.shape
        history = history_embeddings.shape[1]
        candidate_mask = candidate_mask.bool()
        history_mask = history_mask.bool()
        history_present = history_mask.any(dim=-1)

        q = self.input_projection(query_embeddings.float()) + self.query_type
        c = self.input_projection(candidate_embeddings.float()) + self.candidate_type
        h = self.input_projection(history_embeddings.float()) + self.history_type

        positions = torch.arange(history, device=h.device).clamp_max(self.max_history)
        h = h + self.history_position(positions)[None, :, :]
        # The history-only control removes query content before contextualizing
        # events as well as from the generator projections; otherwise the
        # query token could leak through self-attention into h_context.
        history_query = torch.zeros_like(q) if variant == "history_only" else q
        context = torch.cat([history_query[:, None, :], h], dim=1)
        context_padding = torch.cat(
            [torch.zeros(batch, 1, dtype=torch.bool, device=h.device), ~history_mask],
            dim=1,
        )
        context = self.history_encoder(context, src_key_padding_mask=context_padding)
        q_context = context[:, 0, :]
        h_context = context[:, 1:, :]

        q_pair = q[:, None, :].expand(-1, candidates, -1)
        pair = torch.stack([q_pair, c], dim=2).reshape(
            batch * candidates, 2, self.hidden_dim
        )
        pair = self.pair_encoder(pair)
        candidate_state = pair[:, 1, :].reshape(batch, candidates, self.hidden_dim)
        ffn_hidden = F.gelu(self.ffn_in(self.ffn_norm(candidate_state)))
        base_ffn_output = self.ffn_out(ffn_hidden)

        # The geometric operator is always evaluated in fp32, including under
        # bf16 training autocast.  Otherwise the small r x r solve can lose the
        # orthogonality property that makes the intervention bounded.
        with torch.autocast(device_type=q_context.device.type, enabled=False):
            core = self._build_skew_core(
                q_context.float(),
                candidate_state.float(),
                h_context.float(),
                history_mask,
                history_event_weight.float(),
                repeat_mask,
                history_only=(variant == "history_only"),
            )
            core_norm = torch.linalg.matrix_norm(core, ord="fro", dim=(-2, -1))
            identity = torch.eye(self.rank, dtype=core.dtype, device=core.device)
            identity = identity.view(1, 1, self.rank, self.rank)
            identity_full = identity.expand(batch, candidates, -1, -1)
            cayley = torch.linalg.solve(identity_full + core, identity_full - core)
            orthogonal = cayley.transpose(-1, -2) @ cayley
            orthogonality_error = torch.linalg.matrix_norm(
                orthogonal - identity_full, ord="fro", dim=(-2, -1)
            )

        if variant in {"chht", "history_only"}:
            with torch.autocast(device_type=q_context.device.type, enabled=False):
                basis = torch.linalg.qr(self.rotation_basis.float(), mode="reduced").Q
                coefficients = torch.einsum(
                    "bcd,dr->bcr", base_ffn_output.float(), basis
                )
                rotated = torch.einsum("bcij,bcj->bci", cayley, coefficients)
                delta_hidden = torch.einsum(
                    "bcr,dr->bcd", rotated - coefficients, basis
                )
                raw_residual = self.score_head(delta_hidden).squeeze(-1)
        elif variant == "static_lora":
            delta_hidden = self.static_up(self.static_down(base_ffn_output))
            delta_hidden = delta_hidden * history_present[:, None, None]
            raw_residual = self.score_head(delta_hidden).squeeze(-1)
        elif variant == "output_gate":
            summary = self._candidate_history_summary(
                candidate_state, h_context, history_mask
            )
            output_feature = self.output_gate_mlp(
                torch.cat(
                    [
                        q_context[:, None, :].expand(-1, candidates, -1),
                        candidate_state,
                        summary,
                    ],
                    dim=-1,
                )
            )
            gate = torch.sigmoid(self.output_gate(output_feature)).squeeze(-1)
            raw_residual = gate * self.output_value(output_feature).squeeze(-1)
            delta_hidden = torch.zeros_like(base_ffn_output)
        else:  # mean_history
            weights = history_mask.to(h_context.dtype)
            denominator = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
            mean_history = (h_context * weights[:, :, None]).sum(dim=1) / denominator
            mean_feature = self.mean_history_mlp(
                torch.cat(
                    [
                        q_context[:, None, :].expand(-1, candidates, -1),
                        candidate_state,
                        mean_history[:, None, :].expand(-1, candidates, -1),
                    ],
                    dim=-1,
                )
            )
            raw_residual = self.mean_history_head(mean_feature).squeeze(-1)
            delta_hidden = torch.zeros_like(base_ffn_output)

        residual = self.max_score_residual * torch.tanh(raw_residual)
        residual = residual * history_present[:, None].to(residual.dtype)
        residual = residual * candidate_mask.to(residual.dtype)
        scores = base_scores + residual
        scores = torch.where(candidate_mask, scores, torch.full_like(scores, -1e9))
        return CHHTOutput(
            scores=scores,
            residual=residual,
            core=core,
            core_norm=core_norm,
            delta_hidden=delta_hidden,
            cayley_orthogonality_error=orthogonality_error,
        )

    def _build_skew_core(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        event_weight: torch.Tensor,
        repeat_mask: torch.Tensor,
        *,
        history_only: bool,
    ) -> torch.Tensor:
        batch, candidate_count, _ = candidates.shape
        history_count = history.shape[1]
        q = torch.zeros_like(query) if history_only else query
        c = torch.zeros_like(candidates) if history_only else candidates
        h = history

        q_a = self.a_q(q)[:, None, None, :]
        q_b = self.b_q(q)[:, None, None, :]
        c_a = self.a_c(c)[:, :, None, :]
        c_b = self.b_c(c)[:, :, None, :]
        h_a = self.a_h(h)[:, None, :, :]
        h_b = self.b_h(h)[:, None, :, :]

        if history_count:
            position = torch.linspace(
                0.0, 1.0, history_count, dtype=history.dtype, device=history.device
            )
            position = position.view(1, 1, history_count).expand(
                batch, candidate_count, -1
            )
        else:
            position = history.new_zeros(batch, candidate_count, 0)
        event = event_weight[:, None, :].expand(-1, candidate_count, -1)
        # The history-only control must not recover candidate identity through
        # the exact-recurrence feature.  It may still apply its shared history
        # update to different candidate activations, but its generated core is
        # a function of H alone.
        repeat = (
            torch.zeros_like(repeat_mask, dtype=history.dtype)
            if history_only
            else repeat_mask.to(history.dtype)
        )
        features = torch.stack([event, position, repeat], dim=-1)
        a = torch.tanh(h_a + q_a + c_a + self.a_e(features))
        b = torch.tanh(h_b + q_b + c_b + self.b_e(features))

        rho_h = self.rho_h(h)[:, None, :, :]
        rho_q = self.rho_q(q)[:, None, None, :]
        rho_c = self.rho_c(c)[:, :, None, :]
        triadic = (rho_h * rho_q * rho_c).sum(dim=-1) / math.sqrt(self.rank)
        feature_term = torch.einsum("bchf,f->bch", features, self.rho_feature)
        rho = torch.tanh(triadic + feature_term)
        valid = history_mask[:, None, :].expand(-1, candidate_count, -1)
        rho = rho * valid.to(rho.dtype)
        denominator = rho.abs().sum(dim=-1, keepdim=True).clamp_min(1.0)
        matrix = torch.einsum("bch,bchr,bchs->bcrs", rho, a, b)
        matrix = matrix / denominator[:, :, :, None]
        skew = 0.5 * (matrix - matrix.transpose(-1, -2))
        norm = torch.linalg.matrix_norm(skew, ord="fro", dim=(-2, -1))
        scale = torch.clamp(
            self.max_skew_norm / norm.clamp_min(1e-8), max=1.0
        )
        skew = skew * scale[:, :, None, None]
        skew = skew * valid.any(dim=-1)[:, :, None, None].to(skew.dtype)
        return skew

    @staticmethod
    def _candidate_history_summary(
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.einsum("bcd,bhd->bch", candidates, history)
        logits = logits / math.sqrt(candidates.shape[-1])
        logits = logits.masked_fill(~history_mask[:, None, :], -1e4)
        attention = torch.softmax(logits, dim=-1)
        attention = attention * history_mask[:, None, :].to(attention.dtype)
        attention = attention / attention.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return torch.einsum("bch,bhd->bcd", attention, history)

    @staticmethod
    def _validate_shapes(
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        base_scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        event_weight: torch.Tensor,
        repeat_mask: torch.Tensor,
    ) -> None:
        if query.ndim != 2 or candidates.ndim != 3 or history.ndim != 3:
            raise ValueError("query/candidate/history tensors have invalid rank")
        batch, candidate_count, dimension = candidates.shape
        if query.shape != (batch, dimension) or history.shape[0] != batch:
            raise ValueError("embedding batch/dimension mismatch")
        history_count = history.shape[1]
        expected_candidate = (batch, candidate_count)
        expected_history = (batch, history_count)
        if base_scores.shape != expected_candidate:
            raise ValueError("base score shape mismatch")
        if candidate_mask.shape != expected_candidate:
            raise ValueError("candidate mask shape mismatch")
        if history_mask.shape != expected_history or event_weight.shape != expected_history:
            raise ValueError("history mask/weight shape mismatch")
        if repeat_mask.shape != (batch, candidate_count, history_count):
            raise ValueError("repeat mask shape mismatch")
