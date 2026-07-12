"""Minimal CPU prototype for the C07 pairwise signed-kernel Transformer.

This module intentionally has no dataset or file-I/O code.  It consumes only
query, history, candidate, identity-match, and availability tensors.  In
particular, there is no label/qrels argument or path.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def odd_soft_threshold(x: Tensor, threshold: Tensor | float) -> Tensor:
    """Odd soft-threshold: sign(x) * max(abs(x) - threshold, 0).

    The ReLU-difference form avoids differentiating through ``torch.sign``.
    It is continuous and differentiable except at +/- ``threshold``.
    """

    return F.relu(x - threshold) - F.relu(-x - threshold)


@dataclass(frozen=True)
class SignedKernelResult:
    """Inspectable output of the candidate-competition normalizer."""

    weights: Tensor
    balances: Tensor
    pairwise_margins: Tensor


class PairwiseSignedKernel(nn.Module):
    """Candidate-axis signed attention with pairwise dead-zone abstention.

    Given evidence logits ``scores[b, candidate, history]``, the operator is

      d[i,k,j] = scores[i,j] - scores[k,j]
      u[i,j]   = mean_{k != i} soft_threshold(d[i,k,j], threshold)
      a[i,j]   = u[i,j] / (null_mass + sum_{r,t} abs(u[r,t])).

    Oddness makes ``sum_i a[i,j] == 0`` for every history event.  When every
    candidate margin for an event lies in the dead zone, that event attends to
    no candidate.  A global L1 denominator bounds total signed mass below one.
    """

    def __init__(self, threshold: float = 0.5, null_mass: float = 1.0) -> None:
        super().__init__()
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if null_mass <= 0:
            raise ValueError("null_mass must be positive")
        self.register_buffer("threshold", torch.tensor(float(threshold)))
        self.register_buffer("null_mass", torch.tensor(float(null_mass)))

    def forward(
        self,
        scores: Tensor,
        history_mask: Tensor,
        candidate_mask: Optional[Tensor] = None,
    ) -> SignedKernelResult:
        if scores.ndim != 3:
            raise ValueError("scores must have shape [batch, candidates, history]")
        batch, candidate_count, history_count = scores.shape
        if history_mask.shape != (batch, history_count):
            raise ValueError("history_mask must have shape [batch, history]")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                (batch, candidate_count), dtype=torch.bool, device=scores.device
            )
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask must have shape [batch, candidates]")

        history_mask = history_mask.to(dtype=torch.bool, device=scores.device)
        candidate_mask = candidate_mask.to(dtype=torch.bool, device=scores.device)

        # [B, C(query candidate), C(competing candidate), H]
        margins = scores[:, :, None, :] - scores[:, None, :, :]
        eye = torch.eye(candidate_count, dtype=torch.bool, device=scores.device)
        valid_pair = (
            candidate_mask[:, :, None, None]
            & candidate_mask[:, None, :, None]
            & ~eye[None, :, :, None]
            & history_mask[:, None, None, :]
        )
        shrunk = odd_soft_threshold(margins, self.threshold.to(scores.dtype))
        shrunk = torch.where(valid_pair, shrunk, torch.zeros_like(shrunk))

        competitor_count = (candidate_mask.sum(dim=1) - 1).clamp_min(1)
        balances = shrunk.sum(dim=2) / competitor_count[:, None, None].to(scores.dtype)
        valid_cell = candidate_mask[:, :, None] & history_mask[:, None, :]
        balances = torch.where(valid_cell, balances, torch.zeros_like(balances))

        l1_mass = balances.abs().sum(dim=(1, 2), keepdim=True)
        denominator = self.null_mass.to(scores.dtype) + l1_mass
        weights = balances / denominator
        return SignedKernelResult(
            weights=weights,
            balances=balances,
            pairwise_margins=margins,
        )


@dataclass(frozen=True)
class RankerOutput:
    """Outputs needed to audit the end-to-end prototype."""

    logits: Tensor
    base_logits: Tensor
    personalized_delta: Tensor
    evidence_logits: Tensor
    signed_weights: Tensor
    signed_balances: Tensor


class SignedKernelTransformer(nn.Module):
    """Tiny end-to-end Transformer ranker with a signed attention kernel.

    A shared Transformer contextualizes ``[query, history, candidates]`` under
    a structural information-flow mask:

    * query attends only to query;
    * history attends to query and history;
    * candidates attend to query and candidates;
    * the signed kernel is the sole history-to-candidate path.

    The prototype accepts already embedded tokens.  A production LM would
    replace those inputs with its token/item embedding stack while retaining
    the same mask and normalization contract.
    """

    def __init__(
        self,
        d_model: int = 12,
        nhead: int = 3,
        num_layers: int = 1,
        dim_feedforward: int = 24,
        threshold: float = 0.5,
        null_mass: float = 1.0,
        exact_match_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")
        if exact_match_scale <= 0:
            raise ValueError("exact_match_scale must be positive")

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.type_embedding = nn.Parameter(torch.zeros(3, d_model))
        nn.init.normal_(self.type_embedding, mean=0.0, std=0.02)

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.candidate_projection = nn.Linear(d_model, d_model, bias=False)
        self.history_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.final_norm = nn.LayerNorm(d_model)
        self.score_head = nn.Linear(d_model, 1, bias=False)

        raw_scale = math.log(math.expm1(exact_match_scale))
        self.raw_exact_match_scale = nn.Parameter(torch.tensor(raw_scale))
        self.signed_kernel = PairwiseSignedKernel(threshold, null_mass)
        self.d_model = d_model

    @property
    def exact_match_scale(self) -> Tensor:
        return F.softplus(self.raw_exact_match_scale)

    @staticmethod
    def _flow_mask(history_count: int, candidate_count: int, device: torch.device) -> Tensor:
        length = 1 + history_count + candidate_count
        blocked = torch.zeros((length, length), dtype=torch.bool, device=device)
        history = slice(1, 1 + history_count)
        candidates = slice(1 + history_count, length)

        # Query state is a stable request anchor; neither evidence family can
        # write back into it and create a hidden history-to-base path.
        blocked[0, 1:] = True
        # Candidate and history groups share a Transformer but communicate only
        # through query plus the explicit signed kernel below.
        blocked[history, candidates] = True
        blocked[candidates, history] = True
        return blocked

    def forward(
        self,
        query: Tensor,
        history: Tensor,
        candidates: Tensor,
        history_mask: Tensor,
        exact_match: Optional[Tensor] = None,
        candidate_mask: Optional[Tensor] = None,
    ) -> RankerOutput:
        if query.ndim != 2:
            raise ValueError("query must have shape [batch, d_model]")
        if history.ndim != 3 or candidates.ndim != 3:
            raise ValueError("history/candidates must be rank-3 tensors")
        batch, history_count, width = history.shape
        candidate_batch, candidate_count, candidate_width = candidates.shape
        if query.shape != (batch, self.d_model):
            raise ValueError("query width or batch does not match the model")
        if candidate_batch != batch or width != self.d_model or candidate_width != self.d_model:
            raise ValueError("history/candidate width or batch does not match the model")
        if history_mask.shape != (batch, history_count):
            raise ValueError("history_mask must have shape [batch, history]")
        history_mask = history_mask.to(dtype=torch.bool, device=query.device)

        if candidate_mask is None:
            candidate_mask = torch.ones(
                (batch, candidate_count), dtype=torch.bool, device=query.device
            )
        else:
            candidate_mask = candidate_mask.to(dtype=torch.bool, device=query.device)
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask must have shape [batch, candidates]")

        if exact_match is None:
            exact_match = torch.zeros(
                (batch, candidate_count, history_count),
                dtype=query.dtype,
                device=query.device,
            )
        if exact_match.shape != (batch, candidate_count, history_count):
            raise ValueError("exact_match must have shape [batch, candidates, history]")
        exact_match = exact_match.to(dtype=query.dtype, device=query.device)

        typed_query = query[:, None, :] + self.type_embedding[0]
        typed_history = history + self.type_embedding[1]
        typed_candidates = candidates + self.type_embedding[2]
        sequence = torch.cat((typed_query, typed_history, typed_candidates), dim=1)

        padding_mask = torch.cat(
            (
                torch.zeros((batch, 1), dtype=torch.bool, device=query.device),
                ~history_mask,
                ~candidate_mask,
            ),
            dim=1,
        )
        contextual = self.encoder(
            sequence,
            mask=self._flow_mask(history_count, candidate_count, query.device),
            src_key_padding_mask=padding_mask,
        )
        query_state = contextual[:, 0, :]
        history_state = contextual[:, 1 : 1 + history_count, :]
        candidate_state = contextual[:, 1 + history_count :, :]

        q = torch.tanh(self.query_projection(query_state))
        c = self.candidate_projection(candidate_state)
        h = self.history_projection(history_state)
        evidence_logits = torch.einsum("bcd,bhd,bd->bch", c, h, q)
        evidence_logits = evidence_logits / math.sqrt(self.d_model)
        query_candidate_support = torch.einsum("bcd,bd->bc", c, q)
        query_candidate_support = F.softplus(
            query_candidate_support / math.sqrt(self.d_model)
        )
        valid_evidence = candidate_mask[:, :, None] & history_mask[:, None, :]
        masked_exact_match = torch.where(
            valid_evidence, exact_match, torch.zeros_like(exact_match)
        )
        evidence_logits = evidence_logits + (
            self.exact_match_scale
            * masked_exact_match
            * query_candidate_support[:, :, None]
        )
        evidence_logits = torch.where(
            valid_evidence, evidence_logits, torch.zeros_like(evidence_logits)
        )

        kernel = self.signed_kernel(evidence_logits, history_mask, candidate_mask)
        history_values = self.value_projection(history_state)
        personalized_delta = torch.einsum("bch,bhd->bcd", kernel.weights, history_values)
        personalized_delta = self.output_projection(personalized_delta)

        base_state = self.final_norm(candidate_state)
        ranked_state = self.final_norm(candidate_state + personalized_delta)
        base_logits = self.score_head(base_state).squeeze(-1)
        logits = self.score_head(ranked_state).squeeze(-1)
        return RankerOutput(
            logits=logits,
            base_logits=base_logits,
            personalized_delta=personalized_delta,
            evidence_logits=evidence_logits,
            signed_weights=kernel.weights,
            signed_balances=kernel.balances,
        )
