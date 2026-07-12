"""Candidate-local C03 Transformer and partial-transport primitive.

The module deliberately has no data-path or evaluator dependencies.  In
particular, it cannot open records, qrels, run outputs, or manifests.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Operator = Literal["cycle_null", "softmax", "no_null", "no_cycle", "mean_pool"]


def _log_sinkhorn_iterations(
    log_scores: Tensor,
    log_mu: Tensor,
    log_nu: Tensor,
    iterations: int,
) -> Tensor:
    """Return a log transport plan with the requested log marginals."""

    if log_scores.ndim != 2:
        raise ValueError("log_scores must be a matrix")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(iterations):
        u = log_mu - torch.logsumexp(log_scores + v.unsqueeze(0), dim=1)
        v = log_nu - torch.logsumexp(log_scores + u.unsqueeze(1), dim=0)
    # Finish with a row update and a final column update.  This makes both
    # marginal errors small at finite iteration count without detaching.
    u = log_mu - torch.logsumexp(log_scores + v.unsqueeze(0), dim=1)
    v = log_nu - torch.logsumexp(log_scores + u.unsqueeze(1), dim=0)
    return log_scores + u.unsqueeze(1) + v.unsqueeze(0)


def dustbin_marginals(
    rows: int,
    cols: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """SuperGlue-style partial-assignment marginals for an augmented plan."""

    if rows < 1 or cols < 1:
        raise ValueError("real transport dimensions must be positive")
    norm = -math.log(rows + cols)
    log_mu = torch.full((rows + 1,), norm, device=device, dtype=dtype)
    log_nu = torch.full((cols + 1,), norm, device=device, dtype=dtype)
    log_mu[-1] = math.log(cols) + norm
    log_nu[-1] = math.log(rows) + norm
    return log_mu, log_nu


def dustbin_transport(
    real_scores: Tensor,
    bin_score: Tensor | float,
    *,
    iterations: int = 20,
    temperature: float = 1.0,
) -> Tensor:
    """Compute a differentiable entropy-regularized partial assignment.

    ``real_scores`` has shape ``[M, N]``.  The returned plan has shape
    ``[M + 1, N + 1]``; the final row and column are learned dustbins.
    """

    if real_scores.ndim != 2 or min(real_scores.shape) < 1:
        raise ValueError("real_scores must have shape [M,N] with M,N >= 1")
    if not math.isfinite(float(temperature)) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    rows, cols = real_scores.shape
    if not isinstance(bin_score, Tensor):
        bin_score = real_scores.new_tensor(float(bin_score))
    bin_score = bin_score.to(device=real_scores.device, dtype=real_scores.dtype)

    # Every C03 pair has a singleton query or candidate side.  For that common
    # case the Sinkhorn scaling equations reduce to one smooth scalar root; a
    # Newton solve is both faster and substantially more accurate than hundreds
    # of generic matrix-scaling iterations.  It is the same entropic OT
    # optimum, not a different normalization.
    if rows == 1:
        return _singleton_dustbin_transport(
            real_scores,
            bin_score,
            iterations=iterations,
            temperature=temperature,
        )
    if cols == 1:
        return _singleton_dustbin_transport(
            real_scores.transpose(0, 1),
            bin_score,
            iterations=iterations,
            temperature=temperature,
        ).transpose(0, 1)

    augmented = bin_score.expand(rows + 1, cols + 1).clone()
    augmented[:rows, :cols] = real_scores
    log_mu, log_nu = dustbin_marginals(
        rows,
        cols,
        device=real_scores.device,
        dtype=real_scores.dtype,
    )
    log_plan = _log_sinkhorn_iterations(
        augmented / temperature,
        log_mu,
        log_nu,
        iterations,
    )
    return log_plan.exp()


def _singleton_dustbin_transport(
    real_scores: Tensor,
    bin_score: Tensor,
    *,
    iterations: int,
    temperature: float,
) -> Tensor:
    """Exact-marginal solver for a 1-by-N SuperGlue transport problem.

    If ``y_j`` is the real-real mass scaled by ``N+1`` and
    ``S=sum_j y_j``, stationarity gives

    ``y_j = sigmoid((score_j-bin)/temperature - logit(S))``.

    Newton iterations solve the single monotone root while retaining autograd
    through the scores, bin, and temperature-scaled logits.
    """

    count = real_scores.shape[1]
    logits = (real_scores[0] - bin_score) / temperature
    tiny = max(torch.finfo(real_scores.dtype).eps * 16.0, 1e-12)
    total = torch.sigmoid(logits).mean().clamp(min=0.05, max=0.95)
    # Twelve iterations are ample for the tiny, monotone scalar problem; honor
    # a larger configured count but avoid silently weakening a smaller one.
    for _ in range(max(iterations, 12)):
        total_safe = total.clamp(min=tiny, max=1.0 - tiny)
        logit_total = torch.log(total_safe) - torch.log1p(-total_safe)
        real_mass = torch.sigmoid(logits - logit_total)
        function = real_mass.sum() - total_safe
        derivative = -(
            (real_mass * (1.0 - real_mass)).sum()
            / (total_safe * (1.0 - total_safe)).clamp_min(tiny)
        ) - 1.0
        total = (total_safe - function / derivative).clamp(min=tiny, max=1.0 - tiny)

    logit_total = torch.log(total) - torch.log1p(-total)
    real_mass = torch.sigmoid(logits - logit_total)
    # One differentiable correction makes total and sum(real_mass) agree to
    # machine precision without changing the stationary point materially.
    total = real_mass.sum().clamp(min=tiny, max=1.0 - tiny)
    normalizer = float(count + 1)
    plan = real_scores.new_empty((2, count + 1))
    plan[0, :count] = real_mass / normalizer
    plan[1, :count] = (1.0 - real_mass) / normalizer
    plan[0, count] = (1.0 - total) / normalizer
    plan[1, count] = total / normalizer
    return plan


def plan_marginal_error(plan: Tensor) -> Tensor:
    """Maximum absolute error against the matching dustbin marginals."""

    rows = plan.shape[0] - 1
    cols = plan.shape[1] - 1
    log_mu, log_nu = dustbin_marginals(
        rows,
        cols,
        device=plan.device,
        dtype=plan.dtype,
    )
    row_error = (plan.sum(dim=1) - log_mu.exp()).abs().max()
    col_error = (plan.sum(dim=0) - log_nu.exp()).abs().max()
    return torch.maximum(row_error, col_error)


class TriadicTransportRanker(nn.Module):
    """Compact Transformer ranker with candidate-anchored transport.

    Inputs are frozen-LM representations.  The local Transformer, pairwise
    projections, learned dustbins, transport bottleneck, and ranking update are
    jointly trainable.  Every operator variant uses the same parameter tensors;
    variants differ only in the normalization/information-flow rule.
    """

    OPERATORS: tuple[Operator, ...] = (
        "cycle_null",
        "softmax",
        "no_null",
        "no_cycle",
        "mean_pool",
    )

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        ff_dim: int,
        max_history: int,
        transport_dim: int,
        sinkhorn_iterations: int,
        sinkhorn_temperature: float,
        cycle_lambda: float,
        identity_bonus_floor: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if max_history < 1:
            raise ValueError("max_history must be positive")
        self.max_history = max_history
        self.sinkhorn_iterations = sinkhorn_iterations
        self.sinkhorn_temperature = sinkhorn_temperature
        self.cycle_lambda = cycle_lambda
        self.identity_bonus_floor = identity_bonus_floor
        self.eps = 1e-8

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.role_embedding = nn.Embedding(3, hidden_dim)
        self.position_embedding = nn.Embedding(max_history + 2, hidden_dim)
        # 0=unknown/pad, 1=click, 2=purchase.
        self.event_embedding = nn.Embedding(3, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.interaction_transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)

        self.qh_query = nn.Linear(hidden_dim, transport_dim, bias=False)
        self.qh_history = nn.Linear(hidden_dim, transport_dim, bias=False)
        self.hc_history = nn.Linear(hidden_dim, transport_dim, bias=False)
        self.hc_candidate = nn.Linear(hidden_dim, transport_dim, bias=False)
        self.qc_query = nn.Linear(hidden_dim, transport_dim, bias=False)
        self.qc_candidate = nn.Linear(hidden_dim, transport_dim, bias=False)

        self.qh_bin_score = nn.Parameter(torch.tensor(0.0))
        self.hc_bin_score = nn.Parameter(torch.tensor(0.0))
        self.qc_bin_score = nn.Parameter(torch.tensor(0.0))
        self.raw_identity_bonus = nn.Parameter(torch.tensor(0.0))

        self.history_update = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rank_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.rank_candidate = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mass_bias = nn.Parameter(torch.tensor(0.0))

    @property
    def identity_bonus(self) -> Tensor:
        return self.raw_identity_bonus.new_tensor(self.identity_bonus_floor) + F.softplus(
            self.raw_identity_bonus
        )

    def forward(
        self,
        query: Tensor,
        candidate: Tensor,
        history: Tensor,
        history_mask: Tensor,
        exact_match: Tensor,
        *,
        event_types: Tensor | None = None,
        operator: Operator = "cycle_null",
    ) -> dict[str, Tensor]:
        if operator not in self.OPERATORS:
            raise ValueError(f"unknown operator: {operator}")
        self._validate_inputs(query, candidate, history, history_mask, exact_match)
        batch_size, history_slots, _ = history.shape
        if event_types is None:
            event_types = torch.zeros_like(history_mask, dtype=torch.long)
        if event_types.shape != history_mask.shape:
            raise ValueError("event_types must match history_mask")

        q0 = self.input_projection(query)
        h0 = self.input_projection(history)
        c0 = self.input_projection(candidate)

        q0 = q0 + self.role_embedding.weight[0]
        h0 = h0 + self.role_embedding.weight[1]
        c0 = c0 + self.role_embedding.weight[2]
        h0 = h0 + self.event_embedding(event_types.clamp(min=0, max=2))

        positions = torch.arange(
            history_slots + 2,
            device=query.device,
            dtype=torch.long,
        )
        sequence = torch.cat((q0.unsqueeze(1), h0, c0.unsqueeze(1)), dim=1)
        sequence = sequence + self.position_embedding(positions).unsqueeze(0)
        padding_mask = torch.cat(
            (
                torch.zeros((batch_size, 1), dtype=torch.bool, device=query.device),
                ~history_mask,
                torch.zeros((batch_size, 1), dtype=torch.bool, device=query.device),
            ),
            dim=1,
        )
        contextual = self.interaction_transformer(
            sequence,
            src_key_padding_mask=padding_mask,
        )
        contextual = self.output_norm(contextual)
        q = contextual[:, 0]
        h = contextual[:, 1 : history_slots + 1]
        c = contextual[:, history_slots + 1]

        scale = math.sqrt(self.qh_query.out_features)
        qh = torch.einsum("bd,bhd->bh", self.qh_query(q), self.qh_history(h)) / scale
        hc = torch.einsum("bhd,bd->bh", self.hc_history(h), self.hc_candidate(c)) / scale
        hc = hc + self.identity_bonus * exact_match.to(dtype=hc.dtype)
        qc = (self.qc_query(q) * self.qc_candidate(c)).sum(dim=-1) / scale

        event_weights = h.new_zeros((batch_size, history_slots))
        trusted_mass = h.new_zeros((batch_size,))
        null_mass = h.new_ones((batch_size,))
        cycle_gap = h.new_zeros((batch_size,))
        marginal_error = h.new_zeros((batch_size,))

        for batch_index in range(batch_size):
            valid_count = int(history_mask[batch_index].sum().item())
            if valid_count == 0:
                continue
            result = self._operator_mass(
                qh[batch_index, :valid_count],
                hc[batch_index, :valid_count],
                qc[batch_index],
                operator=operator,
            )
            event_weights[batch_index, :valid_count] = result["weights"]
            trusted_mass[batch_index] = result["trusted_mass"]
            null_mass[batch_index] = result["null_mass"]
            cycle_gap[batch_index] = result["cycle_gap"]
            marginal_error[batch_index] = result["marginal_error"]

        history_summary = torch.einsum("bh,bhd->bd", event_weights, h)
        candidate_plus = c + trusted_mass.unsqueeze(-1) * self.history_update(history_summary)
        rank_q = self.rank_query(q)
        base_logit = (rank_q * self.rank_candidate(c)).sum(dim=-1) / math.sqrt(q.shape[-1])
        history_logit = (
            rank_q * self.rank_candidate(candidate_plus)
        ).sum(dim=-1) / math.sqrt(q.shape[-1])
        raw_residual = trusted_mass * (
            history_logit - base_logit + F.softplus(self.mass_bias)
        )
        has_history = history_mask.any(dim=1)
        # Algebraic no-history contract: exact zeros, not merely small values.
        raw_residual = torch.where(has_history, raw_residual, torch.zeros_like(raw_residual))
        trusted_mass = torch.where(has_history, trusted_mass, torch.zeros_like(trusted_mass))
        null_mass = torch.where(has_history, null_mass, torch.ones_like(null_mass))

        return {
            "base_logit": base_logit,
            "history_logit": history_logit,
            "raw_residual": raw_residual,
            "trusted_mass": trusted_mass,
            "null_mass": null_mass,
            "cycle_gap": cycle_gap,
            "event_weights": event_weights,
            "marginal_error": marginal_error,
            "qh_scores": qh,
            "hc_scores": hc,
            "qc_scores": qc,
        }

    def _operator_mass(
        self,
        qh_scores: Tensor,
        hc_scores: Tensor,
        qc_score: Tensor,
        *,
        operator: Operator,
    ) -> dict[str, Tensor]:
        eps = torch.finfo(qh_scores.dtype).eps
        zero = qh_scores.new_zeros(())
        if operator == "softmax":
            weights = torch.softmax(hc_scores / self.sinkhorn_temperature, dim=0)
            one = qh_scores.new_ones(())
            return self._mass_result(weights, one, zero, zero, zero)

        if operator == "mean_pool":
            weights = torch.full_like(qh_scores, 1.0 / qh_scores.numel())
            one = qh_scores.new_ones(())
            return self._mass_result(weights, one, zero, zero, zero)

        if operator == "no_null":
            a = torch.softmax(qh_scores / self.sinkhorn_temperature, dim=0)
            b = torch.softmax(hc_scores / self.sinkhorn_temperature, dim=0)
            direct = torch.sigmoid(qc_score / self.sinkhorn_temperature)
            overlap = torch.sqrt((a * b).clamp_min(0.0))
            trusted = (direct * overlap.sum()).clamp(min=0.0, max=1.0)
            weights = overlap / overlap.sum().clamp_min(eps)
            cycle = (a - b).abs().sum() / (a.sum() + b.sum()).clamp_min(eps)
            return self._mass_result(weights, trusted, zero, cycle, zero)

        hc_plan = dustbin_transport(
            hc_scores.unsqueeze(1),
            self.hc_bin_score,
            iterations=self.sinkhorn_iterations,
            temperature=self.sinkhorn_temperature,
        )
        count = hc_scores.numel()
        b = hc_plan[:count, 0] * (count + 1)
        hc_error = plan_marginal_error(hc_plan)
        if operator == "no_cycle":
            trusted = b.sum().clamp(min=0.0, max=1.0)
            weights = b / b.sum().clamp_min(eps)
            return self._mass_result(
                weights,
                trusted,
                (1.0 - trusted).clamp(min=0.0, max=1.0),
                zero,
                hc_error,
            )

        qh_plan = dustbin_transport(
            qh_scores.unsqueeze(0),
            self.qh_bin_score,
            iterations=self.sinkhorn_iterations,
            temperature=self.sinkhorn_temperature,
        )
        qc_plan = dustbin_transport(
            qc_score.reshape(1, 1),
            self.qc_bin_score,
            iterations=self.sinkhorn_iterations,
            temperature=self.sinkhorn_temperature,
        )
        a = qh_plan[0, :count] * (count + 1)
        direct = (qc_plan[0, 0] * 2.0).clamp(min=0.0, max=1.0)
        overlap = torch.sqrt((a * b).clamp_min(0.0))
        cycle = (a - b).abs().sum() / (a.sum() + b.sum()).clamp_min(eps)
        agreement = torch.exp(-self.cycle_lambda * cycle)
        event_mass = direct * agreement * overlap
        trusted = event_mass.sum().clamp(min=0.0, max=1.0)
        weights = event_mass / event_mass.sum().clamp_min(eps)
        error = torch.stack(
            (plan_marginal_error(qh_plan), hc_error, plan_marginal_error(qc_plan))
        ).max()
        return self._mass_result(
            weights,
            trusted,
            (1.0 - trusted).clamp(min=0.0, max=1.0),
            cycle,
            error,
        )

    @staticmethod
    def _mass_result(
        weights: Tensor,
        trusted_mass: Tensor,
        null_mass: Tensor,
        cycle_gap: Tensor,
        marginal_error: Tensor,
    ) -> dict[str, Tensor]:
        return {
            "weights": weights,
            "trusted_mass": trusted_mass,
            "null_mass": null_mass,
            "cycle_gap": cycle_gap,
            "marginal_error": marginal_error,
        }

    def _validate_inputs(
        self,
        query: Tensor,
        candidate: Tensor,
        history: Tensor,
        history_mask: Tensor,
        exact_match: Tensor,
    ) -> None:
        if query.ndim != 2 or candidate.shape != query.shape:
            raise ValueError("query and candidate must have matching [B,D] shapes")
        if history.ndim != 3 or history.shape[:2] != history_mask.shape:
            raise ValueError("history must be [B,H,D] and match history_mask")
        if history.shape[0] != query.shape[0] or history.shape[2] != query.shape[1]:
            raise ValueError("history batch/input dimensions must match query")
        if history.shape[1] > self.max_history:
            raise ValueError("history exceeds configured max_history")
        if history_mask.dtype != torch.bool:
            raise ValueError("history_mask must be bool")
        if exact_match.shape != history_mask.shape:
            raise ValueError("exact_match must match history_mask")
        if torch.any(exact_match & ~history_mask):
            raise ValueError("padding cannot be marked as an exact match")


def center_request_residual(raw_residual: Tensor, scale: float) -> Tensor:
    """Center a complete request's candidate residuals without variance scaling."""

    if raw_residual.ndim != 1:
        raise ValueError("a complete request residual must be one-dimensional")
    if raw_residual.numel() == 0:
        raise ValueError("cannot center an empty candidate set")
    return float(scale) * (raw_residual - raw_residual.mean())
