"""History-conditioned candidate-local Hodge-trusted ranking flow.

The layer replaces an unconstrained personalization residual with an implicit
skew-symmetric candidate graph.  History can only transfer final-logit mass
between candidates; it cannot create a candidate-common score translation.
Candidate-local Hodge trust lets cycle energy attenuate the projected gradient
at its incident candidates, but never lets a cycle supply ranking direction.
The external base-score wrapper is for a cheap falsifier only.  In the final
system the same layer consumes jointly trained LM hidden states and lives in
the LM ranking head.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F


def _clamp_roundoff_nonnegative(
    values: torch.Tensor,
    magnitude: torch.Tensor,
    *,
    context: str,
    contraction_terms: int,
) -> torch.Tensor:
    """Clamp only FP64 roundoff; fail on a materially negative energy."""

    if values.dtype != torch.float64 or magnitude.shape != values.shape:
        raise ValueError("roundoff guard requires shape-matched FP64 tensors")
    if contraction_terms <= 0:
        raise ValueError("contraction_terms must be positive")
    tolerance = _roundoff_tolerance(magnitude, contraction_terms)
    if bool((values < -tolerance).any().item()):
        raise FloatingPointError(
            f"{context} is negative beyond the FP64 roundoff allowance"
        )
    return values.clamp_min(0.0)


def _roundoff_tolerance(
    magnitude: torch.Tensor, contraction_terms: int
) -> torch.Tensor:
    if magnitude.dtype != torch.float64 or contraction_terms <= 0:
        raise ValueError("roundoff tolerance requires FP64 magnitude and terms")
    finfo = torch.finfo(magnitude.dtype)
    return (
        32.0 * float(contraction_terms) * finfo.eps * magnitude.abs()
        + finfo.tiny
    )


def _cycle_identity_forward_error_bound(
    primitive_absolute_sum: torch.Tensor, contraction_terms: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an upper primitive scale and FP64 identity-comparison bound.

    `primitive_absolute_sum` expands the row energy before any sign
    cancellation.  The comparison composes four contraction families: the
    registered Gram path, the direct-dot magnitude path, the explicit
    squared-edge path, and construction of the absolute primitive bound.  Each
    receives the existing conservative `32*(C+r)` operation allowance.  The
    standard `gamma_k = k*u/(1-k*u)` model then bounds their composition.
    """

    if primitive_absolute_sum.dtype != torch.float64:
        raise ValueError("cycle identity error bound requires FP64 primitives")
    if contraction_terms <= 0:
        raise ValueError("cycle identity error bound requires positive terms")
    finfo = torch.finfo(primitive_absolute_sum.dtype)
    operation_allowance = 128.0 * float(contraction_terms)
    product = operation_allowance * finfo.eps
    if product >= 1.0:
        raise FloatingPointError("cycle identity FP64 error model is invalid")
    gamma = product / (1.0 - product)
    primitive_upper = primitive_absolute_sum.abs() / (1.0 - gamma)
    return primitive_upper, gamma * primitive_upper + finfo.tiny


def _repair_materially_negative_cycle_rows(
    candidate_cycle_energy: torch.Tensor,
    candidate_cycle_magnitude: torch.Tensor,
    centered_a: torch.Tensor,
    centered_b: torch.Tensor,
    active_candidate_event: torch.Tensor,
    no_cycle_subspace: torch.Tensor,
    *,
    alpha: float,
    contraction_terms: int,
    chunk_rows: int = 32,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Replace only invalid contracted rows by explicit FP64 squared edges.

    For an offending candidate/event row `(i,j)`, the exact residual edge is

      `C_ikj = alpha * (x_ij^T y_kj - y_ij^T x_kj)`,

    where `x,y` are candidate-centered factors.  Therefore
    `EC_ij = sum_k C_ikj^2` is a stable nonnegative expression requiring
    `O(C*r)` work for that row.  The explicit expression remains inside the
    autograd graph; only the discrete choice of rows is nondifferentiable.
    """

    if candidate_cycle_energy.dtype != torch.float64:
        raise ValueError("cycle repair requires FP64 contracted energies")
    if candidate_cycle_energy.shape != candidate_cycle_magnitude.shape:
        raise ValueError("cycle repair energy/magnitude shape mismatch")
    if centered_a.shape != centered_b.shape or centered_a.ndim != 4:
        raise ValueError("cycle repair factors must share [B,C,H,R]")
    if candidate_cycle_energy.shape != centered_a.shape[:3]:
        raise ValueError("cycle repair row/factor shape mismatch")
    if active_candidate_event.shape != candidate_cycle_energy.shape:
        raise ValueError("cycle repair active-mask shape mismatch")
    if no_cycle_subspace.shape != candidate_cycle_energy.shape:
        raise ValueError("cycle repair subspace-mask shape mismatch")
    if chunk_rows <= 0:
        raise ValueError("cycle repair chunk size must be positive")

    tolerance = _roundoff_tolerance(
        candidate_cycle_magnitude, contraction_terms
    )
    offending = (
        active_candidate_event
        & ~no_cycle_subspace
        & (candidate_cycle_energy < -tolerance)
    )
    rows = offending.nonzero(as_tuple=False)
    fallback_count = int(rows.shape[0])
    if fallback_count == 0:
        return candidate_cycle_energy, candidate_cycle_magnitude, 0

    repaired_energy = candidate_cycle_energy
    repaired_magnitude = candidate_cycle_magnitude
    for start in range(0, fallback_count, chunk_rows):
        selected = rows[start : start + chunk_rows]
        batch_index = selected[:, 0]
        candidate_index = selected[:, 1]
        history_index = selected[:, 2]
        row_a = centered_a[batch_index, candidate_index, history_index, :]
        row_b = centered_b[batch_index, candidate_index, history_index, :]
        peer_a = centered_a[batch_index, :, history_index, :]
        peer_b = centered_b[batch_index, :, history_index, :]
        forward = torch.einsum("mr,mcr->mc", row_a, peer_b)
        reverse = torch.einsum("mr,mcr->mc", row_b, peer_a)
        absolute_forward = torch.einsum(
            "mr,mcr->mc", row_a.abs(), peer_b.abs()
        )
        absolute_reverse = torch.einsum(
            "mr,mcr->mc", row_b.abs(), peer_a.abs()
        )
        explicit_edges = float(alpha) * (forward - reverse)
        explicit_energy = explicit_edges.square().sum(dim=-1)
        direct_magnitude = (float(alpha) ** 2) * (
            forward.square().sum(dim=-1)
            + reverse.square().sum(dim=-1)
            + 2.0 * (forward * reverse).sum(dim=-1).abs()
        )
        registered_magnitude = candidate_cycle_magnitude[
            batch_index, candidate_index, history_index
        ]
        registered_energy = candidate_cycle_energy[
            batch_index, candidate_index, history_index
        ]
        primitive_absolute_sum = (float(alpha) ** 2) * (
            absolute_forward + absolute_reverse
        ).square().sum(dim=-1)
        primitive_upper, consistency_tolerance = (
            _cycle_identity_forward_error_bound(
                primitive_absolute_sum, contraction_terms
            )
        )
        magnitude_consistent = (
            (registered_magnitude - direct_magnitude).abs()
            <= consistency_tolerance
        )
        energy_consistent = (
            (registered_energy - explicit_energy).abs()
            <= consistency_tolerance
        )
        energy_bounded_by_magnitude = (
            explicit_energy <= direct_magnitude + consistency_tolerance
        )
        components_bounded_by_primitives = (
            (direct_magnitude <= primitive_upper + consistency_tolerance)
            & (registered_magnitude <= primitive_upper + consistency_tolerance)
        )
        if not bool(
            (
                torch.isfinite(explicit_energy)
                & torch.isfinite(direct_magnitude)
                & magnitude_consistent
                & energy_consistent
                & energy_bounded_by_magnitude
                & components_bounded_by_primitives
            )
            .all()
            .item()
        ):
            raise FloatingPointError(
                "explicit FP64 cycle fallback failed independent magnitude consistency"
            )
        repaired_energy = repaired_energy.index_put(
            (batch_index, candidate_index, history_index), explicit_energy
        )
        # Keep the independently recomputed component magnitude, rather than
        # replacing it by the answer being validated. This prevents a bad Gram
        # index contraction from being silently hidden by the fallback.
        repaired_magnitude = repaired_magnitude.index_put(
            (batch_index, candidate_index, history_index), direct_magnitude
        )
    return repaired_energy, repaired_magnitude, fallback_count


@dataclass
class WedgeFlowOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    conservative_score_delta: torch.Tensor
    applied_score_delta: torch.Tensor
    divergence: torch.Tensor
    event_potential: torch.Tensor
    trusted_event_divergence: torch.Tensor
    candidate_event_trust: torch.Tensor
    hodge_candidate_trust: torch.Tensor
    candidate_gradient_energy: torch.Tensor
    candidate_cycle_energy: torch.Tensor
    flow_energy: torch.Tensor
    gradient_energy: torch.Tensor
    cycle_energy: torch.Tensor
    factor_a: torch.Tensor
    factor_b: torch.Tensor
    event_weights: torch.Tensor
    history_present: torch.Tensor
    residual_scale: torch.Tensor
    trust_mode: str
    cycle_energy_fallback_count: int


TrustMode = Literal[
    "local_hodge",
    "untrusted",
    "global_hodge",
    "direct_learned",
]


def trusted_gradient_divergence(
    event_potential: torch.Tensor,
    endpoint_trust: torch.Tensor,
    candidate_mask: torch.Tensor,
    history_mask: torch.Tensor,
) -> torch.Tensor:
    """Return divergence of `.5*t_i*t_k*(u_i-u_k)` without `C*C` edges."""

    if event_potential.shape != endpoint_trust.shape or event_potential.ndim != 3:
        raise ValueError("potential and trust must share shape [B, C, H]")
    batch, candidates, history = event_potential.shape
    if candidate_mask.shape != (batch, candidates):
        raise ValueError("candidate_mask shape mismatch")
    if history_mask.shape != (batch, history):
        raise ValueError("history_mask shape mismatch")
    candidate_mask = candidate_mask.bool()
    history_mask = history_mask.bool()
    active = candidate_mask[:, :, None] & history_mask[:, None, :]
    potential = torch.where(
        active, event_potential.double(), torch.zeros_like(event_potential.double())
    )
    trust = torch.where(
        active, endpoint_trust.double(), torch.zeros_like(endpoint_trust.double())
    )
    if not bool(torch.isfinite(trust[active]).all().item()):
        raise ValueError("endpoint trust contains non-finite values")
    if active.any() and not bool(
        ((trust[active] >= 0.0) & (trust[active] <= 1.0)).all().item()
    ):
        raise ValueError("endpoint trust must lie in [0, 1]")
    valid_count = candidate_mask.sum(dim=-1).clamp_min(1)
    count = valid_count[:, None, None].to(potential.dtype)
    trust_sum = trust.sum(dim=1)
    trusted_potential_sum = (trust * potential).sum(dim=1)
    divergence = trust / (2.0 * count) * (
        potential * trust_sum[:, None, :]
        - trusted_potential_sum[:, None, :]
    )
    divergence = torch.where(active, divergence, torch.zeros_like(divergence))
    numerical_mean = divergence.sum(dim=1, keepdim=True) / count
    divergence = torch.where(
        active, divergence - numerical_mean, torch.zeros_like(divergence)
    )
    return divergence.float()


def low_rank_hodge_calibration(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    candidate_mask: torch.Tensor,
    history_mask: torch.Tensor,
    *,
    diagnostics: dict[str, int] | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return candidate-local Hodge trust and trusted event divergence.

    For each event, ``u_i = n^-1 sum_k F_ik`` and the projected gradient is
    ``G_ik = u_i - u_k``.  The residual cycle field ``C = F - G`` supplies no
    direction.  It only lowers the endpoint trust

      ``t_i = EG_i / (EG_i + EC_i + 1e-12)``.

    The trusted edge is ``T_ik = .5*t_i*t_k*(u_i-u_k)`` and its divergence is
    evaluated implicitly.  Centered-factor Gram identities keep the operation
    ``O(B*H*C*R^2)``.  Means, energy moments, and conservation arithmetic use
    FP64 to prevent cancellation when the two wedge factors are nearly equal;
    all returned diagnostics are FP32.
    """

    if factor_a.shape != factor_b.shape or factor_a.ndim != 4:
        raise ValueError("factors must share shape [B, C, H, R]")
    batch, candidates, history, rank = factor_a.shape
    if candidate_mask.shape != (batch, candidates):
        raise ValueError("candidate_mask shape mismatch")
    if history_mask.shape != (batch, history):
        raise ValueError("history_mask shape mismatch")
    if rank == 0:
        raise ValueError("factor rank must be positive")

    # This cast is intentional even when the surrounding model is FP32.  The
    # centered EC expression subtracts nearly equal Gram terms for A ~= B.
    a = factor_a.double()
    b = factor_b.double()
    candidate_mask = candidate_mask.bool()
    history_mask = history_mask.bool()
    pair_mask = candidate_mask[:, :, None, None] & history_mask[:, None, :, None]
    a = torch.where(pair_mask, a, torch.zeros_like(a))
    b = torch.where(pair_mask, b, torch.zeros_like(b))
    valid_count = candidate_mask.sum(dim=-1).clamp_min(1)
    count = valid_count[:, None, None].to(a.dtype)
    mean_a = a.sum(dim=1) / count
    mean_b = b.sum(dim=1) / count
    alpha = 1.0 / float(2 * rank)
    event_potential = alpha * (
        a * mean_b[:, None, :, :] - b * mean_a[:, None, :, :]
    ).sum(dim=-1)
    active_candidate_event = (
        candidate_mask[:, :, None] & history_mask[:, None, :]
    )
    event_potential = torch.where(
        active_candidate_event,
        event_potential,
        torch.zeros_like(event_potential),
    )
    numerical_mean = event_potential.sum(dim=1, keepdim=True) / count
    event_potential = torch.where(
        active_candidate_event,
        event_potential - numerical_mean,
        torch.zeros_like(event_potential),
    )

    centered_a = torch.where(
        pair_mask, a - mean_a[:, None, :, :], torch.zeros_like(a)
    )
    centered_b = torch.where(
        pair_mask, b - mean_b[:, None, :, :], torch.zeros_like(b)
    )
    sum_potential_square = event_potential.square().sum(dim=1)
    candidate_gradient_energy = (
        valid_count[:, None, None].to(a.dtype) * event_potential.square()
        + sum_potential_square[:, None, :]
    )
    candidate_gradient_energy = torch.where(
        active_candidate_event,
        candidate_gradient_energy,
        torch.zeros_like(candidate_gradient_energy),
    )

    gram_aa = torch.einsum(
        "bchr,bchs->bhrs", centered_a, centered_a
    )
    gram_bb = torch.einsum(
        "bchr,bchs->bhrs", centered_b, centered_b
    )
    gram_ba = torch.einsum(
        "bchr,bchs->bhrs", centered_b, centered_a
    )
    cycle_a = torch.einsum(
        "bchr,bhrs,bchs->bch", centered_a, gram_bb, centered_a
    )
    cycle_b = torch.einsum(
        "bchr,bhrs,bchs->bch", centered_b, gram_aa, centered_b
    )
    cycle_cross = torch.einsum(
        "bchr,bhrs,bchs->bch", centered_a, gram_ba, centered_b
    )
    candidate_cycle_magnitude = (alpha * alpha) * (
        cycle_a.abs() + cycle_b.abs() + 2.0 * cycle_cross.abs()
    )
    candidate_cycle_energy = (alpha * alpha) * (
        cycle_a + cycle_b - 2.0 * cycle_cross
    )
    # A complete skew field on one or two valid candidates has no cycle
    # subspace: every edge is already a potential difference.  Evaluating the
    # three contracted Gram terms separately can nevertheless leave a tiny
    # negative cancellation residue after centering, especially for n=2.
    # Enforce this exact graph-theoretic identity before applying the guard;
    # this is not an evidence threshold and cannot hide a real cycle.
    no_cycle_subspace = valid_count[:, None, None] <= 2
    candidate_cycle_energy = torch.where(
        no_cycle_subspace,
        torch.zeros_like(candidate_cycle_energy),
        candidate_cycle_energy,
    )
    candidate_cycle_magnitude = torch.where(
        no_cycle_subspace,
        torch.zeros_like(candidate_cycle_magnitude),
        candidate_cycle_magnitude,
    )
    # A contracted difference of three large Gram terms can very rarely cross
    # its FP64 roundoff guard on GPU for quantized, nearly degenerate factors.
    # Only those invalid rows are recomputed as the mathematically identical
    # explicit sum of squared cycle edges. This is not an evidence threshold.
    candidate_cycle_energy, candidate_cycle_magnitude, fallback_count = (
        _repair_materially_negative_cycle_rows(
            candidate_cycle_energy,
            candidate_cycle_magnitude,
            centered_a,
            centered_b,
            active_candidate_event,
            no_cycle_subspace.expand_as(candidate_cycle_energy),
            alpha=alpha,
            contraction_terms=candidates + rank,
        )
    )
    if diagnostics is not None:
        diagnostics["candidate_cycle_energy_fallback_count"] = fallback_count
    # After the exact repair, any remaining material negative indicates an
    # index/mask regression and must not be hidden by a blind clamp.
    candidate_cycle_energy = _clamp_roundoff_nonnegative(
        candidate_cycle_energy,
        candidate_cycle_magnitude,
        context="candidate cycle energy",
        contraction_terms=candidates + rank,
    )
    candidate_cycle_energy = torch.where(
        active_candidate_event,
        candidate_cycle_energy,
        torch.zeros_like(candidate_cycle_energy),
    )

    candidate_event_trust = candidate_gradient_energy / (
        candidate_gradient_energy + candidate_cycle_energy + 1e-12
    )
    candidate_event_trust = torch.where(
        active_candidate_event,
        candidate_event_trust,
        torch.zeros_like(candidate_event_trust),
    )
    trusted_event_divergence = trusted_gradient_divergence(
        event_potential,
        candidate_event_trust,
        candidate_mask,
        history_mask,
    ).double()

    gradient_energy = candidate_gradient_energy.sum(dim=1)
    cycle_energy = candidate_cycle_energy.sum(dim=1)
    flow_energy = gradient_energy + cycle_energy
    outputs = (
        event_potential,
        candidate_event_trust,
        candidate_gradient_energy,
        candidate_cycle_energy,
        trusted_event_divergence,
        flow_energy,
        gradient_energy,
        cycle_energy,
    )
    return tuple(value.float() for value in outputs)


def explicit_wedge_flow(
    factor_a: torch.Tensor,
    factor_b: torch.Tensor,
    event_weights: torch.Tensor,
    candidate_mask: torch.Tensor,
    history_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the small-C edge field for contracts and diagnostics.

    The production path uses an algebraically equivalent low-rank implicit
    divergence and never creates the `[B, C, C, H]` tensor.
    """

    if factor_a.shape != factor_b.shape or factor_a.ndim != 4:
        raise ValueError("factors must share shape [B, C, H, R]")
    batch, candidates, history, rank = factor_a.shape
    if event_weights.shape != (batch, history):
        raise ValueError("event_weights shape mismatch")
    if candidate_mask.shape != (batch, candidates):
        raise ValueError("candidate_mask shape mismatch")
    if history_mask.shape != (batch, history):
        raise ValueError("history_mask shape mismatch")
    if rank == 0:
        raise ValueError("factor rank must be positive")

    forward = torch.einsum("bihr,bkhr->bikh", factor_a, factor_b)
    reverse = torch.einsum("bihr,bkhr->bikh", factor_b, factor_a)
    edge_by_event = (forward - reverse) / float(2 * rank)
    pair_mask = (
        candidate_mask[:, :, None, None].bool()
        & candidate_mask[:, None, :, None].bool()
        & history_mask[:, None, None, :].bool()
    )
    edge_by_event = torch.where(
        pair_mask, edge_by_event, torch.zeros_like(edge_by_event)
    )
    edge = torch.einsum("bikh,bh->bik", edge_by_event, event_weights)
    valid_count = candidate_mask.sum(dim=-1).clamp_min(1).to(edge.dtype)
    divergence = edge.sum(dim=-1) / valid_count[:, None]
    divergence = torch.where(
        candidate_mask.bool(), divergence, torch.zeros_like(divergence)
    )
    return edge, divergence


class ConservativeWedgeFlowProbeRanker(nn.Module):
    """Implicit low-rank skew flow with candidate-local Hodge trust.

    For event `j`, candidates `i` and `k` receive learned bounded factors
    `(a_ij, b_ij)`.  The conceptual directed edge is

      F_ikj = (a_ij^T b_kj - b_ij^T a_kj) / (2R),

    so `F_ikj = -F_kij`.  The score update is the history-weighted graph
    divergence, divided by the same valid-candidate count for every node.  Each
    projected edge is gated by the product of its two endpoint-local Hodge
    trusts.  Pure cycles therefore abstain, while heterogeneous cycle evidence
    attenuates only its incident candidates.  Production uses candidate means
    and low-rank Gram matrices in O(B*H*C*R^2), not O(C^2*H).
    """

    def __init__(
        self,
        input_dim: int,
        flow_dim: int,
        *,
        score_delta_max: float = 1.0,
        trust_mode: TrustMode = "local_hodge",
    ) -> None:
        super().__init__()
        if input_dim <= 0 or flow_dim <= 0:
            raise ValueError("input_dim and flow_dim must be positive")
        if score_delta_max <= 0.0:
            raise ValueError("score_delta_max must be positive")
        if trust_mode not in {
            "local_hodge",
            "untrusted",
            "global_hodge",
            "direct_learned",
        }:
            raise ValueError(f"unknown trust_mode: {trust_mode}")
        self.input_dim = int(input_dim)
        self.flow_dim = int(flow_dim)
        self.score_delta_max = float(score_delta_max)
        self.trust_mode = trust_mode

        self.query_norm = nn.LayerNorm(input_dim)
        self.candidate_norm = nn.LayerNorm(input_dim)
        self.history_norm = nn.LayerNorm(input_dim)
        self.query_projection = nn.Linear(input_dim, flow_dim, bias=False)
        self.candidate_projection = nn.Linear(input_dim, flow_dim, bias=False)
        self.history_projection = nn.Linear(input_dim, flow_dim, bias=False)
        self.factor_a_projection = nn.Linear(flow_dim, flow_dim, bias=False)
        self.factor_b_projection = nn.Linear(flow_dim, flow_dim, bias=False)
        self.direct_gate_projection = (
            nn.Linear(flow_dim, 1, bias=False)
            if trust_mode == "direct_learned"
            else None
        )
        # Zero starts exactly at the registered base.  The first optimizer step
        # opens the scalar score path; the second reaches the factor projections.
        self.raw_residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    @property
    def residual_scale(self) -> torch.Tensor:
        return self.score_delta_max * torch.tanh(self.raw_residual_scale)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_prior: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> WedgeFlowOutput:
        self._validate(
            query,
            candidates,
            history,
            candidate_mask,
            history_mask,
            history_prior,
            base_scores,
        )
        candidate_mask = candidate_mask.bool()
        history_mask = history_mask.bool()
        history_present = history_mask.any(dim=-1)

        safe_candidates = torch.where(
            candidate_mask[:, :, None], candidates, torch.zeros_like(candidates)
        )
        safe_history = torch.where(
            history_mask[:, :, None], history, torch.zeros_like(history)
        )
        safe_prior = torch.where(
            history_mask, history_prior, torch.zeros_like(history_prior)
        )

        q = self.query_projection(self.query_norm(query))
        c = self.candidate_projection(self.candidate_norm(safe_candidates))
        h = self.history_projection(self.history_norm(safe_history))
        joint = torch.tanh(
            q[:, None, None, :] + c[:, :, None, :] + h[:, None, :, :]
        )
        factor_a = torch.tanh(self.factor_a_projection(joint))
        factor_b = torch.tanh(self.factor_b_projection(joint))
        pair_mask = candidate_mask[:, :, None, None] & history_mask[:, None, :, None]
        factor_a = torch.where(pair_mask, factor_a, torch.zeros_like(factor_a))
        factor_b = torch.where(pair_mask, factor_b, torch.zeros_like(factor_b))

        # Factors may be produced under AMP. The helper promotes the
        # cancellation-sensitive Hodge moments to FP64 and returns FP32;
        # final-logit arithmetic also stays FP32.
        with torch.autocast(device_type=query.device.type, enabled=False):
            factor_a = factor_a.float()
            factor_b = factor_b.float()
            q_gate = F.normalize(q.float(), dim=-1, eps=1e-6)
            h_gate = F.normalize(h.float(), dim=-1, eps=1e-6)
            confidence = torch.sigmoid(
                math.sqrt(float(self.flow_dim))
                * torch.einsum("br,bhr->bh", q_gate, h_gate)
            )
            event_mass = confidence * safe_prior.float()
            event_mass = torch.where(
                history_mask, event_mass, torch.zeros_like(event_mass)
            )
            event_weights = event_mass / (
                1.0 + event_mass.sum(dim=-1, keepdim=True)
            )

            hodge_diagnostics: dict[str, int] = {}
            (
                event_potential,
                hodge_candidate_trust,
                candidate_gradient_energy,
                candidate_cycle_energy,
                trusted_event_divergence,
                flow_energy,
                gradient_energy,
                cycle_energy,
            ) = low_rank_hodge_calibration(
                factor_a,
                factor_b,
                candidate_mask,
                history_mask,
                diagnostics=hodge_diagnostics,
            )
            active_candidate_event = (
                candidate_mask[:, :, None] & history_mask[:, None, :]
            )
            if self.trust_mode == "local_hodge":
                candidate_event_trust = hodge_candidate_trust
            elif self.trust_mode == "untrusted":
                candidate_event_trust = active_candidate_event.to(
                    event_potential.dtype
                )
                trusted_event_divergence = trusted_gradient_divergence(
                    event_potential,
                    candidate_event_trust,
                    candidate_mask,
                    history_mask,
                )
            elif self.trust_mode == "global_hodge":
                global_fraction = gradient_energy / (
                    gradient_energy + cycle_energy + 1e-12
                )
                candidate_event_trust = torch.where(
                    active_candidate_event,
                    global_fraction[:, None, :],
                    torch.zeros_like(event_potential),
                )
                trusted_event_divergence = (
                    0.5 * global_fraction[:, None, :] * event_potential
                )
                trusted_event_divergence = torch.where(
                    active_candidate_event,
                    trusted_event_divergence,
                    torch.zeros_like(trusted_event_divergence),
                )
            else:
                if self.direct_gate_projection is None:
                    raise AssertionError("direct gate projection is missing")
                candidate_event_trust = torch.sigmoid(
                    self.direct_gate_projection(joint.float()).squeeze(-1)
                )
                candidate_event_trust = torch.where(
                    active_candidate_event,
                    candidate_event_trust,
                    torch.zeros_like(candidate_event_trust),
                )
                trusted_event_divergence = trusted_gradient_divergence(
                    event_potential,
                    candidate_event_trust,
                    candidate_mask,
                    history_mask,
                )
            # C affects only endpoint trust inside the helper. Ranking direction
            # comes exclusively from the projected potential u_i-u_k.
            divergence = torch.einsum(
                "bch,bh->bc", trusted_event_divergence, event_weights
            )
            divergence = torch.where(
                candidate_mask & history_present[:, None],
                divergence,
                torch.zeros_like(divergence),
            )
            valid_count = candidate_mask.sum(dim=-1).clamp_min(1)
            numerical_common_mode = divergence.sum(
                dim=-1, keepdim=True
            ) / valid_count[:, None].to(divergence.dtype)
            divergence = torch.where(
                candidate_mask & history_present[:, None],
                divergence - numerical_common_mode,
                torch.zeros_like(divergence),
            )

            scale = self.residual_scale.float()
            conservative_delta = scale * divergence
            active = candidate_mask & history_present[:, None]
            immutable_base = base_scores.detach()
            personalized = immutable_base + conservative_delta
            scores = torch.where(active, personalized, immutable_base)
            applied_delta = torch.where(
                active, scores - immutable_base, torch.zeros_like(immutable_base)
            )
        return WedgeFlowOutput(
            scores=scores,
            base_scores=immutable_base,
            conservative_score_delta=conservative_delta,
            applied_score_delta=applied_delta,
            divergence=divergence,
            event_potential=event_potential,
            trusted_event_divergence=trusted_event_divergence,
            candidate_event_trust=candidate_event_trust,
            hodge_candidate_trust=hodge_candidate_trust,
            candidate_gradient_energy=candidate_gradient_energy,
            candidate_cycle_energy=candidate_cycle_energy,
            flow_energy=flow_energy,
            gradient_energy=gradient_energy,
            cycle_energy=cycle_energy,
            factor_a=factor_a,
            factor_b=factor_b,
            event_weights=event_weights,
            history_present=history_present,
            residual_scale=scale,
            trust_mode=self.trust_mode,
            cycle_energy_fallback_count=hodge_diagnostics[
                "candidate_cycle_energy_fallback_count"
            ],
        )

    def _validate(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        candidate_mask: torch.Tensor,
        history_mask: torch.Tensor,
        history_prior: torch.Tensor,
        base_scores: torch.Tensor,
    ) -> None:
        if query.ndim != 2 or query.shape[-1] != self.input_dim:
            raise ValueError(f"query must have shape [B, {self.input_dim}]")
        if candidates.ndim != 3 or candidates.shape[-1] != self.input_dim:
            raise ValueError(
                f"candidates must have shape [B, C, {self.input_dim}]"
            )
        if history.ndim != 3 or history.shape[-1] != self.input_dim:
            raise ValueError(f"history must have shape [B, H, {self.input_dim}]")
        batch, candidate_count, _ = candidates.shape
        history_count = history.shape[1]
        if candidate_count == 0:
            raise ValueError("at least one candidate column is required")
        if query.shape[0] != batch or history.shape[0] != batch:
            raise ValueError("batch dimensions must match")
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask shape mismatch")
        if history_mask.shape != (batch, history_count):
            raise ValueError("history_mask shape mismatch")
        if history_prior.shape != (batch, history_count):
            raise ValueError("history_prior shape mismatch")
        if base_scores.shape != (batch, candidate_count):
            raise ValueError("base_scores shape mismatch")
        if base_scores.dtype != torch.float32:
            raise ValueError("base_scores must be FP32 for exact score contracts")
        c_mask = candidate_mask.bool()
        h_mask = history_mask.bool()
        if not bool(c_mask.any(dim=-1).all().item()):
            raise ValueError("every request needs at least one valid candidate")
        if not bool(torch.isfinite(query).all().item()):
            raise ValueError("query contains non-finite values")
        if not bool(torch.isfinite(candidates[c_mask]).all().item()):
            raise ValueError("valid candidates contain non-finite values")
        if h_mask.any() and not bool(torch.isfinite(history[h_mask]).all().item()):
            raise ValueError("valid history contains non-finite values")
        if h_mask.any() and not bool(torch.isfinite(history_prior[h_mask]).all().item()):
            raise ValueError("valid history prior contains non-finite values")
        if h_mask.any() and not bool((history_prior[h_mask] > 0).all().item()):
            raise ValueError("valid history prior must be positive")
        if h_mask.any() and not bool((history_prior[h_mask] <= 1).all().item()):
            raise ValueError("valid history prior must be at most one")
        if not bool(torch.isfinite(base_scores[c_mask]).all().item()):
            raise ValueError("valid base scores contain non-finite values")
