"""History-transition cone layer inside a compact Transformer ranker."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SolverOutput:
    coefficients: torch.Tensor
    reconstruction: torch.Tensor
    objective_trace: torch.Tensor


def _objective(
    relation: torch.Tensor,
    transitions: torch.Tensor,
    coefficients: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    reconstruction = torch.einsum("bct,btd->bcd", coefficients, transitions)
    residual = relation - reconstruction
    return 0.5 * residual.square().sum(dim=-1) + 0.5 * ridge * coefficients.square().sum(dim=-1)


def _masked_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask[:, None, :].expand_as(logits)
    any_valid = expanded.any(dim=-1, keepdim=True)
    safe = logits.masked_fill(~expanded, -torch.inf)
    safe = torch.where(any_valid, safe, torch.zeros_like(safe))
    values = torch.softmax(safe, dim=-1).masked_fill(~expanded, 0.0)
    return torch.where(any_valid, values, torch.zeros_like(values))


def solve_transition_coefficients(
    relation: torch.Tensor,
    transitions: torch.Tensor,
    transition_mask: torch.Tensor,
    *,
    mode: str,
    steps: int,
    ridge: float,
    temperature: torch.Tensor | float = 1.0,
) -> SolverOutput:
    """Solve or approximate a request-local transition reconstruction.

    Args:
        relation: query-to-candidate displacements ``[B,C,D]``.
        transitions: chronological history displacements ``[B,T,D]``.
        transition_mask: valid adjacent transitions ``[B,T]``.
    """

    if relation.ndim != 3 or transitions.ndim != 3:
        raise ValueError("relation and transitions must be rank three")
    if relation.shape[0] != transitions.shape[0] or relation.shape[2] != transitions.shape[2]:
        raise ValueError("relation/transition shape mismatch")
    if transition_mask.shape != transitions.shape[:2]:
        raise ValueError("transition mask shape mismatch")
    if mode not in {"cone", "span", "relu1", "simplex"}:
        raise ValueError(mode)
    if steps < 1:
        raise ValueError("steps must be positive")

    mask = transition_mask.to(transitions.dtype)
    dictionary = transitions * mask.unsqueeze(-1)
    gram = torch.einsum("btd,bsd->bts", dictionary, dictionary)
    cross = torch.einsum("bcd,btd->bct", relation, dictionary)
    valid_pair = mask.unsqueeze(2) * mask.unsqueeze(1)
    gram = gram * valid_pair

    if mode == "simplex":
        logits = cross / torch.as_tensor(temperature, dtype=cross.dtype, device=cross.device).clamp_min(1e-4)
        coefficients = _masked_softmax(logits, transition_mask.bool())
        reconstruction = torch.einsum("bct,btd->bcd", coefficients, dictionary)
        objective = _objective(relation, dictionary, coefficients, ridge).unsqueeze(0)
        return SolverOutput(coefficients, reconstruction, objective)

    # The absolute row-sum bound dominates the largest eigenvalue of the
    # positive-semidefinite Gram matrix and therefore gives a safe fixed step.
    lipschitz = gram.abs().sum(dim=-1).amax(dim=-1) + ridge
    step_size = lipschitz.clamp_min(1e-6).reciprocal()[:, None, None]
    coefficients = torch.zeros_like(cross)
    objective_values = [_objective(relation, dictionary, coefficients, ridge)]
    iterations = 1 if mode == "relu1" else steps
    for _ in range(iterations):
        gradient = torch.einsum("bct,bts->bcs", coefficients, gram) - cross
        gradient = gradient + ridge * coefficients
        coefficients = coefficients - step_size * gradient
        if mode in {"cone", "relu1"}:
            coefficients = torch.relu(coefficients)
        coefficients = coefficients * mask[:, None, :]
        objective_values.append(_objective(relation, dictionary, coefficients, ridge))
    reconstruction = torch.einsum("bct,btd->bcd", coefficients, dictionary)
    return SolverOutput(
        coefficients=coefficients,
        reconstruction=reconstruction,
        objective_trace=torch.stack(objective_values),
    )


@dataclass(frozen=True)
class HTCTOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    relation: torch.Tensor
    transitions: torch.Tensor
    transition_mask: torch.Tensor
    coefficients: torch.Tensor
    reconstruction: torch.Tensor
    reconstruction_reduction: torch.Tensor
    objective_trace: torch.Tensor


class HTCTRanker(nn.Module):
    VALID_MODES = frozenset({"cone", "span", "relu1", "simplex", "pooled_mlp"})

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        nhead: int,
        ffn_dim: int,
        lower_layers: int,
        upper_layers: int,
        relation_dim: int,
        history_slots: int,
        dropout: float,
        solver_steps: int,
        ridge: float,
        evidence_scale_max: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(mode)
        if d_model % nhead:
            raise ValueError("d_model must be divisible by nhead")
        self.mode = mode
        self.history_slots = int(history_slots)
        self.solver_steps = int(solver_steps)
        self.ridge = float(ridge)
        self.evidence_scale_max = float(evidence_scale_max)

        self.input_projection = nn.Linear(input_dim, d_model)
        self.query_type = nn.Parameter(torch.empty(d_model))
        self.candidate_type = nn.Parameter(torch.empty(d_model))

        def encoder(layers: int) -> nn.TransformerEncoder:
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=ffn_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            return nn.TransformerEncoder(layer, num_layers=layers, norm=nn.LayerNorm(d_model))

        self.lower_transformer = encoder(lower_layers)
        self.upper_transformer = encoder(upper_layers)
        self.shared_token_encoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.relation_projection = nn.Linear(d_model, relation_dim, bias=False)
        self.relation_write = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.pooled_control = nn.Sequential(
            nn.LayerNorm(2 * relation_dim),
            nn.Linear(2 * relation_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, relation_dim),
        )
        self.score_head = nn.Linear(d_model, 1)
        self.layer_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.temperature_raw = nn.Parameter(torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for value in (self.query_type, self.candidate_type):
            nn.init.normal_(value, std=0.02)
        nn.init.xavier_uniform_(self.score_head.weight)
        nn.init.zeros_(self.score_head.bias)

    def _encode_query_candidates(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        projected_query = self.input_projection(query)
        projected_candidates = self.input_projection(candidates)
        relation_query = projected_query + self.shared_token_encoder(projected_query)
        relation_candidates = projected_candidates + self.shared_token_encoder(projected_candidates)
        query_token = relation_query.unsqueeze(1) + self.query_type
        candidate_tokens = relation_candidates + self.candidate_type
        tokens = torch.cat((query_token, candidate_tokens), dim=1)
        padding = torch.cat(
            (
                torch.zeros(query.shape[0], 1, dtype=torch.bool, device=query.device),
                ~candidate_mask.bool(),
            ),
            dim=1,
        )
        lower = self.lower_transformer(tokens, src_key_padding_mask=padding)
        base_upper = self.upper_transformer(lower, src_key_padding_mask=padding)
        base = self.score_head(base_upper[:, 1:]).squeeze(-1)
        base = base.masked_fill(~candidate_mask.bool(), -1e4)
        return base, lower[:, 0], lower[:, 1:], relation_query, relation_candidates

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
        query_present: torch.Tensor | None = None,
        mode: str | None = None,
    ) -> HTCTOutput:
        active_mode = self.mode if mode is None else mode
        if active_mode not in self.VALID_MODES:
            raise ValueError(active_mode)
        batch, candidate_count, _ = candidates.shape
        if history.shape[1] != self.history_slots or history_mask.shape != history.shape[:2]:
            raise ValueError("history shape mismatch")
        if candidate_mask is None:
            candidate_mask = torch.ones(batch, candidate_count, dtype=torch.bool, device=query.device)
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool, device=query.device)

        base, query_state, candidate_states, relation_query, relation_candidates = self._encode_query_candidates(
            query, candidates, candidate_mask
        )
        projected_history = self.input_projection(history)
        history_states = projected_history + self.shared_token_encoder(projected_history)

        relation = self.relation_projection(
            relation_candidates - relation_query.unsqueeze(1)
        )
        transition_mask = history_mask[:, :-1].bool() & history_mask[:, 1:].bool()
        transitions = self.relation_projection(history_states[:, 1:] - history_states[:, :-1])
        transitions = transitions * transition_mask.unsqueeze(-1).to(transitions.dtype)

        if active_mode == "pooled_mlp":
            count = transition_mask.sum(dim=1, keepdim=True).clamp_min(1).to(transitions.dtype)
            pooled = transitions.sum(dim=1) / count
            pooled = pooled[:, None, :].expand(-1, candidate_count, -1)
            reconstruction = self.pooled_control(torch.cat((relation, pooled), dim=-1))
            present = transition_mask.any(dim=1)
            reconstruction = torch.where(
                present[:, None, None], reconstruction, torch.zeros_like(reconstruction)
            )
            coefficients = torch.zeros(
                batch,
                candidate_count,
                transitions.shape[1],
                dtype=relation.dtype,
                device=relation.device,
            )
            objective_trace = torch.empty(
                0, batch, candidate_count, dtype=relation.dtype, device=relation.device
            )
        else:
            temperature = F.softplus(self.temperature_raw) + 1e-4
            solver = solve_transition_coefficients(
                relation,
                transitions,
                transition_mask,
                mode=active_mode,
                steps=self.solver_steps,
                ridge=self.ridge,
                temperature=temperature,
            )
            coefficients = solver.coefficients
            reconstruction = solver.reconstruction
            objective_trace = solver.objective_trace

        zero_relation = torch.zeros_like(reconstruction)
        write = self.relation_write(reconstruction) - self.relation_write(zero_relation)
        valid_count = candidate_mask.to(write.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
        write_mean = write.masked_fill(~candidate_mask[:, :, None], 0.0).sum(dim=1, keepdim=True)
        write = write - write_mean / valid_count.unsqueeze(-1)
        write = write.masked_fill(~candidate_mask[:, :, None], 0.0)
        layer_scale = self.evidence_scale_max * torch.sigmoid(self.layer_scale_raw)
        personalized_candidates = candidate_states + layer_scale * write
        personalized_tokens = torch.cat((query_state.unsqueeze(1), personalized_candidates), dim=1)
        padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=query.device),
                ~candidate_mask.bool(),
            ),
            dim=1,
        )
        personalized_upper = self.upper_transformer(
            personalized_tokens, src_key_padding_mask=padding
        )
        personalized = self.score_head(personalized_upper[:, 1:]).squeeze(-1)
        personalized = personalized.masked_fill(~candidate_mask.bool(), -1e4)
        evidence_present = transition_mask.any(dim=1) & query_present.bool()
        scores = torch.where(evidence_present[:, None], personalized, base)

        baseline_error = relation.square().sum(dim=-1)
        reconstruction_error = (relation - reconstruction).square().sum(dim=-1)
        reduction = (baseline_error - reconstruction_error) / baseline_error.clamp_min(1e-8)
        reduction = torch.where(
            evidence_present[:, None], reduction, torch.zeros_like(reduction)
        )
        return HTCTOutput(
            scores=scores,
            base_scores=base,
            relation=relation,
            transitions=transitions,
            transition_mask=transition_mask,
            coefficients=torch.where(
                evidence_present[:, None, None], coefficients, torch.zeros_like(coefficients)
            ),
            reconstruction=torch.where(
                evidence_present[:, None, None], reconstruction, torch.zeros_like(reconstruction)
            ),
            reconstruction_reduction=reduction,
            objective_trace=objective_trace,
        )
