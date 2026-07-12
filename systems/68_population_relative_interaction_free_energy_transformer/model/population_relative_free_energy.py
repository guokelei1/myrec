from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


MODES = (
    "interaction_free_energy",
    "mean_interaction",
    "single_null_interaction",
    "user_only_free_energy",
    "pooled_joint_transformer",
)


def _validate_event_tensor(values: Tensor, mask: Tensor | None) -> Tensor:
    if values.ndim != 3:
        raise ValueError(f"event energies must have shape [B,C,E], got {values.shape}")
    if mask is None:
        return torch.ones(
            values.shape[0], values.shape[2], dtype=torch.bool, device=values.device
        )
    if mask.shape != (values.shape[0], values.shape[2]):
        raise ValueError(f"mask {mask.shape} does not match energies {values.shape}")
    if not bool(mask.any(dim=1).all()):
        raise ValueError("each free-energy set must contain at least one event")
    return mask.to(dtype=torch.bool, device=values.device)


def masked_free_energy(values: Tensor, mask: Tensor | None, temperature: float) -> Tensor:
    """Return tau * log(mean(exp(values/tau))) over the event axis."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    valid = _validate_event_tensor(values, mask)
    scaled = values / temperature
    neg_inf = torch.finfo(values.dtype).min
    scaled = torch.where(valid[:, None, :], scaled, neg_inf)
    counts = valid.sum(dim=1).to(values.dtype).clamp_min(1)
    return temperature * (
        torch.logsumexp(scaled, dim=-1) - torch.log(counts)[:, None]
    )


def masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    valid = _validate_event_tensor(values, mask)
    weights = valid[:, None, :].to(values.dtype)
    return (values * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1)


def interaction_from_energies(
    user_candidate: Tensor,
    reference_candidate: Tensor,
    user_null: Tensor,
    reference_null: Tensor,
    *,
    mode: str,
    temperature: float,
    user_mask: Tensor | None = None,
    reference_mask: Tensor | None = None,
) -> Tensor:
    """Compute a candidate correction before candidate centering.

    Candidate tensors are [B,C,E]. Null tensors may be [B,1,E] and are
    broadcast over candidates after event aggregation.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if user_candidate.ndim != 3 or reference_candidate.ndim != 3:
        raise ValueError("candidate energy tensors must be rank three")
    if user_null.ndim != 3 or reference_null.ndim != 3:
        raise ValueError("null energy tensors must be rank three")
    if user_null.shape[1] != 1 or reference_null.shape[1] != 1:
        raise ValueError("null energy tensors must have one candidate channel")

    if mode == "mean_interaction":
        u_c = masked_mean(user_candidate, user_mask)
        r_c = masked_mean(reference_candidate, reference_mask)
        u_0 = masked_mean(user_null, user_mask)
        r_0 = masked_mean(reference_null, reference_mask)
        return (u_c - r_c) - (u_0 - r_0)

    if mode == "single_null_interaction":
        # The caller supplies one fixed zero event as the reference tensors.
        u_c = masked_free_energy(user_candidate, user_mask, temperature)
        r_c = masked_free_energy(reference_candidate, reference_mask, temperature)
        u_0 = masked_free_energy(user_null, user_mask, temperature)
        r_0 = masked_free_energy(reference_null, reference_mask, temperature)
        return (u_c - r_c) - (u_0 - r_0)

    if mode == "user_only_free_energy":
        u_c = masked_free_energy(user_candidate, user_mask, temperature)
        u_0 = masked_free_energy(user_null, user_mask, temperature)
        return u_c - u_0

    # `pooled_joint_transformer` reaches this helper with one pooled event in
    # both sets, so the same four-way equation applies.
    u_c = masked_free_energy(user_candidate, user_mask, temperature)
    r_c = masked_free_energy(reference_candidate, reference_mask, temperature)
    u_0 = masked_free_energy(user_null, user_mask, temperature)
    r_0 = masked_free_energy(reference_null, reference_mask, temperature)
    return (u_c - r_c) - (u_0 - r_0)


@dataclass(frozen=True)
class RankerOutput:
    scores: Tensor
    correction: Tensor
    raw_interaction: Tensor


class PopulationRelativeFreeEnergyRanker(nn.Module):
    """Shared triplet Transformer with a population-relative energy read."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        temperature: float,
        mode: str,
        correction_scale: float = 1.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown mode: {mode}")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by heads")
        self.mode = mode
        self.temperature = float(temperature)
        self.correction_scale = float(correction_scale)
        self.input_dim = int(input_dim)

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.token_type = nn.Parameter(torch.empty(3, hidden_dim))
        self.null_candidate = nn.Parameter(torch.empty(input_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.triplet_transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.token_type, std=0.02)
        nn.init.normal_(self.null_candidate, std=0.02)
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        nn.init.xavier_uniform_(self.energy_head.weight)
        nn.init.zeros_(self.energy_head.bias)

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _triplet_energy(self, query: Tensor, candidates: Tensor, events: Tensor) -> Tensor:
        if query.ndim != 2:
            raise ValueError("query must have shape [B,D]")
        if candidates.ndim != 3 or events.ndim != 3:
            raise ValueError("candidates/events must have shape [B,N,D]")
        batch, candidate_count, dim = candidates.shape
        event_count = events.shape[1]
        if query.shape != (batch, dim) or events.shape[0] != batch or events.shape[2] != dim:
            raise ValueError("incompatible query/candidate/event shapes")

        q = query[:, None, None, :].expand(batch, candidate_count, event_count, dim)
        c = candidates[:, :, None, :].expand(batch, candidate_count, event_count, dim)
        e = events[:, None, :, :].expand(batch, candidate_count, event_count, dim)
        tokens = torch.stack((q, c, e), dim=-2)
        flat = tokens.reshape(batch * candidate_count * event_count, 3, dim)
        hidden = self.input_projection(flat) + self.token_type[None, :, :]
        hidden = self.triplet_transformer(hidden)
        # The candidate token has seen q and e through self-attention.
        raw_energy = self.energy_head(self.output_norm(hidden[:, 1])).squeeze(-1)
        # Every matched mode shares this bounded map. It prevents a mean
        # reduction from emulating tail selection only through unbounded scale.
        energy = 2.0 * torch.tanh(0.5 * raw_energy)
        return energy.reshape(batch, candidate_count, event_count)

    def _prepare_sets(self, history: Tensor, reference: Tensor) -> tuple[Tensor, Tensor]:
        if self.mode == "pooled_joint_transformer":
            return history.mean(dim=1, keepdim=True), reference.mean(dim=1, keepdim=True)
        if self.mode == "single_null_interaction":
            zero_reference = torch.zeros(
                reference.shape[0], 1, reference.shape[2],
                dtype=reference.dtype, device=reference.device,
            )
            return history, zero_reference
        return history, reference

    def forward(
        self,
        *,
        query: Tensor,
        candidates: Tensor,
        history: Tensor,
        reference: Tensor,
        base_scores: Tensor,
        history_present: Tensor | None = None,
        query_present: Tensor | None = None,
        repeat_mask: Tensor | None = None,
        repeat_scores: Tensor | None = None,
    ) -> RankerOutput:
        batch, candidate_count, _ = candidates.shape
        if base_scores.shape != (batch, candidate_count):
            raise ValueError("base_scores has the wrong shape")
        history_set, reference_set = self._prepare_sets(history, reference)

        all_candidates = torch.cat(
            (
                candidates,
                self.null_candidate[None, None, :].expand(batch, 1, -1),
            ),
            dim=1,
        )
        user_energy = self._triplet_energy(query, all_candidates, history_set)
        ref_energy = self._triplet_energy(query, all_candidates, reference_set)
        raw = interaction_from_energies(
            user_energy[:, :candidate_count],
            ref_energy[:, :candidate_count],
            user_energy[:, candidate_count:],
            ref_energy[:, candidate_count:],
            mode=self.mode,
            temperature=self.temperature,
        )
        correction = raw - raw.mean(dim=1, keepdim=True)
        correction = self.correction_scale * correction

        active = torch.ones(batch, dtype=torch.bool, device=query.device)
        if history_present is not None:
            active &= history_present.to(dtype=torch.bool, device=query.device)
        if query_present is not None:
            active &= query_present.to(dtype=torch.bool, device=query.device)
        scores = torch.where(active[:, None], base_scores + correction, base_scores)
        correction = torch.where(active[:, None], correction, torch.zeros_like(correction))
        raw = torch.where(active[:, None], raw, torch.zeros_like(raw))

        if repeat_mask is not None:
            if repeat_scores is None or repeat_scores.shape != base_scores.shape:
                raise ValueError("repeat_scores are required with repeat_mask")
            use_repeat = repeat_mask.to(dtype=torch.bool, device=query.device)
            scores = torch.where(use_repeat[:, None], repeat_scores, scores)
            correction = torch.where(use_repeat[:, None], torch.zeros_like(correction), correction)
            raw = torch.where(use_repeat[:, None], torch.zeros_like(raw), raw)
        return RankerOutput(scores=scores, correction=correction, raw_interaction=raw)
