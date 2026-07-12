"""Minimal Counterfactual Evidence-Contract Transformer (CECT).

The model consumes frozen text-Transformer states, but all personalized ranking
information is formed inside the trainable joint Transformer below.  Training
code constructs counterfactual twins; inference calls this module with only the
observational query, candidate, and strictly-prior history.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def masked_zscore(
    values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Within-row z-score with exact zeros at masked positions."""

    weights = mask.to(values.dtype)
    count = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
    mean = (values * weights).sum(dim=-1, keepdim=True) / count
    centered = (values - mean) * weights
    variance = centered.square().sum(dim=-1, keepdim=True) / count
    result = centered / torch.sqrt(variance + eps)
    return result * weights


def counterfactual_upper_quantile(
    values: torch.Tensor, alpha: float
) -> torch.Tensor:
    """Finite-sample upper conformal-style quantile frozen in the protocol."""

    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("counterfactual quantile requires a non-empty vector")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not torch.isfinite(values).all():
        raise ValueError("counterfactual energies must be finite")
    ordered = torch.sort(values).values
    # Protocol uses one-indexed ceil((n + 1) * (1-alpha)), clipped to n.
    one_indexed = math.ceil((ordered.numel() + 1) * (1.0 - alpha))
    index = min(max(one_indexed, 1), ordered.numel()) - 1
    return ordered[index]


@dataclass
class CECTOutput:
    scores: torch.Tensor
    contract_scores: torch.Tensor
    exact_scores: torch.Tensor
    transfer_scores: torch.Tensor
    energies: torch.Tensor
    values: torch.Tensor
    gates: torch.Tensor
    hard_admission: torch.Tensor
    event_mask: torch.Tensor
    nonexact_mask: torch.Tensor
    request_evidence_present: torch.Tensor


class CECTModel(nn.Module):
    """Joint query-candidate-event Transformer with a calibrated contract."""

    def __init__(
        self,
        frozen_text_dim: int = 512,
        d_model: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 192,
        dropout: float = 0.1,
        max_history: int = 20,
        category_buckets: int = 4096,
        beta: float = 0.30,
        gate_temperature: float = 0.10,
        mode: str = "contract",
    ) -> None:
        super().__init__()
        if mode not in {"contract", "plain"}:
            raise ValueError(f"unknown mode: {mode}")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.frozen_text_dim = frozen_text_dim
        self.d_model = d_model
        self.max_history = max_history
        self.category_buckets = category_buckets
        self.beta = float(beta)
        self.gate_temperature = float(gate_temperature)
        self.mode = mode

        self.query_projection = nn.Linear(frozen_text_dim, d_model, bias=False)
        self.candidate_projection = nn.Linear(frozen_text_dim, d_model, bias=False)
        self.event_projection = nn.Linear(frozen_text_dim, d_model, bias=False)
        self.category_embedding = nn.Embedding(category_buckets, d_model, padding_idx=0)
        self.segment_embedding = nn.Embedding(3, d_model)
        self.position_embedding = nn.Embedding(max_history + 1, d_model, padding_idx=0)
        self.event_type_embedding = nn.Embedding(3, d_model, padding_idx=0)
        # relation 0=none/pad, 1=category match, 2=exact item.
        self.relation_embedding = nn.Embedding(3, d_model, padding_idx=0)
        self.query_mask_token = nn.Parameter(torch.zeros(d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.energy_head = nn.Linear(d_model, 1)
        self.value_head = nn.Linear(d_model, 1)
        self.transfer_logit = nn.Parameter(torch.tensor(-1.0))
        # A buffer, not a parameter: plain and contract variants remain exactly
        # parameter matched.  It is replaced only after train-only calibration.
        self.register_buffer("certificate_threshold", torch.tensor(0.0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
        nn.init.zeros_(self.query_mask_token)

    @property
    def transfer_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.transfer_logit)

    def set_certificate_threshold(self, threshold: float | torch.Tensor) -> None:
        value = torch.as_tensor(
            threshold,
            dtype=self.certificate_threshold.dtype,
            device=self.certificate_threshold.device,
        )
        if value.numel() != 1 or not bool(torch.isfinite(value).all().item()):
            raise ValueError("certificate threshold must be one finite scalar")
        self.certificate_threshold.copy_(value.reshape(()))

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def freeze_certificate_path(self) -> None:
        """Freeze all state that can change event energies after calibration."""

        trainable = {"value_head.weight", "value_head.bias", "transfer_logit"}
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name in trainable

    def unfreeze_all(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = True

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        *,
        condition: str = "true",
        force_mode: str | None = None,
    ) -> CECTOutput:
        """Score a padded request batch.

        Required tensor shapes:
          query [B,D0], candidates [B,C,D0], history [B,H,D0],
          candidate/history indices and categories, event weights, masks, base.
        """

        mode = force_mode or self.mode
        if mode not in {"contract", "plain"}:
            raise ValueError(f"unknown scoring mode: {mode}")
        prepared = self._condition_batch(batch, condition)
        query = prepared["query"]
        candidates = prepared["candidates"]
        history = prepared["history"]
        candidate_indices = prepared["candidate_indices"]
        history_indices = prepared["history_indices"]
        candidate_categories = prepared["candidate_categories"]
        history_categories = prepared["history_categories"]
        history_event_weights = prepared["history_event_weights"]
        history_mask = prepared["history_mask"].bool()
        candidate_mask = batch["candidate_mask"].bool()
        base_scores = batch["base_scores"]

        batch_size, candidate_count, _ = candidates.shape
        history_count = history.shape[1]
        if history_count != self.max_history:
            raise ValueError(
                f"expected history width {self.max_history}, got {history_count}"
            )

        # Flatten only conceptually; padded candidates are masked at the output.
        flat_count = batch_size * candidate_count
        q = query[:, None, :].expand(-1, candidate_count, -1).reshape(
            flat_count, -1
        )
        c = candidates.reshape(flat_count, -1)
        h = history[:, None, :, :].expand(-1, candidate_count, -1, -1).reshape(
            flat_count, history_count, -1
        )
        hmask = history_mask[:, None, :].expand(-1, candidate_count, -1).reshape(
            flat_count, history_count
        )
        cidx = candidate_indices.reshape(flat_count)
        hidx = history_indices[:, None, :].expand(-1, candidate_count, -1).reshape(
            flat_count, history_count
        )
        ccat = candidate_categories.reshape(flat_count)
        hcat = history_categories[:, None, :].expand(-1, candidate_count, -1).reshape(
            flat_count, history_count
        )
        hevent = history_event_weights[:, None, :].expand(
            -1, candidate_count, -1
        ).reshape(flat_count, history_count)

        disable_exact = condition in {"wrong", "coarse"}
        exact = (cidx[:, None] == hidx) & hmask & (hidx >= 0)
        if disable_exact:
            exact = torch.zeros_like(exact)
        category_match = (
            (ccat[:, None] == hcat) & (ccat[:, None] > 0) & hmask & ~exact
        )
        relation = torch.zeros_like(hidx, dtype=torch.long)
        relation = torch.where(category_match, torch.ones_like(relation), relation)
        relation = torch.where(exact, torch.full_like(relation, 2), relation)

        history_lengths = hmask.sum(dim=-1)
        positions = torch.arange(history_count, device=h.device)[None, :]
        reverse_age = (history_lengths[:, None] - positions).clamp(
            min=0, max=self.max_history
        )
        reverse_age = reverse_age * hmask.long()
        event_type = torch.where(
            hmask,
            torch.where(
                hevent > 1.25,
                torch.full_like(hevent, 2, dtype=torch.long),
                torch.ones_like(hevent, dtype=torch.long),
            ),
            torch.zeros_like(hevent, dtype=torch.long),
        )

        query_token = self.query_projection(q) + self.segment_embedding.weight[0]
        if condition == "query_masked":
            query_token = self.query_mask_token[None, :].expand_as(query_token)
        candidate_token = self.candidate_projection(c)
        candidate_token = (
            candidate_token
            + self.category_embedding(ccat.clamp(0, self.category_buckets - 1))
            + self.segment_embedding.weight[1]
        )
        event_tokens = (
            self.event_projection(h)
            + self.category_embedding(hcat.clamp(0, self.category_buckets - 1))
            + self.position_embedding(reverse_age)
            + self.event_type_embedding(event_type)
            + self.relation_embedding(relation)
            + self.segment_embedding.weight[2]
        )
        event_tokens = event_tokens * hmask[:, :, None].to(event_tokens.dtype)

        sequence = torch.cat(
            [query_token[:, None, :], candidate_token[:, None, :], event_tokens],
            dim=1,
        )
        padding_mask = torch.cat(
            [
                torch.zeros(flat_count, 2, dtype=torch.bool, device=h.device),
                ~hmask,
            ],
            dim=1,
        )
        hidden = self.transformer(sequence, src_key_padding_mask=padding_mask)
        event_hidden = self.final_norm(hidden[:, 2:, :])
        energies = self.energy_head(event_hidden).squeeze(-1)
        values = self.value_head(event_hidden).squeeze(-1)
        energies = energies.masked_fill(~hmask, 0.0)
        values = values.masked_fill(~hmask, 0.0)

        nonexact = hmask & ~exact
        if mode == "plain":
            masked_energy = energies.masked_fill(~nonexact, -1e9)
            gates = torch.softmax(masked_energy, dim=-1)
            gates = gates * nonexact.to(gates.dtype)
            gates = gates / gates.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            hard_admission = nonexact & (gates > 0.0)
        else:
            probability = torch.sigmoid(
                (energies - self.certificate_threshold) / self.gate_temperature
            )
            # Exact zero below the calibrated boundary; smooth above it.
            gates = (2.0 * F.relu(probability - 0.5)) * nonexact.to(probability.dtype)
            hard_admission = nonexact & (energies > self.certificate_threshold)

        age_denominator = torch.sqrt(reverse_age.clamp_min(1).to(values.dtype))
        exact_floor = (
            exact.to(values.dtype) * 3.0 * hevent.to(values.dtype) / age_denominator
        )
        exact_scores = exact_floor.sum(dim=-1)
        transfer_values = F.softplus(values) / age_denominator
        transfer_scores = (gates * transfer_values * nonexact).sum(dim=-1)
        contract_scores = exact_scores + self.transfer_scale * transfer_scores

        contract_scores = contract_scores.reshape(batch_size, candidate_count)
        exact_scores = exact_scores.reshape(batch_size, candidate_count)
        transfer_scores = transfer_scores.reshape(batch_size, candidate_count)
        energies = energies.reshape(batch_size, candidate_count, history_count)
        values = values.reshape(batch_size, candidate_count, history_count)
        gates = gates.reshape(batch_size, candidate_count, history_count)
        hard_admission = hard_admission.reshape(
            batch_size, candidate_count, history_count
        )
        event_mask = hmask.reshape(batch_size, candidate_count, history_count)
        nonexact = nonexact.reshape(batch_size, candidate_count, history_count)

        evidence_range = (
            contract_scores.masked_fill(~candidate_mask, torch.inf).amin(dim=-1),
            contract_scores.masked_fill(~candidate_mask, -torch.inf).amax(dim=-1),
        )
        has_history = history_mask.any(dim=-1)
        has_contract_variation = (evidence_range[1] - evidence_range[0]) > 1e-8
        request_evidence_present = has_history & has_contract_variation

        mixed = self.beta * masked_zscore(base_scores, candidate_mask) + (
            1.0 - self.beta
        ) * masked_zscore(contract_scores, candidate_mask)
        scores = torch.where(request_evidence_present[:, None], mixed, base_scores)
        scores = scores.masked_fill(~candidate_mask, 0.0)

        return CECTOutput(
            scores=scores,
            contract_scores=contract_scores,
            exact_scores=exact_scores,
            transfer_scores=transfer_scores,
            energies=energies,
            values=values,
            gates=gates,
            hard_admission=hard_admission,
            event_mask=event_mask,
            nonexact_mask=nonexact,
            request_evidence_present=request_evidence_present,
        )

    def _condition_batch(
        self, batch: dict[str, torch.Tensor], condition: str
    ) -> dict[str, torch.Tensor]:
        allowed = {"true", "wrong", "shuffled", "query_masked", "coarse"}
        if condition not in allowed:
            raise ValueError(f"unknown evidence condition: {condition}")
        result = {
            key: batch[key]
            for key in [
                "query",
                "candidates",
                "history",
                "candidate_indices",
                "history_indices",
                "candidate_categories",
                "history_categories",
                "history_event_weights",
                "history_mask",
            ]
        }
        if condition == "wrong":
            required = [
                "wrong_history",
                "wrong_history_indices",
                "wrong_history_categories",
                "wrong_history_event_weights",
                "wrong_history_mask",
            ]
            missing = [key for key in required if key not in batch]
            if missing:
                raise ValueError(f"wrong-history tensors missing: {missing}")
            result.update(
                {
                    "history": batch["wrong_history"],
                    "history_indices": batch["wrong_history_indices"],
                    "history_categories": batch["wrong_history_categories"],
                    "history_event_weights": batch["wrong_history_event_weights"],
                    "history_mask": batch["wrong_history_mask"],
                }
            )
        elif condition == "shuffled":
            for key in [
                "history",
                "history_indices",
                "history_categories",
                "history_event_weights",
                "history_mask",
            ]:
                result[key] = _rotate_valid_events(result[key], batch["history_mask"])
        elif condition == "coarse":
            result["candidates"] = torch.zeros_like(result["candidates"])
            result["history"] = torch.zeros_like(result["history"])
        return result


def _rotate_valid_events(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Rotate each valid event sequence by one; length-zero/one stays defined."""

    output = values.clone()
    for row in range(values.shape[0]):
        length = int(mask[row].sum().item())
        if length > 1:
            output[row, :length] = torch.roll(values[row, :length], shifts=1, dims=0)
    return output


def robust_sequence_energy(
    energies: torch.Tensor,
    nonexact_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Masked multiple-instance log-sum-exp certificate per candidate."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    masked = energies.masked_fill(~nonexact_mask, -1e9)
    aggregate = temperature * torch.logsumexp(masked / temperature, dim=-1)
    return torch.where(nonexact_mask.any(dim=-1), aggregate, torch.zeros_like(aggregate))


def multi_positive_listwise_loss(
    scores: torch.Tensor, labels: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    """Multi-positive listwise softmax used only with train labels."""

    valid = candidate_mask.bool()
    positive = valid & labels.bool()
    if not positive.any(dim=-1).all():
        raise ValueError("each train request must contain a clicked positive")
    all_lse = torch.logsumexp(scores.masked_fill(~valid, -1e9), dim=-1)
    positive_lse = torch.logsumexp(scores.masked_fill(~positive, -1e9), dim=-1)
    return (all_lse - positive_lse).mean()


def counterfactual_margin_loss(
    true_output: CECTOutput,
    twin_outputs: list[CECTOutput],
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    margin: float,
    temperature: float,
) -> torch.Tensor:
    """Robust max-over-twins sequence margin on clicked, non-exact candidates."""

    true_energy = robust_sequence_energy(
        true_output.energies, true_output.nonexact_mask, temperature
    )
    twin_energy = torch.stack(
        [
            robust_sequence_energy(output.energies, output.nonexact_mask, temperature)
            for output in twin_outputs
        ],
        dim=0,
    ).amax(dim=0)
    eligible = (
        labels.bool()
        & candidate_mask.bool()
        & true_output.nonexact_mask.any(dim=-1)
        & ~(true_output.exact_scores > 0)
    )
    if not eligible.any():
        # Preserve a differentiable zero for tiny smoke batches.
        return true_energy.sum() * 0.0
    return F.relu(float(margin) - (true_energy - twin_energy))[eligible].mean()


def model_signature(model: CECTModel) -> dict[str, Any]:
    return {
        "class": type(model).__name__,
        "mode": model.mode,
        "parameter_count": model.parameter_count(),
        "trainable_parameter_count": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "certificate_threshold": float(model.certificate_threshold.item()),
        "transfer_scale": float(model.transfer_scale.detach().cpu().item()),
    }
