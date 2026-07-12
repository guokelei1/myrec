"""C67 exact held-out-validated fast-weight Transformer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "cross_validated_write",
    "gradient_agreement_write",
    "standard_ttt_write",
    "self_validated_write",
)


@dataclass(frozen=True)
class FastWeightOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_delta: torch.Tensor
    event_weight: torch.Tensor
    event_utility: torch.Tensor
    fast_weight: torch.Tensor
    active_request: torch.Tensor


def listwise_training_loss(
    output: FastWeightOutput,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    wrong_output: FastWeightOutput | None,
    correction_l2_weight: float,
    wrong_history_neutrality_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(dim=-1) > 0
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1)
    row = -(target * log_probability.masked_fill(~mask, 0.0)).sum(dim=-1)
    ranking = row[valid].mean() if bool(valid.any()) else logits.sum() * 0.0
    active = output.active_request[:, None] & mask
    correction_l2 = (
        output.correction[active].square().mean()
        if bool(active.any())
        else output.correction.sum() * 0.0
    )
    wrong_neutrality = output.correction.sum() * 0.0
    if wrong_output is not None:
        wrong_active = wrong_output.active_request[:, None] & mask
        wrong_neutrality = (
            wrong_output.correction[wrong_active].square().mean()
            if bool(wrong_active.any())
            else wrong_output.correction.sum() * 0.0
        )
    total = (
        ranking
        + float(correction_l2_weight) * correction_l2
        + float(wrong_history_neutrality_weight) * wrong_neutrality
    )
    return total, {
        "ranking": ranking,
        "correction_l2": correction_l2,
        "wrong_neutrality": wrong_neutrality,
    }


class CrossValidatedFastWeightTransformer(nn.Module):
    """Encode pairs with a Transformer and write a request-local learner."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        projection_ffn_dim: int,
        heads: int,
        dropout: float,
        initial_inner_step: float,
    ) -> None:
        super().__init__()
        if not 0.0 < initial_inner_step < 1.0:
            raise ValueError("C67 initial inner step must lie in (0, 1)")
        if hidden_dim % heads:
            raise ValueError("C67 hidden dimension must divide heads")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.key_role = nn.Parameter(torch.empty(hidden_dim))
        self.value_role = nn.Parameter(torch.empty(hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=projection_ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pair_transformer = nn.TransformerEncoder(
            layer, num_layers=1, enable_nested_tensor=False
        )
        self.key_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.value_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim, bias=False)
        )
        self.initial_weight = nn.Parameter(torch.empty(hidden_dim, hidden_dim))
        self.inner_step_logit = nn.Parameter(
            torch.tensor(math.log(initial_inner_step / (1.0 - initial_inner_step)))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.key_role, std=0.02)
        nn.init.normal_(self.value_role, std=0.02)
        nn.init.zeros_(self.initial_weight)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    @property
    def inner_step(self) -> torch.Tensor:
        return torch.sigmoid(self.inner_step_logit)

    def _encode_pairs(
        self, first: torch.Tensor, second: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if first.shape != second.shape or first.shape[-1] != self.input_dim:
            raise ValueError("C67 pair shape differs")
        shape = first.shape[:-1]
        tokens = torch.stack(
            (
                self.input_projection(first.float()) + self.key_role,
                self.input_projection(second.float()) + self.value_role,
            ),
            dim=-2,
        ).reshape(-1, 2, self.hidden_dim)
        state = self.pair_transformer(tokens).reshape(*shape, 2, self.hidden_dim)
        key = F.normalize(self.key_head(state[..., 0, :]), dim=-1, eps=1e-6)
        value = self.value_head(state[..., 1, :])
        return key, value

    def _event_gradients(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = torch.einsum("vk,bhk->bhv", self.initial_weight, key)
        error = prediction - value
        gradients = torch.einsum("bhv,bhk->bhvk", error, key)
        base_loss = 0.5 * error.square().sum(dim=-1)
        return gradients, base_loss

    def _write(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        *,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in MODES:
            raise ValueError(f"unknown C67 mode: {mode}")
        if key.shape != value.shape or key.shape[:2] != mask.shape:
            raise ValueError("C67 write shape differs")
        mask = mask.bool()
        gradients, base_loss = self._event_gradients(key, value)
        step = self.inner_step
        proposal = self.initial_weight[None, None] - step * gradients
        post_prediction = torch.einsum("bevk,bjk->bejv", proposal, key)
        post_loss = 0.5 * (
            post_prediction - value[:, None]
        ).square().sum(dim=-1)
        batch, events = mask.shape
        diagonal = torch.eye(events, dtype=torch.bool, device=mask.device)[None]
        cross_mask = mask[:, :, None] & mask[:, None, :] & ~diagonal
        cross_count = cross_mask.sum(dim=-1)
        exact_improvement = (
            (base_loss[:, None, :] - post_loss)
            * cross_mask.to(post_loss.dtype)
        ).sum(dim=-1) / cross_count.clamp_min(1).to(post_loss.dtype)
        exact_improvement = exact_improvement.masked_fill(cross_count == 0, 0.0)

        if mode == "cross_validated_write":
            utility = exact_improvement
        elif mode == "gradient_agreement_write":
            total_gradient = (
                gradients * mask[..., None, None].to(gradients.dtype)
            ).sum(dim=1, keepdim=True)
            other_count = (mask.sum(dim=1, keepdim=True) - 1).clamp_min(1)
            other_gradient = (total_gradient - gradients) / other_count[
                ..., None, None
            ].to(gradients.dtype)
            utility = step * (gradients * other_gradient).mean(dim=(-2, -1))
            utility = utility.masked_fill(cross_count == 0, 0.0)
        elif mode == "self_validated_write":
            self_loss = post_loss.diagonal(dim1=1, dim2=2)
            utility = base_loss - self_loss
        else:
            utility = mask.to(gradients.dtype)

        if mode == "standard_ttt_write":
            positive = mask.to(gradients.dtype)
        else:
            positive = F.relu(utility) * mask.to(utility.dtype)
        denominator = positive.sum(dim=-1, keepdim=True)
        event_weight = torch.where(
            denominator > 1e-12,
            positive / denominator.clamp_min(1e-12),
            torch.zeros_like(positive),
        )
        weighted_gradient = torch.einsum("bh,bhvk->bvk", event_weight, gradients)
        fast_weight = self.initial_weight[None].expand(batch, -1, -1) - step * weighted_gradient
        return fast_weight, event_weight, utility

    @staticmethod
    def _orders(
        candidate_keys: torch.Tensor, candidate_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if candidate_keys.shape != candidate_mask.shape or candidate_keys.dtype != torch.int64:
            raise ValueError("C67 candidate keys differ")
        for row in range(candidate_keys.shape[0]):
            valid = candidate_keys[row, candidate_mask[row].bool()]
            if len(torch.unique(valid)) != len(valid):
                raise ValueError("C67 valid candidate keys must be unique")
        sentinel = torch.iinfo(torch.int64).max
        sortable = torch.where(
            candidate_mask.bool(), candidate_keys, torch.full_like(candidate_keys, sentinel)
        )
        order = torch.argsort(sortable, dim=1, stable=True)
        return order, torch.argsort(order, dim=1, stable=True)

    @staticmethod
    def _gather(value: torch.Tensor, order: torch.Tensor) -> torch.Tensor:
        index = order
        while index.ndim < value.ndim:
            index = index.unsqueeze(-1)
        return torch.gather(
            value,
            1,
            index.expand(value.shape[0], order.shape[1], *value.shape[2:]),
        )

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        candidate_keys: torch.Tensor,
        history_keys: torch.Tensor,
        history_values: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
        query_present: torch.Tensor,
        mode: str,
    ) -> FastWeightOutput:
        order, inverse = self._orders(candidate_keys, candidate_mask)
        canonical_candidates = self._gather(candidates, order)
        canonical_mask = self._gather(candidate_mask, order)
        canonical_base = self._gather(base_scores, order)
        canonical_item = self._gather(item_only_scores, order)
        history_key, history_value = self._encode_pairs(history_keys, history_values)
        history_key = history_key * history_mask[..., None].to(history_key.dtype)
        history_value = history_value * history_mask[..., None].to(history_value.dtype)
        fast_weight, event_weight, utility = self._write(
            history_key, history_value, history_mask, mode=mode
        )
        expanded_query = query[:, None].expand_as(canonical_candidates)
        read_key, candidate_value = self._encode_pairs(
            expanded_query, canonical_candidates
        )
        adapted_prediction = torch.einsum("bvk,bck->bcv", fast_weight, read_key)
        initial_prediction = torch.einsum(
            "vk,bck->bcv", self.initial_weight, read_key
        )
        adapted_loss = 0.5 * (
            adapted_prediction - candidate_value
        ).square().mean(dim=-1)
        initial_loss = 0.5 * (
            initial_prediction - candidate_value
        ).square().mean(dim=-1)
        raw = (initial_loss - adapted_loss) * canonical_mask.to(initial_loss.dtype)
        weight = canonical_mask.to(raw.dtype)
        centered = raw - (raw * weight).sum(dim=-1, keepdim=True) / weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0)
        active = (
            (history_mask.sum(dim=-1) >= 2)
            & query_present.bool()
            & ~repeat_request.bool()
        )
        correction = centered * weight * active[:, None].to(centered.dtype)
        scores = canonical_base.float() + correction
        scores = torch.where(
            repeat_request[:, None].bool(), canonical_item.float(), scores
        ).masked_fill(~canonical_mask.bool(), 0.0)
        restored_scores = self._gather(scores, inverse)
        restored_correction = self._gather(correction, inverse)
        restored_raw = self._gather(raw, inverse)
        return FastWeightOutput(
            scores=restored_scores,
            correction=restored_correction,
            raw_delta=restored_raw,
            event_weight=event_weight,
            event_utility=utility,
            fast_weight=fast_weight,
            active_request=active,
        )
