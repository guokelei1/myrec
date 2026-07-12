"""Pretrained token-level C74 ranker with masked query WordPiece relay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


PRIMARY = "semantic_conservative_relay"
MODES = (
    PRIMARY,
    "coupled_value_relay",
    "pooled_semantic_relay",
    "factual_semantic_relay",
)


@dataclass(frozen=True)
class AdaptiveSemanticRelayOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    history_attention: torch.Tensor
    factual_candidate_attention: torch.Tensor
    null_candidate_attention: torch.Tensor
    factual_query_state: torch.Tensor
    active_request: torch.Tensor


def listwise_loss(
    output: AdaptiveSemanticRelayOutput,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(-1) > 0
    target = target / target.sum(-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1).masked_fill(~mask, 0.0)
    row = -(target * log_probability).sum(-1)
    return row[valid].mean() if bool(valid.any()) else output.scores.sum() * 0.0


class _ResidualRoute(nn.Module):
    def __init__(self, dim: int, rank: int, init_std: float) -> None:
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        nn.init.normal_(self.down.weight, std=init_std)
        nn.init.zeros_(self.up.weight)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.normalize(value + self.up(self.down(value)), dim=-1, eps=1e-6)


class AdaptiveSemanticRelayLMRanker(nn.Module):
    def __init__(
        self,
        *,
        backbone: nn.Module,
        mode: str,
        trainable_last_lm_layers: int,
        input_dim: int,
        route_rank: int,
        max_history: int,
        temperature: float,
        profile_scale: float,
        correction_scale: float,
        route_init_std: float,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C74 LM mode: {mode}")
        if int(getattr(backbone.config, "hidden_size")) != int(input_dim):
            raise ValueError("C74 LM hidden width differs")
        self.backbone = backbone
        self.mode = mode
        self.input_dim = int(input_dim)
        self.max_history = int(max_history)
        self.trainable_last_lm_layers = int(trainable_last_lm_layers)
        self.temperature = float(temperature)
        self.profile_scale = float(profile_scale)
        self.correction_scale = float(correction_scale)
        self.history_route = _ResidualRoute(input_dim, route_rank, route_init_std)
        self.candidate_route = _ResidualRoute(input_dim, route_rank, route_init_std)
        self.chronology_bias = nn.Parameter(torch.zeros(max_history))
        self._configure_backbone()

    def _backbone_layers(self) -> list[nn.Module]:
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise ValueError("C74 backbone does not expose encoder.layer")
        return list(layers)

    def _configure_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        layers = self._backbone_layers()
        if not 0 < self.trainable_last_lm_layers <= len(layers):
            raise ValueError("C74 trainable LM layer count differs")
        for layer in layers[-self.trainable_last_lm_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def backbone_trainable_names(self) -> list[str]:
        return [name for name, value in self.backbone.named_parameters() if value.requires_grad]

    def _encode_flat_tokens(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        output: Any = self.backbone(
            input_ids=input_ids.long(), attention_mask=attention_mask.long()
        )
        return output.last_hidden_state.float()

    def _encode_group_mean(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        prefix = input_ids.shape[:-1]
        length = input_ids.shape[-1]
        flat_valid = valid_mask.reshape(-1).bool()
        output = torch.zeros(
            flat_valid.numel(), self.input_dim, device=input_ids.device, dtype=torch.float32
        )
        if bool(flat_valid.any()):
            states = self._encode_flat_tokens(
                input_ids.reshape(-1, length)[flat_valid],
                attention_mask.reshape(-1, length)[flat_valid],
            )
            weight = content_mask.reshape(-1, length)[flat_valid].float()
            pooled = (states * weight[..., None]).sum(1) / weight.sum(
                1, keepdim=True
            ).clamp_min(1.0)
            output[flat_valid] = pooled
        return output.reshape(*prefix, self.input_dim)

    def _core(
        self,
        *,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
        candidate_mask: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        raw_query = F.normalize(query, dim=-1, eps=1e-6)
        raw_history = F.normalize(history, dim=-1, eps=1e-6)
        raw_candidates = F.normalize(candidates, dim=-1, eps=1e-6)
        route_query = self.history_route(raw_query)
        route_history = self.history_route(raw_history)
        history_logits = torch.einsum("bqd,bhd->bqh", route_query, route_history)
        history_logits = history_logits / self.temperature
        history_logits = history_logits + self.chronology_bias[: history.shape[1]][None, None]
        history_logits = history_logits.masked_fill(~history_mask[:, None].bool(), -1e9)
        history_attention = torch.softmax(history_logits, dim=-1)
        coupled = self.mode == "coupled_value_relay"
        pooled = self.mode == "pooled_semantic_relay"
        factual_only = self.mode == "factual_semantic_relay"
        carrier_query = route_query if coupled else raw_query
        carrier_history = route_history if coupled else raw_history
        profile = torch.einsum("bqh,bhd->bqd", history_attention, carrier_history)
        if pooled:
            weight = query_mask.float()
            pooled_profile = (profile * weight[..., None]).sum(1, keepdim=True) / weight.sum(
                1, keepdim=True
            )[..., None].clamp_min(1.0)
            profile = pooled_profile.expand_as(profile)
        factual_query = F.normalize(
            carrier_query
            + self.profile_scale * profile * active[:, None, None].float(),
            dim=-1,
            eps=1e-6,
        )
        null_query = route_query if coupled else raw_query
        route_candidates = self.candidate_route(raw_candidates)
        route_factual = self.candidate_route(factual_query)
        route_null = self.candidate_route(null_query)
        factual_logits = torch.einsum(
            "bcd,bqd->bcq", route_candidates, route_factual
        ) / self.temperature
        null_logits = torch.einsum(
            "bcd,bqd->bcq", route_candidates, route_null
        ) / self.temperature
        factual_logits = factual_logits.masked_fill(~query_mask[:, None].bool(), -1e9)
        null_logits = null_logits.masked_fill(~query_mask[:, None].bool(), -1e9)
        factual_attention = torch.softmax(factual_logits, dim=-1)
        null_attention = torch.softmax(null_logits, dim=-1)
        if coupled:
            value_candidates = route_candidates
            value_factual = route_factual
            value_null = route_null
        else:
            value_candidates = raw_candidates
            value_factual = factual_query
            value_null = raw_query
        factual_similarity = torch.einsum(
            "bcd,bqd->bcq", value_candidates, value_factual
        )
        null_similarity = torch.einsum(
            "bcd,bqd->bcq", value_candidates, value_null
        )
        factual_energy = (factual_attention * factual_similarity).sum(-1)
        null_energy = (null_attention * null_similarity).sum(-1)
        raw = factual_energy if factual_only else factual_energy - null_energy
        raw = raw.masked_fill(~candidate_mask.bool(), 0.0)
        return raw, history_attention, factual_attention, null_attention, factual_query

    def forward(
        self,
        *,
        query_input_ids: torch.Tensor,
        query_attention_mask: torch.Tensor,
        query_content_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,
        candidate_attention_mask: torch.Tensor,
        candidate_content_mask: torch.Tensor,
        history_input_ids: torch.Tensor,
        history_attention_mask: torch.Tensor,
        history_content_mask: torch.Tensor,
        history_event_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
        query_present: torch.Tensor,
    ) -> AdaptiveSemanticRelayOutput:
        if candidate_input_ids.shape[:2] != candidate_mask.shape:
            raise ValueError("C74 LM candidate shape differs")
        if history_input_ids.shape[:2] != history_event_mask.shape:
            raise ValueError("C74 LM history shape differs")
        if history_input_ids.shape[1] > self.max_history:
            raise ValueError("C74 LM history exceeds maximum")
        query = self._encode_flat_tokens(query_input_ids, query_attention_mask)
        query_mask = query_content_mask.bool()
        history = self._encode_group_mean(
            history_input_ids,
            history_attention_mask,
            history_content_mask,
            history_event_mask,
        )
        candidates = self._encode_group_mean(
            candidate_input_ids,
            candidate_attention_mask,
            candidate_content_mask,
            candidate_mask,
        )
        active = history_event_mask.bool().any(-1) & query_present.bool()
        with torch.autocast(device_type=query.device.type, enabled=False):
            raw, history_attention, factual_attention, null_attention, factual_query = self._core(
                query=query.float(),
                query_mask=query_mask,
                history=history.float(),
                history_mask=history_event_mask,
                candidates=candidates.float(),
                candidate_mask=candidate_mask,
                active=active,
            )
            bounded = self.correction_scale * torch.tanh(raw)
            weight = candidate_mask.float()
            mean = (bounded * weight).sum(-1, keepdim=True) / weight.sum(
                -1, keepdim=True
            ).clamp_min(1.0)
            active = active & ~repeat_request.bool()
            correction = (bounded - mean) * weight * active[:, None].float()
            scores = base_scores.float() + correction
            scores = torch.where(
                repeat_request[:, None].bool(), item_only_scores.float(), scores
            )
            scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        return AdaptiveSemanticRelayOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            history_attention=history_attention,
            factual_candidate_attention=factual_attention,
            null_candidate_attention=null_attention,
            factual_query_state=factual_query,
            active_request=active,
        )
