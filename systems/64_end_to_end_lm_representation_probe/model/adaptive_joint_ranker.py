"""C64 late-layer adaptive LM plus directed joint-context ranker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "adaptive_history_lm",
    "adaptive_query_candidate_lm",
    "frozen_history_lm",
)


@dataclass(frozen=True)
class AdaptiveLMOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    raw_correction: torch.Tensor
    candidate_state: torch.Tensor
    active_request: torch.Tensor


def _safe_padding(mask: torch.Tensor) -> torch.Tensor:
    safe = mask.bool().clone()
    empty = ~safe.any(dim=-1)
    if bool(empty.any()):
        safe[empty, 0] = True
    return safe


def listwise_loss(
    output: AdaptiveLMOutput,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    correction_l2_weight: float,
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
    energy = (
        output.correction[active].square().mean()
        if bool(active.any())
        else output.correction.sum() * 0.0
    )
    total = ranking + float(correction_l2_weight) * energy
    return total, {"ranking": ranking, "correction_l2": energy}


class AdaptiveJointLMRanker(nn.Module):
    """Adapt late LM layers, then jointly contextualize history and candidates."""

    def __init__(
        self,
        *,
        backbone: nn.Module,
        mode: str,
        trainable_last_lm_layers: int,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        ffn_dim: int,
        dropout: float,
        max_history: int,
        zero_initial_output: bool,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C64 mode: {mode}")
        if hidden_dim % heads:
            raise ValueError("C64 hidden dimension must divide heads")
        self.backbone = backbone
        self.mode = mode
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_history = int(max_history)
        self.trainable_last_lm_layers = int(trainable_last_lm_layers)
        if int(getattr(backbone.config, "hidden_size")) != self.input_dim:
            raise ValueError("C64 backbone hidden width differs")
        self._configure_backbone()

        self.content_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.base_projection = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim)
        )
        self.query_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_type = nn.Parameter(torch.empty(hidden_dim))
        self.candidate_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_position = nn.Parameter(torch.empty(max_history, hidden_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.joint_transformer = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters(zero_initial_output=zero_initial_output)

    def _backbone_layers(self) -> list[nn.Module]:
        encoder = getattr(self.backbone, "encoder", None)
        layers = getattr(encoder, "layer", None)
        if layers is None:
            raise ValueError("C64 backbone does not expose encoder.layer")
        return list(layers)

    def _configure_backbone(self) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        if self.mode == "frozen_history_lm":
            return
        layers = self._backbone_layers()
        if not 0 < self.trainable_last_lm_layers <= len(layers):
            raise ValueError("C64 trainable LM layer count differs")
        for layer in layers[-self.trainable_last_lm_layers :]:
            for parameter in layer.parameters():
                parameter.requires_grad_(True)

    def reset_parameters(self, *, zero_initial_output: bool) -> None:
        for value in (
            self.query_type,
            self.history_type,
            self.candidate_type,
            self.history_position,
        ):
            nn.init.normal_(value, std=0.02)
        if zero_initial_output:
            nn.init.zeros_(self.output_head.weight)
        else:
            nn.init.normal_(self.output_head.weight, std=0.02)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def backbone_trainable_names(self) -> list[str]:
        return [name for name, value in self.backbone.named_parameters() if value.requires_grad]

    def _encode_flat(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        content_mask: torch.Tensor,
    ) -> torch.Tensor:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("C64 token shape differs")
        if content_mask.shape != input_ids.shape:
            raise ValueError("C64 content mask differs")
        context = torch.no_grad() if self.mode == "frozen_history_lm" else _null_context()
        with context:
            output: Any = self.backbone(
                input_ids=input_ids.long(), attention_mask=attention_mask.long()
            )
            states = output.last_hidden_state
        weight = content_mask.to(states.dtype)
        return (states * weight[..., None]).sum(dim=1) / weight.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)

    def _encode_group(
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
            values = self._encode_flat(
                input_ids.reshape(-1, length)[flat_valid],
                attention_mask.reshape(-1, length)[flat_valid],
                content_mask.reshape(-1, length)[flat_valid],
            )
            output[flat_valid] = values.float()
        return output.reshape(*prefix, self.input_dim)

    @staticmethod
    def attention_mask(
        history_slots: int, candidate_slots: int, *, device: torch.device
    ) -> torch.Tensor:
        context = 1 + history_slots
        length = context + candidate_slots
        mask = torch.zeros(length, length, dtype=torch.bool, device=device)
        mask[:context, context:] = True
        return mask

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
        query_present: torch.Tensor | None = None,
    ) -> AdaptiveLMOutput:
        batch = query_input_ids.shape[0]
        if candidate_input_ids.shape[:2] != candidate_mask.shape:
            raise ValueError("C64 candidate mask shape differs")
        if history_input_ids.shape[:2] != history_event_mask.shape:
            raise ValueError("C64 history mask shape differs")
        if history_input_ids.shape[1] > self.max_history:
            raise ValueError("C64 history exceeds maximum")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != base_scores.shape:
            raise ValueError("C64 score shape differs")
        if query_present is None:
            query_present = torch.ones(batch, dtype=torch.bool, device=query_input_ids.device)

        query = self._encode_flat(
            query_input_ids, query_attention_mask, query_content_mask
        )
        candidates = self._encode_group(
            candidate_input_ids,
            candidate_attention_mask,
            candidate_content_mask,
            candidate_mask,
        )
        if self.mode == "adaptive_query_candidate_lm":
            history = torch.zeros(
                *history_event_mask.shape,
                self.input_dim,
                device=query.device,
                dtype=query.dtype,
            )
            effective_history_mask = torch.zeros_like(history_event_mask)
        else:
            history = self._encode_group(
                history_input_ids,
                history_attention_mask,
                history_content_mask,
                history_event_mask,
            )
            effective_history_mask = history_event_mask.bool()

        q = self.content_projection(F.normalize(query, dim=-1, eps=1e-6)) + self.query_type
        h = self.content_projection(F.normalize(history, dim=-1, eps=1e-6))
        h = h + self.history_type + self.history_position[: history.shape[1]][None]
        c = self.content_projection(F.normalize(candidates, dim=-1, eps=1e-6))
        c = c + self.candidate_type + self.base_projection(base_scores.float()[..., None])
        sequence = torch.cat((q[:, None], h, c), dim=1)
        padding = torch.cat(
            (
                torch.zeros(batch, 1, dtype=torch.bool, device=q.device),
                ~effective_history_mask,
                ~candidate_mask.bool(),
            ),
            dim=1,
        )
        encoded = self.joint_transformer(
            sequence,
            mask=self.attention_mask(history.shape[1], candidates.shape[1], device=q.device),
            src_key_padding_mask=padding,
        )
        candidate_state = encoded[:, 1 + history.shape[1] :]
        raw = self.output_head(self.output_norm(candidate_state)).squeeze(-1)
        weight = candidate_mask.to(raw.dtype)
        mean = (raw * weight).sum(-1, keepdim=True) / weight.sum(-1, keepdim=True).clamp_min(1.0)
        if self.mode == "adaptive_query_candidate_lm":
            active = query_present.bool()
        else:
            active = effective_history_mask.any(-1) & query_present.bool()
        active = active & ~repeat_request.bool()
        correction = (raw - mean) * weight * active[:, None].to(raw.dtype)
        scores = base_scores.float() + correction
        scores = torch.where(repeat_request[:, None].bool(), item_only_scores.float(), scores)
        scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        return AdaptiveLMOutput(
            scores=scores,
            correction=correction,
            raw_correction=raw,
            candidate_state=candidate_state,
            active_request=active,
        )


class _null_context:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        return False
