"""C63 finite-evidence event-to-memory Transformer."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "finite_evidence_memory",
    "slot_competition_memory",
    "balanced_transport_memory",
    "standard_slot_memory",
    "single_pooled_memory",
)


@dataclass(frozen=True)
class FiniteMemoryOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    memory: torch.Tensor
    allocation: torch.Tensor
    null_mass: torch.Tensor
    slot_mass: torch.Tensor
    candidate_state: torch.Tensor
    active_request: torch.Tensor


def _safe_padding(mask: torch.Tensor) -> torch.Tensor:
    safe = mask.bool().clone()
    empty = ~safe.any(dim=-1)
    if bool(empty.any()):
        safe[empty, 0] = True
    return safe


def _masked_mean(value: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weight = mask.to(value.dtype)
    while weight.ndim < value.ndim:
        weight = weight.unsqueeze(-1)
    return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def listwise_training_loss(
    output: FiniteMemoryOutput,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    wrong_output: FiniteMemoryOutput | None = None,
    base_scores: torch.Tensor | None = None,
    listwise_weight: float = 1.0,
    wrong_base_kl_weight: float = 0.25,
    correction_l2_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(dim=-1) > 0
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1)
    row_loss = -(target * log_probability.masked_fill(~mask, 0.0)).sum(dim=-1)
    ranking = row_loss[valid].mean() if bool(valid.any()) else logits.sum() * 0.0
    active = output.active_request[:, None] & mask
    energy = (
        output.correction[active].square().mean()
        if bool(active.any())
        else output.correction.sum() * 0.0
    )
    wrong_kl = output.correction.sum() * 0.0
    if wrong_output is not None:
        if base_scores is None:
            raise ValueError("C63 wrong-history KL requires base scores")
        base = base_scores.float().masked_fill(~mask, -torch.inf)
        wrong = wrong_output.scores.float().masked_fill(~mask, -torch.inf)
        base_probability = F.softmax(base, dim=-1).detach()
        base_log_probability = F.log_softmax(base, dim=-1).detach()
        wrong_log_probability = F.log_softmax(wrong, dim=-1)
        row_kl = (
            base_probability
            * (base_log_probability - wrong_log_probability).masked_fill(~mask, 0.0)
        ).sum(dim=-1)
        wrong_kl = row_kl[valid].mean() if bool(valid.any()) else wrong.sum() * 0.0
    total = (
        float(listwise_weight) * ranking
        + float(wrong_base_kl_weight) * wrong_kl
        + float(correction_l2_weight) * energy
    )
    return total, {"ranking": ranking, "wrong_base_kl": wrong_kl, "correction_l2": energy}


class FiniteEvidenceMemoryTransformer(nn.Module):
    """Allocate each history event's finite evidence across memory destinations."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        history_layers: int,
        candidate_layers: int,
        memory_slots: int,
        max_history: int,
        sinkhorn_iterations: int,
        dropout: float,
        zero_initial_output: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("C63 hidden dimension must divide heads")
        if memory_slots < 2 or sinkhorn_iterations < 1:
            raise ValueError("C63 memory/sinkhorn setting is invalid")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.memory_slots = int(memory_slots)
        self.max_history = int(max_history)
        self.sinkhorn_iterations = int(sinkhorn_iterations)

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.base_projection = nn.Linear(1, hidden_dim, bias=False)
        self.event_key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.event_value_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.slot_key_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.query_type = nn.Parameter(torch.empty(hidden_dim))
        self.candidate_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_position = nn.Parameter(torch.empty(max_history, hidden_dim))
        self.slot_seed = nn.Parameter(torch.empty(memory_slots, hidden_dim))
        self.break_bias = nn.Parameter(torch.empty(memory_slots))

        history_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.history_transformer = nn.TransformerEncoder(
            history_layer,
            num_layers=history_layers,
            enable_nested_tensor=False,
        )
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.memory_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.memory_output_norm = nn.LayerNorm(hidden_dim)
        self.memory_read_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.read_norm = nn.LayerNorm(hidden_dim)
        self.read_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        candidate_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.candidate_transformer = nn.TransformerEncoder(
            candidate_layer,
            num_layers=candidate_layers,
            enable_nested_tensor=False,
        )
        self.score_norm = nn.LayerNorm(hidden_dim)
        self.score_head = nn.Linear(hidden_dim, 1, bias=False)
        self.reset_parameters(zero_initial_output=zero_initial_output)

    def reset_parameters(self, *, zero_initial_output: bool) -> None:
        for value in (
            self.query_type,
            self.candidate_type,
            self.history_type,
            self.history_position,
            self.slot_seed,
        ):
            nn.init.normal_(value, std=0.02)
        # Equal expected prior mass for S real slots and one NULL destination.
        remaining = self.memory_slots + 1
        with torch.no_grad():
            for slot in range(self.memory_slots):
                probability = 1.0 / float(remaining - slot)
                self.break_bias[slot] = math.log(probability / (1.0 - probability))
        if zero_initial_output:
            nn.init.zeros_(self.score_head.weight)
        else:
            nn.init.normal_(self.score_head.weight, std=0.02)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def _encode_history(
        self, history: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        if history.ndim != 3 or history.shape[:2] != history_mask.shape:
            raise ValueError("C63 history shape differs")
        if history.shape[-1] != self.input_dim or history.shape[1] > self.max_history:
            raise ValueError("C63 history width/length differs")
        safe = _safe_padding(history_mask)
        value = (
            self.input_projection(history.float())
            + self.history_type
            + self.history_position[: history.shape[1]][None]
        )
        encoded = self.history_transformer(value, src_key_padding_mask=~safe)
        return encoded * history_mask[..., None].to(encoded.dtype)

    def _allocation_logits(self, history_state: torch.Tensor) -> torch.Tensor:
        keys = F.normalize(self.event_key_projection(history_state), dim=-1, eps=1e-6)
        slots = F.normalize(
            self.slot_key_projection(self.slot_seed), dim=-1, eps=1e-6
        )
        return torch.einsum("bhd,sd->bhs", keys, slots) / math.sqrt(
            float(self.hidden_dim)
        ) + self.break_bias

    def _finite_allocation(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        beta = torch.sigmoid(logits)
        one = torch.ones(*beta.shape[:-1], 1, dtype=beta.dtype, device=beta.device)
        survival = torch.cumprod(
            torch.cat((one, 1.0 - beta), dim=-1), dim=-1
        )
        allocation = beta * survival[..., :-1]
        null = survival[..., -1]
        weight = mask.to(allocation.dtype)
        return allocation * weight[..., None], null * weight

    def _slot_competition(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allocation = torch.softmax(logits, dim=-1) * mask[..., None].to(logits.dtype)
        return allocation, torch.zeros_like(mask, dtype=logits.dtype)

    def _balanced_transport(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        allocation = torch.exp(logits - logits.amax(dim=(-2, -1), keepdim=True))
        allocation = allocation * mask[..., None].to(logits.dtype)
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(logits.dtype)
        target_column = count / float(self.memory_slots)
        for _ in range(self.sinkhorn_iterations):
            allocation = allocation / allocation.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            allocation = allocation * mask[..., None].to(logits.dtype)
            allocation = allocation / allocation.sum(dim=-2, keepdim=True).clamp_min(1e-8)
            allocation = allocation * target_column[..., None]
        allocation = allocation / allocation.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        allocation = allocation * mask[..., None].to(logits.dtype)
        return allocation, torch.zeros_like(mask, dtype=logits.dtype)

    def _standard_slots(
        self, logits: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        safe = _safe_padding(mask)
        slot_event = logits.transpose(1, 2).masked_fill(~safe[:, None], -torch.inf)
        allocation = torch.softmax(slot_event, dim=-1).transpose(1, 2)
        allocation = allocation * mask[..., None].to(logits.dtype)
        return allocation, torch.zeros_like(mask, dtype=logits.dtype)

    def build_memory(
        self,
        *,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        query: torch.Tensor,
        mode: str = "finite_evidence_memory",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if mode not in MODES:
            raise ValueError(f"unknown C63 mode: {mode}")
        if query.ndim != 2 or query.shape[-1] != self.input_dim:
            raise ValueError("C63 query shape differs")
        history_state = self._encode_history(history, history_mask)
        logits = self._allocation_logits(history_state)
        if mode in {"finite_evidence_memory", "single_pooled_memory"}:
            allocation, null = self._finite_allocation(logits, history_mask)
        elif mode == "slot_competition_memory":
            allocation, null = self._slot_competition(logits, history_mask)
        elif mode == "balanced_transport_memory":
            allocation, null = self._balanced_transport(logits, history_mask)
        else:
            allocation, null = self._standard_slots(logits, history_mask)
        values = self.event_value_projection(history_state)
        slot_mass = allocation.sum(dim=1)
        aggregated = torch.einsum("bhs,bhd->bsd", allocation, values)
        aggregated = aggregated / slot_mass[..., None].clamp_min(1e-8)
        memory = self.memory_norm(self.slot_seed[None] + aggregated)
        memory = self.memory_output_norm(memory + self.memory_ffn(memory))
        present = history_mask.any(dim=-1)[:, None, None]
        memory = memory * present.to(memory.dtype)
        if mode == "single_pooled_memory":
            memory = memory.mean(dim=1, keepdim=True).expand(-1, self.memory_slots, -1)
        return memory, allocation, null, slot_mass

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        item_only_scores: torch.Tensor,
        repeat_request: torch.Tensor,
        query_present: torch.Tensor | None = None,
        mode: str = "finite_evidence_memory",
    ) -> FiniteMemoryOutput:
        if candidates.ndim != 3 or candidates.shape[:2] != candidate_mask.shape:
            raise ValueError("C63 candidate shape differs")
        if candidates.shape[-1] != self.input_dim:
            raise ValueError("C63 candidate width differs")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != base_scores.shape:
            raise ValueError("C63 score shape differs")
        if query_present is None:
            query_present = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)
        query_state = self.input_projection(query.float()) + self.query_type
        candidate_state = (
            self.input_projection(candidates.float())
            + self.candidate_type
            + self.query_projection(query_state)[:, None]
            + self.base_projection(base_scores.float()[..., None])
        )
        memory, allocation, null, slot_mass = self.build_memory(
            history=history,
            history_mask=history_mask,
            query=query,
            mode=mode,
        )
        memory_mask = history_mask.any(dim=-1)[:, None].expand(-1, self.memory_slots)
        read, _ = self.memory_read_attention(
            candidate_state,
            memory,
            memory,
            key_padding_mask=~_safe_padding(memory_mask),
            need_weights=False,
        )
        history_present = history_mask.any(dim=-1)
        read = read * history_present[:, None, None].to(read.dtype)
        state = self.read_norm(candidate_state + read)
        state = self.read_norm(state + self.read_ffn(state))
        state = self.candidate_transformer(
            state, src_key_padding_mask=~_safe_padding(candidate_mask)
        )
        state = self.score_norm(state)
        raw = self.score_head(state).squeeze(-1) * candidate_mask.to(state.dtype)
        correction = raw - _masked_mean(raw, candidate_mask, dim=1)[:, None]
        correction = correction * candidate_mask.to(correction.dtype)
        active = history_present & query_present.bool() & ~repeat_request.bool()
        correction = correction * active[:, None].to(correction.dtype)
        scores = base_scores.float() + correction
        scores = torch.where(repeat_request[:, None].bool(), item_only_scores.float(), scores)
        scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        return FiniteMemoryOutput(
            scores=scores,
            correction=correction,
            memory=memory,
            allocation=allocation,
            null_mass=null,
            slot_mass=slot_mass,
            candidate_state=state,
            active_request=active,
        )
