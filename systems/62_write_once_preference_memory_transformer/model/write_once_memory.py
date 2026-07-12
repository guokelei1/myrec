"""C62 write-once preference-memory Transformer.

The primary attention graph has two phases.  History tokens first write latent
memory slots without query/candidate access.  Candidate states then read the
immutable slots and interact listwise.  Ranking gradients still train both
phases end to end; immutability concerns the forward information graph.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


MODES = (
    "write_once_memory",
    "query_conditioned_writer",
    "direct_history_attention",
    "single_pooled_slot",
)


@dataclass(frozen=True)
class MemoryRankingOutput:
    scores: torch.Tensor
    correction: torch.Tensor
    memory: torch.Tensor
    memory_mask: torch.Tensor
    candidate_state: torch.Tensor
    read_state: torch.Tensor
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
    output: MemoryRankingOutput,
    labels: torch.Tensor,
    candidate_mask: torch.Tensor,
    *,
    wrong_output: MemoryRankingOutput | None = None,
    base_scores: torch.Tensor | None = None,
    listwise_weight: float = 1.0,
    wrong_base_kl_weight: float = 0.25,
    correction_l2_weight: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Shared listwise objective for every C62 mode and domain."""

    mask = candidate_mask.bool()
    target = labels.float().clamp_min(0.0) * mask.to(labels.dtype)
    valid = target.sum(dim=-1) > 0
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0)
    logits = output.scores.float().masked_fill(~mask, -torch.inf)
    log_probability = F.log_softmax(logits, dim=-1)
    row_loss = -(target * log_probability.masked_fill(~mask, 0.0)).sum(dim=-1)
    ranking = row_loss[valid].mean() if bool(valid.any()) else logits.sum() * 0.0

    active = output.active_request[:, None] & mask
    if bool(active.any()):
        energy = output.correction[active].square().mean()
    else:
        energy = output.correction.sum() * 0.0

    wrong_kl = output.correction.sum() * 0.0
    if wrong_output is not None:
        if base_scores is None:
            raise ValueError("C62 wrong-history KL requires base scores")
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


class WriteOncePreferenceMemoryTransformer(nn.Module):
    """Two-phase latent-memory listwise Transformer ranker."""

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
        dropout: float,
        zero_initial_output: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("C62 hidden dimension must divide heads")
        if memory_slots < 2:
            raise ValueError("C62 primary requires at least two memory slots")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.memory_slots = int(memory_slots)
        self.max_history = int(max_history)

        self.input_projection = nn.Linear(input_dim, hidden_dim, bias=False)
        self.query_projection = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.base_projection = nn.Linear(1, hidden_dim, bias=False)
        self.query_type = nn.Parameter(torch.empty(hidden_dim))
        self.candidate_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_type = nn.Parameter(torch.empty(hidden_dim))
        self.history_position = nn.Parameter(torch.empty(max_history, hidden_dim))
        self.slot_seed = nn.Parameter(torch.empty(memory_slots, hidden_dim))

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
        self.slot_write_attention = nn.MultiheadAttention(
            hidden_dim, heads, dropout=dropout, bias=False, batch_first=True
        )
        self.slot_write_norm = nn.LayerNorm(hidden_dim)
        self.slot_write_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.slot_output_norm = nn.LayerNorm(hidden_dim)

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
            raise ValueError("C62 history shape differs")
        if history.shape[-1] != self.input_dim:
            raise ValueError("C62 history input width differs")
        if history.shape[1] > self.max_history:
            raise ValueError("C62 history exceeds maximum")
        safe = _safe_padding(history_mask)
        value = (
            self.input_projection(history.float())
            + self.history_type
            + self.history_position[: history.shape[1]][None]
        )
        encoded = self.history_transformer(value, src_key_padding_mask=~safe)
        return encoded * history_mask[..., None].to(encoded.dtype)

    def _write_slots(
        self,
        history_state: torch.Tensor,
        history_mask: torch.Tensor,
        query_state: torch.Tensor,
        *,
        query_conditioned: bool,
    ) -> torch.Tensor:
        batch = history_state.shape[0]
        slots = self.slot_seed[None].expand(batch, -1, -1)
        if query_conditioned:
            slots = slots + self.query_projection(query_state)[:, None]
        safe = _safe_padding(history_mask)
        written, _ = self.slot_write_attention(
            slots,
            history_state,
            history_state,
            key_padding_mask=~safe,
            need_weights=False,
        )
        present = history_mask.any(dim=-1)[:, None, None]
        state = self.slot_write_norm(slots + written)
        state = self.slot_output_norm(state + self.slot_write_ffn(state))
        return state * present.to(state.dtype)

    def build_memory(
        self,
        *,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        query: torch.Tensor,
        mode: str = "write_once_memory",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return memory, memory mask, and encoded history for interventions."""

        if mode not in MODES:
            raise ValueError(f"unknown C62 mode: {mode}")
        if query.ndim != 2 or query.shape[-1] != self.input_dim:
            raise ValueError("C62 query shape differs")
        query_state = self.input_projection(query.float()) + self.query_type
        history_state = self._encode_history(history, history_mask)
        query_conditioned = mode == "query_conditioned_writer"
        slots = self._write_slots(
            history_state,
            history_mask,
            query_state,
            query_conditioned=query_conditioned,
        )
        present = history_mask.any(dim=-1)
        if mode == "single_pooled_slot":
            pooled = slots.mean(dim=1, keepdim=True)
            slots = pooled.expand(-1, self.memory_slots, -1)
        if mode == "direct_history_attention":
            # Reuse the writer block as an equal-capacity history transformation,
            # then expose event states directly to the candidate reader.
            safe = _safe_padding(history_mask)
            transformed, _ = self.slot_write_attention(
                history_state,
                history_state,
                history_state,
                key_padding_mask=~safe,
                need_weights=False,
            )
            transformed = self.slot_output_norm(
                self.slot_write_norm(history_state + transformed)
                + self.slot_write_ffn(self.slot_write_norm(history_state + transformed))
            )
            memory = transformed * history_mask[..., None].to(transformed.dtype)
            memory_mask = history_mask.bool()
        else:
            memory = slots
            memory_mask = present[:, None].expand(-1, self.memory_slots)
        return memory, memory_mask, history_state

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
        mode: str = "write_once_memory",
    ) -> MemoryRankingOutput:
        if mode not in MODES:
            raise ValueError(f"unknown C62 mode: {mode}")
        if candidates.ndim != 3 or candidates.shape[:2] != candidate_mask.shape:
            raise ValueError("C62 candidate shape differs")
        if candidates.shape[-1] != self.input_dim:
            raise ValueError("C62 candidate input width differs")
        if base_scores.shape != candidate_mask.shape or item_only_scores.shape != base_scores.shape:
            raise ValueError("C62 score shape differs")
        if repeat_request.shape != base_scores.shape[:1]:
            raise ValueError("C62 repeat shape differs")
        if query_present is None:
            query_present = torch.ones(
                query.shape[0], dtype=torch.bool, device=query.device
            )
        query_present = query_present.bool()

        query_state = self.input_projection(query.float()) + self.query_type
        candidate_state = (
            self.input_projection(candidates.float())
            + self.candidate_type
            + self.query_projection(query_state)[:, None]
            + self.base_projection(base_scores.float()[..., None])
        )
        memory, memory_mask, _ = self.build_memory(
            history=history,
            history_mask=history_mask,
            query=query,
            mode=mode,
        )
        safe_memory = _safe_padding(memory_mask)
        read, _ = self.memory_read_attention(
            candidate_state,
            memory,
            memory,
            key_padding_mask=~safe_memory,
            need_weights=False,
        )
        history_present = history_mask.any(dim=-1)
        read = read * history_present[:, None, None].to(read.dtype)
        read_state = self.read_norm(candidate_state + read)
        read_state = self.read_norm(read_state + self.read_ffn(read_state))
        safe_candidates = _safe_padding(candidate_mask)
        contextual = self.candidate_transformer(
            read_state,
            src_key_padding_mask=~safe_candidates,
        )
        contextual = self.score_norm(contextual)
        raw = self.score_head(contextual).squeeze(-1)
        raw = raw * candidate_mask.to(raw.dtype)
        mean = _masked_mean(raw, candidate_mask, dim=1)
        correction = (raw - mean[:, None]) * candidate_mask.to(raw.dtype)

        active = history_present & query_present & ~repeat_request.bool()
        correction = correction * active[:, None].to(correction.dtype)
        scores = base_scores.float() + correction
        scores = torch.where(repeat_request[:, None].bool(), item_only_scores.float(), scores)
        scores = scores.masked_fill(~candidate_mask.bool(), 0.0)
        return MemoryRankingOutput(
            scores=scores,
            correction=correction,
            memory=memory,
            memory_mask=memory_mask,
            candidate_state=contextual,
            read_state=read_state,
            active_request=active,
        )
