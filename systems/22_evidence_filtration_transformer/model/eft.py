"""Evidence-filtration Transformer used by the C22 synthetic falsifier."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


MODES = ("filtration", "dense", "parallel", "final_projection")


@dataclass(frozen=True)
class EFTOutput:
    scores: torch.Tensor
    base_scores: torch.Tensor
    recurrence_delta: torch.Tensor
    transfer_delta: torch.Tensor
    candidate_state: torch.Tensor


class PrefixRMSNorm(nn.Module):
    def __init__(self, block_dims: tuple[int, int, int], eps: float = 1e-6) -> None:
        super().__init__()
        self.block_dims = block_dims
        self.weight = nn.Parameter(torch.ones(sum(block_dims)))
        self.eps = float(eps)

    def forward(self, value: torch.Tensor, mode: str) -> torch.Tensor:
        if mode in ("dense", "final_projection"):
            denominator = value.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            return value * denominator * self.weight
        pieces = value.split(self.block_dims, dim=-1)
        output: list[torch.Tensor] = []
        start = 0
        prefix: list[torch.Tensor] = []
        for piece, width in zip(pieces, self.block_dims):
            prefix.append(piece)
            normalization_input = piece if mode == "parallel" else torch.cat(prefix, dim=-1)
            denominator = normalization_input.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
            output.append(piece * denominator * self.weight[start : start + width])
            start += width
        return torch.cat(output, dim=-1)


class OrderedAttention(nn.Module):
    def __init__(
        self,
        block_dims: tuple[int, int, int],
        heads_per_block: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.block_dims = block_dims
        self.heads_per_block = int(heads_per_block)
        self.dropout = float(dropout)
        total = sum(block_dims)
        for index, width in enumerate(block_dims):
            if width % heads_per_block:
                raise ValueError("each C22 block must divide its attention heads")
            setattr(self, f"qkv_{index}", nn.Linear(total, 3 * width, bias=False))
            setattr(self, f"out_{index}", nn.Linear(width, width, bias=False))

    def _allowed(self, value: torch.Tensor, block: int, mode: str) -> torch.Tensor:
        if mode in ("dense", "final_projection"):
            return value
        pieces = value.split(self.block_dims, dim=-1)
        if mode == "parallel":
            keep = {block}
        else:
            keep = set(range(block + 1))
        return torch.cat(
            [piece if index in keep else torch.zeros_like(piece) for index, piece in enumerate(pieces)],
            dim=-1,
        )

    def forward(self, value: torch.Tensor, token_mask: torch.Tensor, mode: str) -> torch.Tensor:
        batch, tokens, _ = value.shape
        outputs: list[torch.Tensor] = []
        for block, width in enumerate(self.block_dims):
            allowed = self._allowed(value, block, mode)
            qkv = getattr(self, f"qkv_{block}")(allowed)
            q, k, v = qkv.chunk(3, dim=-1)
            head_dim = width // self.heads_per_block

            def heads(tensor: torch.Tensor) -> torch.Tensor:
                return tensor.view(batch, tokens, self.heads_per_block, head_dim).transpose(1, 2)

            qh, kh, vh = heads(q), heads(k), heads(v)
            logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(head_dim)
            logits = logits.masked_fill(~token_mask[:, None, None, :], -torch.inf)
            attention = F.softmax(logits, dim=-1)
            attention = F.dropout(attention, p=self.dropout, training=self.training)
            mixed = torch.matmul(attention, vh).transpose(1, 2).contiguous().view(batch, tokens, width)
            outputs.append(getattr(self, f"out_{block}")(mixed))
        return torch.cat(outputs, dim=-1) * token_mask[:, :, None].to(value.dtype)


class OrderedFFN(nn.Module):
    def __init__(self, block_dims: tuple[int, int, int], multiplier: int) -> None:
        super().__init__()
        self.block_dims = block_dims
        total = sum(block_dims)
        for index, width in enumerate(block_dims):
            hidden = width * multiplier
            setattr(self, f"in_{index}", nn.Linear(total, hidden, bias=False))
            setattr(self, f"out_{index}", nn.Linear(hidden, width, bias=False))

    def _allowed(self, value: torch.Tensor, block: int, mode: str) -> torch.Tensor:
        if mode in ("dense", "final_projection"):
            return value
        pieces = value.split(self.block_dims, dim=-1)
        keep = {block} if mode == "parallel" else set(range(block + 1))
        return torch.cat(
            [piece if index in keep else torch.zeros_like(piece) for index, piece in enumerate(pieces)],
            dim=-1,
        )

    def forward(self, value: torch.Tensor, mode: str) -> torch.Tensor:
        output: list[torch.Tensor] = []
        for block in range(3):
            hidden = F.gelu(getattr(self, f"in_{block}")(self._allowed(value, block, mode)))
            output.append(getattr(self, f"out_{block}")(hidden))
        return torch.cat(output, dim=-1)


class EFTLayer(nn.Module):
    def __init__(
        self,
        block_dims: tuple[int, int, int],
        heads_per_block: int,
        ffn_multiplier: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = PrefixRMSNorm(block_dims)
        self.attention = OrderedAttention(block_dims, heads_per_block, dropout)
        self.ffn_norm = PrefixRMSNorm(block_dims)
        self.ffn = OrderedFFN(block_dims, ffn_multiplier)
        self.dropout = float(dropout)

    def forward(self, value: torch.Tensor, token_mask: torch.Tensor, mode: str) -> torch.Tensor:
        update = self.attention(self.attention_norm(value, mode), token_mask, mode)
        value = value + F.dropout(update, p=self.dropout, training=self.training)
        update = self.ffn(self.ffn_norm(value, mode), mode)
        value = value + F.dropout(update, p=self.dropout, training=self.training)
        return value * token_mask[:, :, None].to(value.dtype)


class EvidenceFiltrationRanker(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        anchor_dim: int,
        recurrence_dim: int,
        transfer_dim: int,
        history_slots: int,
        layers: int,
        heads_per_block: int,
        ffn_multiplier: int,
        dropout: float,
        transfer_delta_max: float,
        recurrence_scale_min: float,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C22 mode: {mode}")
        self.mode = mode
        self.input_dim = int(input_dim)
        self.history_slots = int(history_slots)
        self.block_dims = (int(anchor_dim), int(recurrence_dim), int(transfer_dim))
        self.transfer_delta_max = float(transfer_delta_max)
        self.recurrence_scale_min = float(recurrence_scale_min)
        self.anchor_input = nn.Linear(input_dim, anchor_dim, bias=False)
        self.transfer_input = nn.Linear(input_dim, transfer_dim, bias=False)
        self.recurrence_atom = nn.Parameter(torch.randn(recurrence_dim) / math.sqrt(recurrence_dim))
        self.role_embedding = nn.Parameter(torch.randn(3, transfer_dim) / math.sqrt(transfer_dim))
        self.position_embedding = nn.Parameter(
            torch.randn(history_slots + 2, transfer_dim) / math.sqrt(transfer_dim)
        )
        self.layers = nn.ModuleList(
            [
                EFTLayer(
                    self.block_dims,
                    heads_per_block,
                    ffn_multiplier,
                    dropout,
                )
                for _ in range(layers)
            ]
        )
        self.final_norm = PrefixRMSNorm(self.block_dims)
        self.recurrence_readout = nn.Linear(recurrence_dim, 1, bias=False)
        self.transfer_readout = nn.Linear(transfer_dim, 1, bias=False)
        self.recurrence_log_scale = nn.Parameter(torch.tensor(0.0))

    def _tokens(
        self,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        identity: torch.Tensor,
        event_strength: torch.Tensor,
        history_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, candidate_count, _ = candidates.shape
        history_count = history.shape[1]
        flat = batch * candidate_count
        query_rows = query[:, None, :].expand(-1, candidate_count, -1).reshape(flat, 1, self.input_dim)
        candidate_rows = candidates.reshape(flat, 1, self.input_dim)
        history_rows = history[:, None, :, :].expand(-1, candidate_count, -1, -1).reshape(
            flat, history_count, self.input_dim
        )
        raw = torch.cat([query_rows, candidate_rows, history_rows], dim=1)
        anchor = self.anchor_input(raw)
        transfer = self.transfer_input(raw)
        roles = torch.cat(
            [
                self.role_embedding[0][None, None, :],
                self.role_embedding[1][None, None, :],
                self.role_embedding[2][None, None, :].expand(1, history_count, -1),
            ],
            dim=1,
        )
        transfer = transfer + roles + self.position_embedding[: history_count + 2][None, :, :]

        identity_strength = identity.to(event_strength.dtype) * event_strength[:, None, :]
        flat_strength = identity_strength.reshape(flat, history_count)
        recurrence = raw.new_zeros(flat, history_count + 2, self.block_dims[1])
        recurrence[:, 1, :] = flat_strength.sum(dim=-1, keepdim=True) * self.recurrence_atom
        recurrence[:, 2:, :] = flat_strength[:, :, None] * self.recurrence_atom
        token_mask = torch.cat(
            [
                torch.ones(batch, 2, dtype=torch.bool, device=query.device),
                history_mask.bool(),
            ],
            dim=1,
        )
        token_mask = token_mask[:, None, :].expand(-1, candidate_count, -1).reshape(flat, history_count + 2)
        state = torch.cat([anchor, recurrence, transfer], dim=-1)
        exact_present = identity.any(dim=-1)
        return state, token_mask, exact_present

    def forward(
        self,
        *,
        query: torch.Tensor,
        candidates: torch.Tensor,
        history: torch.Tensor,
        identity: torch.Tensor,
        event_strength: torch.Tensor,
        history_mask: torch.Tensor,
        candidate_mask: torch.Tensor,
        base_scores: torch.Tensor,
        query_present: torch.Tensor | None = None,
    ) -> EFTOutput:
        if candidates.shape[:2] != candidate_mask.shape or base_scores.shape != candidate_mask.shape:
            raise ValueError("C22 candidate shape mismatch")
        if history.shape[:2] != history_mask.shape or identity.shape != (
            query.shape[0], candidates.shape[1], history.shape[1]
        ):
            raise ValueError("C22 history/relation shape mismatch")
        if event_strength.shape != history_mask.shape:
            raise ValueError("C22 event-strength shape mismatch")
        if query_present is None:
            query_present = torch.ones(query.shape[0], dtype=torch.bool, device=query.device)
        state, token_mask, exact_present = self._tokens(
            query, candidates, history, identity, event_strength, history_mask
        )
        for layer in self.layers:
            state = layer(state, token_mask, self.mode)
        state = self.final_norm(state, self.mode)
        candidate_state = state[:, 1, :].reshape(
            query.shape[0], candidates.shape[1], sum(self.block_dims)
        )
        _, recurrence_state, transfer_state = candidate_state.split(self.block_dims, dim=-1)
        recurrence_raw = self.recurrence_readout(recurrence_state).squeeze(-1)
        recurrence_scale = self.recurrence_scale_min + F.softplus(self.recurrence_log_scale)
        recurrence_delta = exact_present.to(recurrence_raw.dtype) * recurrence_scale * torch.tanh(
            recurrence_raw
        )
        transfer_raw = torch.tanh(self.transfer_readout(transfer_state).squeeze(-1))
        mask = candidate_mask.bool()
        count = mask.sum(dim=-1, keepdim=True).clamp_min(1).to(transfer_raw.dtype)
        transfer_raw = transfer_raw.masked_fill(~mask, 0.0)
        transfer_delta = self.transfer_delta_max * (
            transfer_raw - transfer_raw.sum(dim=-1, keepdim=True) / count
        )
        transfer_delta = transfer_delta.masked_fill(~mask, 0.0)
        write = (history_mask.any(dim=-1) & query_present.bool())[:, None].to(base_scores.dtype)
        recurrence_delta = recurrence_delta * write
        transfer_delta = transfer_delta * write
        scores = base_scores + recurrence_delta + transfer_delta
        if self.mode == "final_projection":
            floor = base_scores + self.recurrence_scale_min
            scores = torch.where(exact_present & write.bool(), torch.maximum(scores, floor), scores)
        scores = torch.where(write.bool(), scores, base_scores)
        return EFTOutput(
            scores=scores,
            base_scores=base_scores,
            recurrence_delta=recurrence_delta,
            transfer_delta=transfer_delta,
            candidate_state=candidate_state,
        )
