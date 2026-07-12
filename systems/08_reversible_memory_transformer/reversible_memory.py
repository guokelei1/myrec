"""Minimal CPU prototype for C08's reversible evidence-memory primitive.

This module deliberately contains no dataset reader, evaluator, qrels access, or
device placement.  It is a structural prototype only.  The Transformer is the
end-to-end scoring backbone; the candidate-local modification is inserted as an
internal FFN-state residual between two Transformer blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn


ReadMode = Literal["loop", "ordinary"]


def _unit(vector: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize the last dimension without turning an all-zero axis into NaN."""

    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    return vector / norm.clamp_min(eps)


def reversible_coupling_step(
    state: Tensor,
    first_axis: Tensor,
    second_axis: Tensor,
    strength: Tensor,
    gains: Tensor,
    biases: Tensor,
    *,
    inverse: bool = False,
) -> Tensor:
    """Apply one exactly invertible, evidence-conditioned coupling update.

    ``state`` is split into two streams ``(x, y)``.  The forward equations are

        x' = x + s*g0*p*tanh(<q,y> + b0)
        y' = y + s*g1*q*tanh(<p,x'> + b1)

    and the inverse subtracts the second update before the first.  Each update
    is a triangular additive coupling, so the Jacobian determinant is one.  In
    particular, this is a state transition, not attention over history rows.
    """

    if state.shape[-1] % 2:
        raise ValueError("state width must be even")
    evidence_dim = state.shape[-1] // 2
    if first_axis.shape[-1] != evidence_dim:
        raise ValueError("first_axis width does not match half the state")
    if second_axis.shape[-1] != evidence_dim:
        raise ValueError("second_axis width does not match half the state")
    if gains.shape[-1] != 2 or biases.shape[-1] != 2:
        raise ValueError("gains and biases must each contain two values")

    p = _unit(first_axis)
    q = _unit(second_axis)
    x, y = state.split(evidence_dim, dim=-1)
    scale_x = strength.unsqueeze(-1) * gains[..., 0]
    scale_y = strength.unsqueeze(-1) * gains[..., 1]

    if not inverse:
        read_y = (q * y).sum(dim=-1, keepdim=True)
        x = x + scale_x * p * torch.tanh(read_y + biases[..., 0])
        read_x = (p * x).sum(dim=-1, keepdim=True)
        y = y + scale_y * q * torch.tanh(read_x + biases[..., 1])
        return torch.cat((x, y), dim=-1)

    read_x = (p * x).sum(dim=-1, keepdim=True)
    y = y - scale_y * q * torch.tanh(read_x + biases[..., 1])
    read_y = (q * y).sum(dim=-1, keepdim=True)
    x = x - scale_x * p * torch.tanh(read_y + biases[..., 0])
    return torch.cat((x, y), dim=-1)


@dataclass(frozen=True)
class MemoryDiagnostics:
    """Small in-memory diagnostics; never an evaluator or metric output."""

    raw_residual: Tensor
    centered_residual: Tensor
    history_strength: Tensor
    probe_strength: Tensor


class ReversibleCouplingMemory(nn.Module):
    """Candidate-conditioned reversible write--probe--undo memory cell.

    History events compose a map ``W``; the query/candidate pair defines a probe
    map ``P_c``.  The history read is the closed-loop displacement

        (P_c^-1 W^-1 P_c W - I) z0.

    Only this displacement is injected into the ranking Transformer.  Empty
    history makes ``W`` the identity and is overwritten with an exact zero to
    make fallback a bitwise architectural contract rather than a tolerance.
    """

    def __init__(
        self,
        d_model: int,
        evidence_dim: int,
        *,
        read_mode: ReadMode = "loop",
        max_history_strength: float = 0.45,
        max_probe_strength: float = 0.55,
    ) -> None:
        super().__init__()
        if evidence_dim < 2:
            raise ValueError("evidence_dim must be at least two")
        if read_mode not in ("loop", "ordinary"):
            raise ValueError(f"unknown read_mode: {read_mode}")
        self.d_model = d_model
        self.evidence_dim = evidence_dim
        self.state_dim = 2 * evidence_dim
        self.read_mode: ReadMode = read_mode
        self.max_history_strength = max_history_strength
        self.max_probe_strength = max_probe_strength

        # Shared axes keep exact item recurrence on a direct identity path.
        self.first_axis = nn.Linear(d_model, evidence_dim, bias=False)
        self.second_axis = nn.Linear(d_model, evidence_dim, bias=False)
        self.query_condition = nn.Linear(d_model, d_model, bias=False)
        self.history_strength = nn.Linear(d_model, 1)
        self.probe_strength = nn.Linear(d_model, 1)

        # Role parameters change the state transition, rather than selecting a
        # precomputed score expert.  The ordinary control shares every one.
        self.history_gains = nn.Parameter(torch.tensor([1.0, 0.70]))
        self.probe_gains = nn.Parameter(torch.tensor([0.80, -0.90]))
        self.history_biases = nn.Parameter(torch.tensor([0.0, 0.0]))
        self.probe_biases = nn.Parameter(torch.tensor([0.35, -0.25]))
        self.memory_seed = nn.Parameter(
            torch.linspace(-0.20, 0.20, steps=self.state_dim)
        )
        self.residual_to_hidden = nn.Linear(self.state_dim, d_model, bias=False)

        # Keep the structural smoke away from a zero-strength saddle.
        nn.init.constant_(self.history_strength.bias, 0.30)
        nn.init.constant_(self.probe_strength.bias, 0.40)

    @staticmethod
    def center_candidates(raw_residual: Tensor) -> Tensor:
        """Remove candidate-common memory displacement without parameters."""

        if raw_residual.ndim != 3:
            raise ValueError("raw_residual must have shape [batch,candidate,state]")
        return raw_residual - raw_residual.mean(dim=1, keepdim=True)

    def hidden_from_raw(self, raw_residual: Tensor) -> Tensor:
        """Map a candidate-centered memory displacement into the FFN width."""

        return self.residual_to_hidden(self.center_candidates(raw_residual))

    def _axes_and_strengths(
        self,
        history_repr: Tensor,
        query_repr: Tensor,
        candidate_repr: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        candidate_context = candidate_repr + self.query_condition(query_repr).unsqueeze(1)
        history_first = _unit(self.first_axis(history_repr))
        history_second = _unit(self.second_axis(history_repr))
        probe_first = _unit(self.first_axis(candidate_context))
        probe_second = _unit(self.second_axis(candidate_context))
        history_strength = self.max_history_strength * torch.tanh(
            self.history_strength(history_repr).squeeze(-1)
        )
        probe_strength = self.max_probe_strength * torch.tanh(
            self.probe_strength(candidate_context).squeeze(-1)
        )
        return (
            history_first,
            history_second,
            history_strength,
            probe_first,
            probe_second,
            probe_strength,
        )

    def _apply_history(
        self,
        state: Tensor,
        first_axes: Tensor,
        second_axes: Tensor,
        strengths: Tensor,
        history_mask: Tensor,
        *,
        inverse: bool,
    ) -> Tensor:
        history_len = first_axes.shape[1]
        order = range(history_len - 1, -1, -1) if inverse else range(history_len)
        candidate_count = state.shape[1]
        for index in order:
            first = first_axes[:, index].unsqueeze(1).expand(-1, candidate_count, -1)
            second = second_axes[:, index].unsqueeze(1).expand(-1, candidate_count, -1)
            strength = strengths[:, index].unsqueeze(1).expand(-1, candidate_count)
            strength = strength * history_mask[:, index].unsqueeze(1)
            state = reversible_coupling_step(
                state,
                first,
                second,
                strength,
                self.history_gains,
                self.history_biases,
                inverse=inverse,
            )
        return state

    def interaction_residual_from_axes(
        self,
        history_first: Tensor,
        history_second: Tensor,
        history_strength: Tensor,
        probe_first: Tensor,
        probe_second: Tensor,
        probe_strength: Tensor,
        history_mask: Tensor,
        *,
        seed: Tensor | None = None,
    ) -> Tensor:
        """Run ``W -> P -> W^-1 -> P^-1`` and return the displacement."""

        batch_size, candidate_count = probe_first.shape[:2]
        base = self.memory_seed if seed is None else seed
        state = base.view(1, 1, -1).expand(batch_size, candidate_count, -1)
        initial = state
        state = self._apply_history(
            state,
            history_first,
            history_second,
            history_strength,
            history_mask,
            inverse=False,
        )
        state = reversible_coupling_step(
            state,
            probe_first,
            probe_second,
            probe_strength,
            self.probe_gains,
            self.probe_biases,
            inverse=False,
        )
        state = self._apply_history(
            state,
            history_first,
            history_second,
            history_strength,
            history_mask,
            inverse=True,
        )
        state = reversible_coupling_step(
            state,
            probe_first,
            probe_second,
            probe_strength,
            self.probe_gains,
            self.probe_biases,
            inverse=True,
        )
        residual = state - initial
        present = history_mask.to(torch.bool).any(dim=1).view(batch_size, 1, 1)
        # This overwrite is the exact no-history contract.  It also prevents
        # floating-point P/P^-1 roundoff from becoming a fake personalization.
        return torch.where(present, residual, torch.zeros_like(residual))

    def ordinary_residual_from_axes(
        self,
        history_first: Tensor,
        history_second: Tensor,
        history_strength: Tensor,
        probe_first: Tensor,
        probe_second: Tensor,
        probe_strength: Tensor,
        history_mask: Tensor,
        *,
        seed: Tensor | None = None,
    ) -> Tensor:
        """Parameter-matched ordinary terminal-state memory control.

        It forms the same forward history state ``W z0`` and reads that endpoint
        with the same candidate axes.  It does not execute an inverse trajectory.
        No trainable parameter is added or removed relative to the loop mode.
        """

        batch_size, candidate_count = probe_first.shape[:2]
        base = self.memory_seed if seed is None else seed
        singleton = base.view(1, 1, -1).expand(batch_size, 1, -1)
        terminal = self._apply_history(
            singleton,
            history_first,
            history_second,
            history_strength,
            history_mask,
            inverse=False,
        )
        delta_x, delta_y = (terminal - singleton).split(self.evidence_dim, dim=-1)
        read_x = (delta_x * probe_first).sum(dim=-1, keepdim=True) * probe_first
        read_y = (delta_y * probe_second).sum(dim=-1, keepdim=True) * probe_second
        residual = torch.cat((read_x, read_y), dim=-1)
        residual = residual * probe_strength.unsqueeze(-1)
        present = history_mask.to(torch.bool).any(dim=1).view(batch_size, 1, 1)
        return torch.where(present, residual, torch.zeros_like(residual))

    def forward(
        self,
        history_repr: Tensor,
        query_repr: Tensor,
        candidate_repr: Tensor,
        history_mask: Tensor,
    ) -> tuple[Tensor, MemoryDiagnostics]:
        (
            history_first,
            history_second,
            history_strength,
            probe_first,
            probe_second,
            probe_strength,
        ) = self._axes_and_strengths(history_repr, query_repr, candidate_repr)

        if self.read_mode == "loop":
            raw = self.interaction_residual_from_axes(
                history_first,
                history_second,
                history_strength,
                probe_first,
                probe_second,
                probe_strength,
                history_mask,
            )
        else:
            raw = self.ordinary_residual_from_axes(
                history_first,
                history_second,
                history_strength,
                probe_first,
                probe_second,
                probe_strength,
                history_mask,
            )
        centered = self.center_candidates(raw)
        hidden = self.residual_to_hidden(centered)
        diagnostics = MemoryDiagnostics(
            raw_residual=raw,
            centered_residual=centered,
            history_strength=history_strength,
            probe_strength=probe_strength,
        )
        return hidden, diagnostics


class TinyReversibleRanker(nn.Module):
    """Tiny end-to-end Transformer ranker for CPU structural tests only."""

    def __init__(
        self,
        *,
        vocab_size: int = 64,
        item_vocab_size: int = 32,
        d_model: int = 16,
        evidence_dim: int = 6,
        nhead: int = 4,
        ffn_dim: int = 32,
        max_tokens: int = 12,
        read_mode: ReadMode = "loop",
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_tokens = max_tokens
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.item_embedding = nn.Embedding(item_vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_tokens, d_model)
        self.role_embedding = nn.Embedding(3, d_model)
        self.pre_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.memory = ReversibleCouplingMemory(
            d_model=d_model,
            evidence_dim=evidence_dim,
            read_mode=read_mode,
        )
        self.post_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.score_head = nn.Linear(d_model, 1, bias=False)

    def _encode_text(self, tokens: Tensor, role: int) -> Tensor:
        if tokens.shape[-1] > self.max_tokens:
            raise ValueError("token length exceeds max_tokens")
        leading = tokens.shape[:-1]
        length = tokens.shape[-1]
        flat = tokens.reshape(-1, length)
        positions = torch.arange(length, device=tokens.device).view(1, length)
        hidden = self.token_embedding(flat)
        hidden = hidden + self.position_embedding(positions)
        hidden = hidden + self.role_embedding.weight[role].view(1, 1, -1)
        hidden = self.pre_block(hidden)
        mask = flat.ne(0).unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1)
        pooled = pooled / mask.sum(dim=1).clamp_min(1)
        return pooled.reshape(*leading, self.d_model)

    def _score_representations(
        self,
        query_repr: Tensor,
        candidate_repr: Tensor,
        memory_hidden: Tensor,
    ) -> Tensor:
        batch_size, candidate_count = candidate_repr.shape[:2]
        query = query_repr.unsqueeze(1).expand(-1, candidate_count, -1)
        candidate = candidate_repr + memory_hidden
        pair = torch.stack((query, candidate), dim=2)
        pair = pair.reshape(batch_size * candidate_count, 2, self.d_model)
        pair = self.post_block(pair)
        candidate_hidden = pair[:, 1].reshape(batch_size, candidate_count, self.d_model)
        return self.score_head(candidate_hidden).squeeze(-1)

    def _base_representations(
        self,
        query_tokens: Tensor,
        candidate_tokens: Tensor,
        candidate_item_ids: Tensor,
    ) -> tuple[Tensor, Tensor]:
        query_repr = self._encode_text(query_tokens, role=0)
        candidate_repr = self._encode_text(candidate_tokens, role=2)
        candidate_repr = candidate_repr + self.item_embedding(candidate_item_ids)
        return query_repr, candidate_repr

    def forward_query_only(
        self,
        query_tokens: Tensor,
        candidate_tokens: Tensor,
        candidate_item_ids: Tensor,
    ) -> Tensor:
        """The exact fallback path used to audit empty-history equivalence."""

        query_repr, candidate_repr = self._base_representations(
            query_tokens, candidate_tokens, candidate_item_ids
        )
        return self._score_representations(
            query_repr, candidate_repr, torch.zeros_like(candidate_repr)
        )

    def forward(
        self,
        query_tokens: Tensor,
        history_tokens: Tensor,
        history_item_ids: Tensor,
        history_mask: Tensor,
        candidate_tokens: Tensor,
        candidate_item_ids: Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> Tensor | tuple[Tensor, MemoryDiagnostics]:
        query_repr, candidate_repr = self._base_representations(
            query_tokens, candidate_tokens, candidate_item_ids
        )
        history_repr = self._encode_text(history_tokens, role=1)
        history_repr = history_repr + self.item_embedding(history_item_ids)
        history_repr = history_repr * history_mask.unsqueeze(-1)
        memory_hidden, diagnostics = self.memory(
            history_repr, query_repr, candidate_repr, history_mask
        )
        scores = self._score_representations(query_repr, candidate_repr, memory_hidden)
        if return_diagnostics:
            return scores, diagnostics
        return scores
