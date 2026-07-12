"""Permutation-invariant query-authenticated token interaction ranker.

The only learned ranking core is the pretrained BERT Transformer.  Frozen
input-embedding coordinates decide which candidate/history WordPieces may
participate in the full graph; ranking gradients cannot modify that decision.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

from myrec.analysis.token_history_observability import TokenHistoryData


QUERY = 0
CANDIDATE = 1
HISTORY = 2
PADDING = -1

MODES = (
    "triadic_set",
    "query_filtered_set",
    "pairwise_set",
    "triadic_positional",
    "ungated_full",
)
SET_MODES = frozenset(MODES) - {"triadic_positional"}


@dataclass(frozen=True)
class GraphBatch:
    """Packed inputs plus label-free structural coordinates."""

    input_ids: np.ndarray
    attention_mask: np.ndarray
    segment_ids: np.ndarray
    event_ids: np.ndarray
    set_position_ids: np.ndarray


def pack_candidate_graph(
    data: TokenHistoryData,
    request_index: int,
    candidate_position: int,
    *,
    scenario: str,
    token_config: dict[str, Any],
) -> GraphBatch:
    """Pack the frozen HSO text contract and retain event/set coordinates."""

    query = data._tokens(  # noqa: SLF001 - exact reuse of the frozen token contract
        data.query_token_ids[request_index],
        data.query_attention[request_index],
        int(token_config["query_tokens"]),
    )
    candidate = data._tokens(  # noqa: SLF001
        data.item_token_ids[candidate_position],
        data.item_attention[candidate_position],
        int(token_config["candidate_tokens"]),
    )
    sequence = [data.cls_token_id, *query, data.sep_token_id]
    segments = [QUERY] * len(sequence)
    events = [PADDING] * len(sequence)
    candidate_part = [*candidate, data.sep_token_id]
    sequence.extend(candidate_part)
    segments.extend([CANDIDATE] * len(candidate_part))
    events.extend([PADDING] * len(candidate_part))
    history_origin = len(sequence)
    set_positions = list(range(len(sequence)))

    for event, history_position in enumerate(
        data.history(request_index, scenario, int(token_config["max_history"]))
    ):
        history = data._tokens(  # noqa: SLF001
            data.item_token_ids[int(history_position)],
            data.item_attention[int(history_position)],
            int(token_config["history_item_tokens"]),
        )
        remaining = int(token_config["max_sequence_length"]) - len(sequence) - 1
        if remaining <= 0:
            break
        part = [*history[:remaining], data.sep_token_id]
        sequence.extend(part)
        segments.extend([HISTORY] * len(part))
        events.extend([event] * len(part))
        # Every event reuses the same within-item position coordinates.
        set_positions.extend(history_origin + offset for offset in range(len(part)))

    maximum = int(token_config["max_sequence_length"])
    sequence = sequence[:maximum]
    segments = segments[:maximum]
    events = events[:maximum]
    set_positions = set_positions[:maximum]
    ids = np.full(maximum, data.pad_token_id, dtype=np.int64)
    valid = np.zeros(maximum, dtype=bool)
    segment_ids = np.full(maximum, PADDING, dtype=np.int8)
    event_ids = np.full(maximum, PADDING, dtype=np.int8)
    position_ids = np.zeros(maximum, dtype=np.int64)
    length = len(sequence)
    ids[:length] = sequence
    valid[:length] = True
    segment_ids[:length] = segments
    event_ids[:length] = events
    position_ids[:length] = set_positions
    return GraphBatch(ids, valid, segment_ids, event_ids, position_ids)


def stack_graph_batches(rows: Sequence[GraphBatch]) -> GraphBatch:
    if not rows:
        raise ValueError("C80 cannot stack an empty graph batch")
    return GraphBatch(
        *(
            np.stack([getattr(row, field) for row in rows], axis=0)
            for field in GraphBatch.__dataclass_fields__
        )
    )


def frozen_anchor_table(word_embeddings: torch.Tensor) -> torch.Tensor:
    """Center and normalize the initial pretrained WordPiece coordinates."""

    anchors = word_embeddings.detach().float().clone()
    anchors -= anchors.mean(0, keepdim=True)
    return F.normalize(anchors, p=2, dim=-1, eps=1e-12)


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _topk_mask(
    scores: torch.Tensor,
    eligible: torch.Tensor,
    budget: int,
) -> torch.Tensor:
    """Select positive-scoring positions, with a fixed maximum budget."""

    if int(budget) <= 0:
        return torch.zeros_like(eligible)
    width = scores.shape[-1]
    count = min(int(budget), width)
    ranked = scores.masked_fill(~eligible, -torch.inf)
    values, indices = torch.topk(ranked, k=count, dim=-1, largest=True, sorted=False)
    selected = torch.zeros_like(eligible)
    selected.scatter_(1, indices, values.gt(0))
    return selected & eligible


def admitted_tokens(
    token_ids: torch.Tensor,
    valid: torch.Tensor,
    segments: torch.Tensor,
    events: torch.Tensor,
    anchors: torch.Tensor,
    *,
    mode: str,
    candidate_budget: int,
    history_budget_per_event: int,
    max_history: int,
    special_token_ids: tuple[int, int, int],
) -> torch.Tensor:
    """Return the fixed, label-free active-token mask for one C80 mode."""

    if mode not in MODES:
        raise ValueError(f"unknown C80 mode: {mode}")
    if mode == "ungated_full":
        return valid.bool()

    special = torch.zeros_like(valid, dtype=torch.bool)
    for token in special_token_ids:
        special |= token_ids.eq(int(token))
    semantic = valid.bool() & ~special
    query = semantic & segments.eq(QUERY)
    candidate = semantic & segments.eq(CANDIDATE)
    history = semantic & segments.eq(HISTORY)

    vectors = anchors[token_ids.long()]
    similarity = torch.relu(torch.bmm(vectors, vectors.transpose(1, 2)))
    query_support = similarity.masked_fill(~query[:, None, :], 0.0).amax(-1)

    if mode in {"triadic_set", "triadic_positional"}:
        triangle = torch.zeros_like(similarity)
        # Query length is at most 34.  The loop avoids a B x L x L x Q tensor.
        for position in range(token_ids.shape[1]):
            present = query[:, position]
            if not bool(present.any()):
                continue
            to_query = similarity[:, :, position] * present[:, None]
            value = similarity * to_query[:, :, None] * to_query[:, None, :]
            triangle = torch.maximum(triangle, value)
        candidate_scores = triangle.masked_fill(~history[:, None, :], 0.0).amax(-1)
        history_scores = triangle.masked_fill(~candidate[:, :, None], 0.0).amax(1)
    elif mode == "query_filtered_set":
        # Strong query-only reduction: both sides are admitted by Q support.
        candidate_scores = query_support
        history_scores = query_support
    else:
        candidate_scores = similarity.masked_fill(~history[:, None, :], 0.0).amax(-1)
        history_scores = similarity.masked_fill(~candidate[:, :, None], 0.0).amax(1)

    selected_candidate = _topk_mask(candidate_scores, candidate, candidate_budget)
    selected_history = torch.zeros_like(history)
    for event in range(int(max_history)):
        event_mask = history & events.eq(event)
        selected_history |= _topk_mask(
            history_scores,
            event_mask,
            history_budget_per_event,
        )

    # Query/readout is always available.  Boundary SEP tokens are admitted only
    # for a segment/event that retained semantic content.
    active = valid.bool() & segments.eq(QUERY)
    active |= selected_candidate
    candidate_sep = valid & segments.eq(CANDIDATE) & token_ids.eq(special_token_ids[1])
    active |= candidate_sep & selected_candidate.any(-1, keepdim=True)
    active |= selected_history
    for event in range(int(max_history)):
        selected_event = selected_history.logical_and(events.eq(event)).any(-1, keepdim=True)
        event_sep = (
            valid
            & segments.eq(HISTORY)
            & events.eq(event)
            & token_ids.eq(special_token_ids[1])
        )
        active |= event_sep & selected_event
    return active


def attention_graphs(
    active: torch.Tensor,
    segments: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct factual and history-cut dense attention graphs."""

    full = active.unsqueeze(2) & active.unsqueeze(1)
    history_query = segments.unsqueeze(2).eq(HISTORY)
    history_key = segments.unsqueeze(1).eq(HISTORY)
    cross_history = history_query ^ history_key
    cut = full & ~cross_history
    diagonal = torch.eye(active.shape[1], dtype=torch.bool, device=active.device)[None]
    # Self edges keep inactive/padded rows numerically defined; active rows
    # still cannot read inactive keys.
    return full | diagonal, cut | diagonal


def additive_mask(allowed: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    bias = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias.unsqueeze(1)


class AuthenticatedHistoryRanker(nn.Module):
    """Adaptive BGE scorer with frozen triadic graph admission."""

    def __init__(
        self,
        snapshot: str | Path,
        checkpoint_state: dict[str, torch.Tensor],
        *,
        mode: str,
        token_config: dict[str, Any],
        correction_bound: float,
        score_head_bias: bool = False,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown C80 mode: {mode}")
        self.mode = mode
        self.token_config = dict(token_config)
        self.correction_bound = float(correction_bound)
        self.backbone = AutoModel.from_pretrained(snapshot, local_files_only=True)
        hidden = int(self.backbone.config.hidden_size)
        self.score_head = nn.Linear(hidden, 1, bias=score_head_bias)
        own_state = {
            **{
                name.removeprefix("backbone."): value
                for name, value in checkpoint_state.items()
                if name.startswith("backbone.")
            }
        }
        self.backbone.load_state_dict(own_state, strict=True)
        head_state = {
            name.removeprefix("score_head."): value
            for name, value in checkpoint_state.items()
            if name.startswith("score_head.")
        }
        self.score_head.load_state_dict(head_state, strict=True)
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
        embedding = self.backbone.embeddings.word_embeddings.weight
        self.register_buffer("semantic_anchors", frozen_anchor_table(embedding), persistent=True)

    def parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(value.numel() for value in self.parameters() if value.requires_grad)

    def _position_ids(self, packed: GraphBatch, device: torch.device) -> torch.Tensor:
        if self.mode in SET_MODES:
            value = packed.set_position_ids
        else:
            value = np.broadcast_to(
                np.arange(packed.input_ids.shape[1], dtype=np.int64),
                packed.input_ids.shape,
            )
        return torch.from_numpy(np.array(value, dtype=np.int64, copy=True)).to(device)

    def raw_correction(self, packed: GraphBatch) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return uncentered candidate corrections and graph diagnostics."""

        device = next(self.parameters()).device
        ids = torch.from_numpy(np.asarray(packed.input_ids, dtype=np.int64)).to(device)
        valid = torch.from_numpy(np.asarray(packed.attention_mask, dtype=bool)).to(device)
        segments = torch.from_numpy(np.asarray(packed.segment_ids, dtype=np.int64)).to(device)
        events = torch.from_numpy(np.asarray(packed.event_ids, dtype=np.int64)).to(device)
        positions = self._position_ids(packed, device)
        with torch.no_grad():
            active = admitted_tokens(
                ids,
                valid,
                segments,
                events,
                self.semantic_anchors,
                mode=self.mode,
                candidate_budget=int(self.token_config["candidate_token_budget"]),
                history_budget_per_event=int(
                    self.token_config["history_token_budget_per_event"]
                ),
                max_history=int(self.token_config["max_history"]),
                special_token_ids=(
                    int(self.token_config["cls_token_id"]),
                    int(self.token_config["sep_token_id"]),
                    int(self.token_config["pad_token_id"]),
                ),
            )
            full, cut = attention_graphs(active, segments)
            masks = additive_mask(torch.cat((full, cut), dim=0))
        doubled_ids = torch.cat((ids, ids), dim=0)
        doubled_positions = torch.cat((positions, positions), dim=0)
        output = self.backbone(
            input_ids=doubled_ids.long(),
            attention_mask=masks,
            position_ids=doubled_positions.long(),
        )
        logits = self.score_head(output.last_hidden_state[:, 0]).squeeze(-1)
        factual, counterfactual = logits.chunk(2, dim=0)
        correction = self.correction_bound * torch.tanh(factual - counterfactual)
        diagnostics = {
            "active_fraction": active.float().mean().detach(),
            "candidate_active_fraction": (
                active & segments.eq(CANDIDATE)
            ).float().sum().div(segments.eq(CANDIDATE).float().sum().clamp_min(1)).detach(),
            "history_active_fraction": (
                active & segments.eq(HISTORY)
            ).float().sum().div(segments.eq(HISTORY).float().sum().clamp_min(1)).detach(),
        }
        return correction, diagnostics


def combine_candidate_scores(
    base_scores: torch.Tensor,
    raw_correction: torch.Tensor,
    group_sizes: Sequence[int],
) -> torch.Tensor:
    """Center the correction inside each request candidate set."""

    if int(sum(int(value) for value in group_sizes)) != int(raw_correction.numel()):
        raise ValueError("C80 candidate group sizes differ")
    output = []
    start = 0
    for size_value in group_sizes:
        size = int(size_value)
        stop = start + size
        delta = raw_correction[start:stop]
        output.append(base_scores[start:stop].detach() + delta - delta.mean())
        start = stop
    return torch.cat(output, dim=0)
