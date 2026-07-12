"""Attention-edge masks for the full-token history attribution audit."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from myrec.analysis.token_history_observability import TokenHistoryData


QUERY = 0
CANDIDATE = 1
HISTORY = 2
PADDING = -1

EDGE_MODES = (
    "history_isolated",
    "no_query_history",
    "no_candidate_history",
    "no_query_reads_history",
    "no_candidate_reads_history",
    "no_history_reads_context",
)


def pack_candidate_with_segments(
    data: TokenHistoryData,
    request_index: int,
    candidate_position: int,
    *,
    scenario: str,
    token_config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack the frozen HSO sequence while retaining its semantic segments."""

    query = data._tokens(  # noqa: SLF001 - exact reuse of the frozen packer
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
    candidate_part = [*candidate, data.sep_token_id]
    sequence.extend(candidate_part)
    segments.extend([CANDIDATE] * len(candidate_part))
    for history_position in data.history(
        request_index, scenario, int(token_config["max_history"])
    ):
        history = data._tokens(  # noqa: SLF001
            data.item_token_ids[int(history_position)],
            data.item_attention[int(history_position)],
            int(token_config["history_item_tokens"]),
        )
        remaining = int(token_config["max_sequence_length"]) - len(sequence) - 1
        if remaining <= 0:
            break
        history_part = [*history[:remaining], data.sep_token_id]
        sequence.extend(history_part)
        segments.extend([HISTORY] * len(history_part))
    maximum = int(token_config["max_sequence_length"])
    sequence = sequence[:maximum]
    segments = segments[:maximum]
    ids = np.full(maximum, data.pad_token_id, dtype=np.int64)
    attention = np.zeros(maximum, dtype=bool)
    segment_ids = np.full(maximum, PADDING, dtype=np.int8)
    ids[: len(sequence)] = sequence
    attention[: len(sequence)] = True
    segment_ids[: len(segments)] = segments
    return ids, attention, segment_ids


def allowed_attention_edges(
    attention_mask: torch.Tensor, segment_ids: torch.Tensor, mode: str
) -> torch.Tensor:
    """Return a boolean [batch, query_position, key_position] edge mask."""

    if mode not in EDGE_MODES:
        raise ValueError(f"unknown edge attribution mode: {mode}")
    valid = attention_mask.bool()
    query_valid = valid.unsqueeze(2)
    key_valid = valid.unsqueeze(1)
    allowed = query_valid & key_valid
    query_segment = segment_ids.unsqueeze(2)
    key_segment = segment_ids.unsqueeze(1)

    q_is_query = query_segment.eq(QUERY)
    q_is_candidate = query_segment.eq(CANDIDATE)
    q_is_history = query_segment.eq(HISTORY)
    k_is_query = key_segment.eq(QUERY)
    k_is_candidate = key_segment.eq(CANDIDATE)
    k_is_history = key_segment.eq(HISTORY)

    if mode == "history_isolated":
        blocked = (q_is_history & ~k_is_history) | (~q_is_history & k_is_history)
    elif mode == "no_query_history":
        blocked = (q_is_query & k_is_history) | (q_is_history & k_is_query)
    elif mode == "no_candidate_history":
        blocked = (q_is_candidate & k_is_history) | (q_is_history & k_is_candidate)
    elif mode == "no_query_reads_history":
        blocked = q_is_query & k_is_history
    elif mode == "no_candidate_reads_history":
        blocked = q_is_candidate & k_is_history
    else:
        blocked = q_is_history & (k_is_query | k_is_candidate)
    return allowed & ~blocked


def additive_attention_mask(
    attention_mask: torch.Tensor,
    segment_ids: torch.Tensor,
    mode: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the prepared 4-D additive mask accepted by HuggingFace BERT."""

    allowed = allowed_attention_edges(attention_mask, segment_ids, mode)
    bias = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
    bias.masked_fill_(~allowed, torch.finfo(dtype).min)
    return bias.unsqueeze(1)
