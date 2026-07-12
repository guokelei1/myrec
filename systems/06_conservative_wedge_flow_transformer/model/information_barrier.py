"""Block-sparse token attention contract for the final C06 LM core."""

from __future__ import annotations

import torch


PAD = 0
QUERY = 1
CANDIDATE = 2
HISTORY = 3


def build_information_barrier_mask(
    token_roles: torch.Tensor,
    candidate_group_ids: torch.Tensor,
) -> torch.Tensor:
    """Return `[B, L, L]` allowed-attention mask.

    Query tokens see query tokens only.  A candidate token sees query tokens and
    tokens from its own candidate segment, never another candidate or history.
    History tokens see query/history tokens, never candidates.  Thus the wedge
    layer is simultaneously the sole cross-candidate and history-to-logit path.
    Candidate group IDs are required only for candidate tokens and must be
    nonnegative there.
    """

    if token_roles.ndim != 2 or candidate_group_ids.shape != token_roles.shape:
        raise ValueError("roles and candidate_group_ids must share shape [B, L]")
    if not bool(
        ((token_roles >= PAD) & (token_roles <= HISTORY)).all().item()
    ):
        raise ValueError("unknown token role")
    candidate = token_roles == CANDIDATE
    if candidate.any() and not bool(
        (candidate_group_ids[candidate] >= 0).all().item()
    ):
        raise ValueError("candidate tokens require nonnegative group IDs")

    row_role = token_roles[:, :, None]
    column_role = token_roles[:, None, :]
    valid = (row_role != PAD) & (column_role != PAD)
    query_rows = row_role == QUERY
    candidate_rows = row_role == CANDIDATE
    history_rows = row_role == HISTORY

    query_allowed = query_rows & (column_role == QUERY)
    same_candidate = (
        candidate_group_ids[:, :, None] == candidate_group_ids[:, None, :]
    )
    candidate_allowed = candidate_rows & (
        (column_role == QUERY) | ((column_role == CANDIDATE) & same_candidate)
    )
    history_allowed = history_rows & (
        (column_role == QUERY) | (column_role == HISTORY)
    )
    return valid & (query_allowed | candidate_allowed | history_allowed)
