"""True-input-only scoring helpers used by blind dev and determinism checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator

import numpy as np
import torch

from model.cect import CECTModel
from train.data import PackedSplit, move_batch, request_batches


@dataclass
class ScoredRequest:
    request_id: str
    candidate_item_ids: list[str]
    scores: np.ndarray
    base_scores: np.ndarray
    history_present: bool
    exact_present: bool
    evidence_present: bool
    admitted_count: int
    eligible_event_count: int


@torch.no_grad()
def score_requests(
    model: CECTModel,
    split: PackedSplit,
    indices: Iterable[int],
    device: str,
    batch_size: int,
) -> Iterator[ScoredRequest]:
    """Score observational q/c/H only; no twin is constructed or passed."""

    model.eval()
    for raw_batch in request_batches(
        indices, batch_size, shuffle=False, seed=int(split.config["seed"])
    ):
        batch = move_batch(
            split.build_batch(
                raw_batch,
                all_candidates=True,
                include_wrong_history=False,
            ),
            device,
        )
        output = model(batch, condition="true")
        if not bool(torch.isfinite(output.scores).all().item()):
            raise FloatingPointError("non-finite score during true-only scoring")
        for row, request_id in enumerate(batch["request_ids"]):
            width = int(batch["candidate_mask"][row].sum().item())
            yield ScoredRequest(
                request_id=request_id,
                candidate_item_ids=batch["candidate_item_id_rows"][row],
                scores=output.scores[row, :width].detach().float().cpu().numpy().copy(),
                base_scores=batch["base_scores"][row, :width]
                .detach()
                .float()
                .cpu()
                .numpy()
                .copy(),
                history_present=bool(batch["history_mask"][row].any().item()),
                exact_present=bool((output.exact_scores[row, :width] > 0).any().item()),
                evidence_present=bool(output.request_evidence_present[row].item()),
                admitted_count=int(
                    output.hard_admission[row, :width].sum().item()
                ),
                eligible_event_count=int(
                    output.nonexact_mask[row, :width].sum().item()
                ),
            )
