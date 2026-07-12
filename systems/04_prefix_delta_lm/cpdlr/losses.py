"""Train-only objectives for the paired-prefix probe."""

from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def multi_positive_listwise_loss(
    logits: torch.Tensor, labels: torch.Tensor, candidate_mask: torch.Tensor
) -> torch.Tensor:
    positive = (labels > 0) & candidate_mask
    if not torch.all(positive.any(dim=-1)):
        raise ValueError("every train/internal request must contain a positive candidate")
    floor = -torch.finfo(logits.dtype).max
    denominator = torch.logsumexp(logits.masked_fill(~candidate_mask, floor), dim=-1)
    numerator = torch.logsumexp(logits.masked_fill(~positive, floor), dim=-1)
    return (denominator - numerator).mean()


def anchor_kl_loss(
    null_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    candidate_mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    floor = -torch.finfo(null_logits.dtype).max
    student = null_logits.masked_fill(~candidate_mask, floor) / temperature
    teacher = teacher_scores.masked_fill(~candidate_mask, floor) / temperature
    target = F.softmax(teacher, dim=-1)
    return F.kl_div(F.log_softmax(student, dim=-1), target, reduction="batchmean") * (
        temperature**2
    )


def _request_margin(
    delta: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_mask: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    values = []
    for row in range(delta.shape[0]):
        positives = delta[row][positive_mask[row]]
        negatives = delta[row][negative_mask[row]]
        if positives.numel() and negatives.numel():
            values.append(F.relu(torch.as_tensor(margin, device=delta.device) - positives.mean() + negatives.mean()))
    return torch.stack(values).mean() if values else delta.sum() * 0.0


def compute_probe_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
    corruption_delta: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    weights = config["loss"]
    labels = batch["labels"]
    mask = batch["candidate_mask"]
    ranking = multi_positive_listwise_loss(outputs["final"], labels, mask)
    anchor = anchor_kl_loss(
        outputs["null"],
        batch["teacher_scores"],
        mask,
        float(weights["anchor_temperature"]),
    )
    nonrepeat = (~batch["exact_repeat"]) & mask
    positive = (labels > 0) & nonrepeat
    negative = (labels <= 0) & nonrepeat
    transfer = _request_margin(
        outputs["tangent_delta"],
        positive,
        negative,
        float(weights["transfer_margin"]),
    )
    repeat = batch["exact_repeat"] & mask
    repeat_margin = _request_margin(
        outputs["tangent_delta"],
        repeat,
        (~batch["exact_repeat"]) & mask,
        float(weights["repeat_margin"]),
    )
    consistency = (
        corruption_delta.masked_select(mask).square().mean()
        if corruption_delta is not None
        else outputs["final"].new_zeros(())
    )
    total = (
        float(weights["ranking_weight"]) * ranking
        + float(weights["anchor_weight"]) * anchor
        + float(weights["transfer_weight"]) * transfer
        + float(weights["repeat_weight"]) * repeat_margin
        + float(weights["consistency_weight"]) * consistency
    )
    rows = {
        "anchor": float(anchor.detach()),
        "consistency": float(consistency.detach()),
        "ranking": float(ranking.detach()),
        "repeat": float(repeat_margin.detach()),
        "total": float(total.detach()),
        "transfer": float(transfer.detach()),
    }
    return total, rows
