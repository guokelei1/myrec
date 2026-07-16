"""Shared ranking metrics for PPS experiments."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Iterable

TIE_BREAK_SALT = "20260708"


@dataclass(frozen=True)
class ScoredCandidate:
    item_id: str
    score: float


def tie_break_key(request_id: str, candidate_item_id: str) -> str:
    payload = request_id + candidate_item_id + TIE_BREAK_SALT
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sort_candidates(request_id: str, candidates: Iterable[ScoredCandidate]) -> list[ScoredCandidate]:
    candidate_list = list(candidates)
    for candidate in candidate_list:
        if not math.isfinite(candidate.score):
            raise ValueError(
                f"non-finite score for request_id={request_id} item_id={candidate.item_id}: "
                f"{candidate.score}"
            )
    return sorted(
        candidate_list,
        key=lambda candidate: (-candidate.score, tie_break_key(request_id, candidate.item_id)),
    )


def dcg_at_k(relevances: list[int], k: int) -> float:
    return sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(ranked_item_ids: list[str], positive_item_ids: set[str], k: int = 10) -> float:
    if not positive_item_ids:
        return 0.0
    relevances = [1 if item_id in positive_item_ids else 0 for item_id in ranked_item_ids]
    dcg = dcg_at_k(relevances, k)
    ideal_positives = min(len(positive_item_ids), k)
    idcg = dcg_at_k([1] * ideal_positives, k)
    return dcg / idcg if idcg else 0.0


def mrr(ranked_item_ids: list[str], positive_item_ids: set[str]) -> float:
    if not positive_item_ids:
        return 0.0
    for rank, item_id in enumerate(ranked_item_ids, start=1):
        if item_id in positive_item_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(ranked_item_ids: list[str], positive_item_ids: set[str], k: int = 10) -> float:
    if not positive_item_ids:
        return 0.0
    return len(set(ranked_item_ids[:k]) & positive_item_ids) / len(positive_item_ids)


def request_metrics(
    request_id: str,
    scored_candidates: Iterable[ScoredCandidate],
    clicked_item_ids: set[str],
    purchased_item_ids: set[str],
) -> dict[str, float | int | str | None]:
    ranked = sort_candidates(request_id, scored_candidates)
    ranked_item_ids = [candidate.item_id for candidate in ranked]
    purchase_ndcg = (
        ndcg_at_k(ranked_item_ids, purchased_item_ids, 10) if purchased_item_ids else None
    )
    return {
        "request_id": request_id,
        "ndcg@10": ndcg_at_k(ranked_item_ids, clicked_item_ids, 10),
        "mrr": mrr(ranked_item_ids, clicked_item_ids),
        "recall@10": recall_at_k(ranked_item_ids, clicked_item_ids, 10),
        "purchase_ndcg@10": purchase_ndcg,
        "clicked_positives": len(clicked_item_ids),
        "click_positive_eligible": bool(clicked_item_ids),
        "purchased_positives": len(purchased_item_ids),
    }


def aggregate_request_metrics(rows: list[dict[str, float | int | str | None]]) -> dict[str, float | int]:
    if not rows:
        raise ValueError("cannot aggregate empty request metric list")
    purchase_rows = [row for row in rows if row["purchase_ndcg@10"] is not None]
    click_positive_rows = [row for row in rows if int(row["clicked_positives"]) > 0]
    return {
        "num_requests": len(rows),
        "ndcg@10": _mean(float(row["ndcg@10"]) for row in rows),
        "mrr": _mean(float(row["mrr"]) for row in rows),
        "recall@10": _mean(float(row["recall@10"]) for row in rows),
        "purchase_ndcg@10": _mean(float(row["purchase_ndcg@10"]) for row in purchase_rows)
        if purchase_rows
        else 0.0,
        "purchase_coverage": len(purchase_rows) / len(rows),
        "purchase_num_requests": len(purchase_rows),
        "click_positive_coverage": len(click_positive_rows) / len(rows),
        "click_positive_num_requests": len(click_positive_rows),
        "click_positive_ndcg@10": _mean(
            float(row["ndcg@10"]) for row in click_positive_rows
        ),
        "click_positive_mrr": _mean(float(row["mrr"]) for row in click_positive_rows),
        "click_positive_recall@10": _mean(
            float(row["recall@10"]) for row in click_positive_rows
        ),
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
