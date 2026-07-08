"""Build fixed-sample top-k review sheets for baseline sanity checks."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from myrec.eval.metrics import ScoredCandidate, sort_candidates
from myrec.utils.jsonl import iter_jsonl


def write_topk_review(
    records_path: str | Path,
    scores_path: str | Path,
    output_path: str | Path,
    run_id: str,
    sample_size: int = 20,
    top_k: int = 5,
    seed: int = 20260708,
) -> dict[str, Any]:
    records = {str(row["request_id"]): row for row in iter_jsonl(records_path)}
    scores = _load_scores(scores_path)
    request_ids = sorted(set(records) & set(scores))
    rng = random.Random(seed)
    sampled = sorted(rng.sample(request_ids, min(sample_size, len(request_ids))))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Top-{top_k} Review: {run_id}\n\n")
        handle.write(
            "This sheet is generated without reading qrels. It is for manual sanity "
            "inspection of whether lexical top results match the query intent.\n\n"
        )
        handle.write(f"- Seed: {seed}\n")
        handle.write(f"- Sampled requests: {len(sampled)}\n")
        handle.write(f"- Scores: `{scores_path}`\n\n")
        for index, request_id in enumerate(sampled, start=1):
            record = records[request_id]
            candidates_by_id = {str(candidate["item_id"]): candidate for candidate in record["candidates"]}
            ranked = sort_candidates(
                request_id,
                [
                    ScoredCandidate(item_id=item_id, score=score)
                    for item_id, score in scores[request_id].items()
                ],
            )
            handle.write(f"## {index}. `{request_id}`\n\n")
            handle.write(f"Query: {record.get('query', '')}\n\n")
            for rank, candidate in enumerate(ranked[:top_k], start=1):
                item = candidates_by_id[candidate.item_id]
                cats = " / ".join(str(part) for part in item.get("cat", []))
                handle.write(
                    f"{rank}. `{candidate.item_id}` score={candidate.score:.6f} | "
                    f"{item.get('title', '')} | {item.get('brand', '')} | {cats}\n"
                )
            handle.write("\nReview note: TODO manual pass/fail.\n\n")
    return {
        "output_path": str(output_path),
        "request_count": len(sampled),
        "run_id": run_id,
        "seed": seed,
        "top_k": top_k,
    }


def _load_scores(path: str | Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        scores.setdefault(request_id, {})[str(row["candidate_item_id"])] = float(row["score"])
    return scores
