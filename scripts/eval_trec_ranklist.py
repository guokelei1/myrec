#!/usr/bin/env python
"""Evaluate a TREC ranklist against qrels with MAP, MRR, and NDCG."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--ranklist", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--map-cutoff", type=int, default=100)
    parser.add_argument("--mrr-cutoff", type=int, default=100)
    parser.add_argument("--ndcg-cutoff", type=int, default=10)
    return parser.parse_args()


def _read_qrels(path: Path) -> dict[str, set[str]]:
    positives: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_id, _, doc_id, relevance = line.split()[:4]
            if float(relevance) > 0:
                positives[query_id].add(doc_id)
    return dict(positives)


def _read_ranklist(path: Path) -> dict[str, list[str]]:
    ranking: dict[str, list[str]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            query_id, _, doc_id = line.split()[:3]
            ranking[query_id].append(doc_id)
    return dict(ranking)


def _average_precision(ranked: list[str], positives: set[str], cutoff: int) -> float:
    hits = 0
    score = 0.0
    for rank, doc_id in enumerate(ranked[:cutoff], start=1):
        if doc_id in positives:
            hits += 1
            score += hits / rank
    return score / len(positives) if positives else 0.0


def _mrr(ranked: list[str], positives: set[str], cutoff: int) -> float:
    for rank, doc_id in enumerate(ranked[:cutoff], start=1):
        if doc_id in positives:
            return 1.0 / rank
    return 0.0


def _ndcg(ranked: list[str], positives: set[str], cutoff: int) -> float:
    dcg = 0.0
    for rank, doc_id in enumerate(ranked[:cutoff], start=1):
        if doc_id in positives:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(positives), cutoff)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def evaluate(
    qrels_path: Path,
    ranklist_path: Path,
    map_cutoff: int,
    mrr_cutoff: int,
    ndcg_cutoff: int,
) -> dict:
    positives = _read_qrels(qrels_path)
    ranking = _read_ranklist(ranklist_path)
    query_ids = sorted(positives)
    missing_ranklists = [query_id for query_id in query_ids if query_id not in ranking]

    ap_values = []
    mrr_values = []
    ndcg_values = []
    for query_id in query_ids:
        ranked = ranking.get(query_id, [])
        rel = positives[query_id]
        ap_values.append(_average_precision(ranked, rel, map_cutoff))
        mrr_values.append(_mrr(ranked, rel, mrr_cutoff))
        ndcg_values.append(_ndcg(ranked, rel, ndcg_cutoff))

    return {
        "map@100": sum(ap_values) / len(ap_values) if ap_values else 0.0,
        "mrr@100": sum(mrr_values) / len(mrr_values) if mrr_values else 0.0,
        "ndcg@10": sum(ndcg_values) / len(ndcg_values) if ndcg_values else 0.0,
        "positive_queries": len(query_ids),
        "ranked_queries": len(ranking),
        "missing_ranklist_queries": len(missing_ranklists),
        "missing_ranklist_examples": missing_ranklists[:20],
        "qrels_path": str(qrels_path),
        "ranklist_path": str(ranklist_path),
    }


def main() -> int:
    args = parse_args()
    report = evaluate(
        qrels_path=Path(args.qrels),
        ranklist_path=Path(args.ranklist),
        map_cutoff=args.map_cutoff,
        mrr_cutoff=args.mrr_cutoff,
        ndcg_cutoff=args.ndcg_cutoff,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
