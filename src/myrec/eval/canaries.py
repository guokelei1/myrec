"""C1 instrumentation score generators.

These are protocol canaries, not baselines. The random scorer is label-free;
the shuffle and title-leak canaries deliberately read qrels to test evaluator
failure modes before any real method result is interpreted.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


ScoreFn = Callable[[dict[str, Any], dict[str, Any]], float]


def generate_random_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    seed: int = 20260708,
) -> dict[str, Any]:
    """Write deterministic random scores without reading qrels."""

    method_id = "random"

    def score_fn(record: dict[str, Any], candidate: dict[str, Any]) -> float:
        return _hash_float(seed, method_id, record["request_id"], candidate["item_id"])

    return _write_record_scores(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id=method_id,
        runs_dir=runs_dir,
        seed=seed,
        qrels_read=False,
        score_fn=score_fn,
        description="deterministic hash random scores over the fixed candidate set",
    )


def generate_shuffled_label_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    seed: int = 20260708,
) -> dict[str, Any]:
    """Write a label-shuffle canary expected to collapse to random-like metrics."""

    method_id = "c1_shuffled_label_canary"
    standardized_dir = Path(standardized_dir)
    shuffled_clicked = _shuffled_clicked_labels(standardized_dir / f"qrels_{split}.jsonl", seed)

    def score_fn(record: dict[str, Any], candidate: dict[str, Any]) -> float:
        request_id = str(record["request_id"])
        item_id = str(candidate["item_id"])
        if item_id in shuffled_clicked[request_id]:
            return 1.0
        return _hash_float(seed, method_id, request_id, item_id) * 1.0e-6

    return _write_record_scores(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id=method_id,
        runs_dir=runs_dir,
        seed=seed,
        qrels_read=True,
        score_fn=score_fn,
        description="clicked labels are permuted across requests before scoring",
    )


def generate_positive_title_leak_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    runs_dir: str | Path = "runs",
    seed: int = 20260708,
) -> dict[str, Any]:
    """Write a positive-title leak canary expected to make metrics surge."""

    method_id = "c1_positive_title_leak_canary"
    standardized_dir = Path(standardized_dir)
    clicked = _load_clicked_labels(standardized_dir / f"qrels_{split}.jsonl")

    def score_fn(record: dict[str, Any], candidate: dict[str, Any]) -> float:
        request_id = str(record["request_id"])
        item_id = str(candidate["item_id"])
        positive_titles = [
            str(other.get("title") or "")
            for other in record["candidates"]
            if str(other["item_id"]) in clicked[request_id]
        ]
        leaky_query = str(record.get("query") or "") + "\n" + "\n".join(positive_titles)
        return _positive_title_score(
            leaky_query=leaky_query,
            candidate=candidate,
            request_id=request_id,
            seed=seed,
            method_id=method_id,
        )

    return _write_record_scores(
        standardized_dir=standardized_dir,
        split=split,
        run_id=run_id,
        method_id=method_id,
        runs_dir=runs_dir,
        seed=seed,
        qrels_read=True,
        score_fn=score_fn,
        description="clicked positive item titles are appended to the query before lexical scoring",
    )


def _write_record_scores(
    standardized_dir: str | Path,
    split: str,
    run_id: str,
    method_id: str,
    runs_dir: str | Path,
    seed: int,
    qrels_read: bool,
    score_fn: ScoreFn,
    description: str,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    run_dir = Path(runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = standardized_dir / f"records_{split}.jsonl"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    scores_path = run_dir / "scores.jsonl"

    rows = 0
    requests = 0
    with scores_path.open("w", encoding="utf-8") as handle:
        for record in iter_jsonl(records_path):
            requests += 1
            request_id = str(record["request_id"])
            for candidate in record["candidates"]:
                item_id = str(candidate["item_id"])
                score = float(score_fn(record, candidate))
                if not math.isfinite(score):
                    raise ValueError(f"non-finite score for {request_id} {item_id}: {score}")
                handle.write(
                    json.dumps(
                        {
                            "candidate_item_id": item_id,
                            "method_id": method_id,
                            "request_id": request_id,
                            "score": score,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                rows += 1

    metadata = {
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "dataset_id": "kuaisearch",
        "dataset_version": "v0_lite",
        "description": description,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method_id": method_id,
        "qrels_read": qrels_read,
        "records_path": str(records_path),
        "run_id": run_id,
        "score_rows": rows,
        "seed": seed,
        "split": split,
    }
    write_json(run_dir / "metadata.json", metadata)
    return metadata


def _load_clicked_labels(path: Path) -> dict[str, set[str]]:
    result = {}
    for row in iter_jsonl(path):
        result[str(row["request_id"])] = set(str(item_id) for item_id in row.get("clicked", []))
    if not result:
        raise ValueError(f"empty qrels file: {path}")
    return result


def _shuffled_clicked_labels(path: Path, seed: int) -> dict[str, set[str]]:
    clicked = _load_clicked_labels(path)
    request_ids = sorted(clicked)
    shuffled = list(request_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if len(shuffled) > 1:
        for index, request_id in enumerate(request_ids):
            if shuffled[index] == request_id:
                swap_index = (index + 1) % len(shuffled)
                shuffled[index], shuffled[swap_index] = shuffled[swap_index], shuffled[index]
    return {request_id: clicked[source_request_id] for request_id, source_request_id in zip(request_ids, shuffled)}


def _positive_title_score(
    leaky_query: str,
    candidate: dict[str, Any],
    request_id: str,
    seed: int,
    method_id: str,
) -> float:
    item_id = str(candidate["item_id"])
    title = str(candidate.get("title") or "")
    if title and title in leaky_query:
        return 1000.0 + len(title) + _hash_float(seed, method_id, request_id, item_id) * 1.0e-3

    candidate_text = " ".join(
        [
            title,
            str(candidate.get("brand") or ""),
            str(candidate.get("seller") or ""),
            " ".join(str(part) for part in candidate.get("cat", [])),
        ]
    )
    query_chars = set(leaky_query)
    candidate_chars = set(candidate_text)
    overlap = len(query_chars & candidate_chars)
    denom = max(1.0, math.sqrt(len(candidate_chars)))
    return overlap / denom + _hash_float(seed, method_id, request_id, item_id) * 1.0e-6


def _hash_float(seed: int, *parts: str) -> float:
    payload = "|".join([str(seed), *[str(part) for part in parts]])
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)
