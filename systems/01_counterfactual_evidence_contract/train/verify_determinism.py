#!/usr/bin/env python
"""Rescore the first 1,000 blind dev requests twice and demand exact identity."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from train.data import PackedSplit, iter_jsonl, sha256_file  # noqa: E402
from train.engine import load_checkpoint  # noqa: E402
from train.integrity import (  # noqa: E402
    CONFIG_PATH,
    assert_gpu_binding,
    assert_source_isolation,
    load_config,
    set_determinism,
    verify_proposal_lock,
)
from train.scoring import score_requests  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def collect(model, split, count: int) -> list[tuple[str, list[str], np.ndarray]]:
    result = []
    for request in score_requests(model, split, range(count), "cuda:0", batch_size=24):
        result.append(
            (request.request_id, request.candidate_item_ids, request.scores.copy())
        )
    return result


def score_bytes(rows: list[tuple[str, list[str], np.ndarray]]) -> bytes:
    output = bytearray()
    for request_id, item_ids, scores in rows:
        output.extend(request_id.encode("utf-8") + b"\0")
        for item_id, score in zip(item_ids, scores):
            output.extend(item_id.encode("utf-8") + b"\0")
            output.extend(struct.pack("!f", float(score)))
    return bytes(output)


def load_registered_prefix(
    scores_path: Path, expected_requests: int
) -> list[tuple[str, list[str], np.ndarray]]:
    grouped: list[tuple[str, list[str], list[float]]] = []
    current_request = None
    item_ids: list[str] = []
    scores: list[float] = []
    for row in iter_jsonl(scores_path):
        request_id = str(row["request_id"])
        if current_request is None:
            current_request = request_id
        if request_id != current_request:
            grouped.append((current_request, item_ids, scores))
            if len(grouped) >= expected_requests:
                break
            current_request, item_ids, scores = request_id, [], []
        item_ids.append(str(row["candidate_item_id"]))
        scores.append(float(row["score"]))
    else:
        if current_request is not None and len(grouped) < expected_requests:
            grouped.append((current_request, item_ids, scores))
    if len(grouped) != expected_requests:
        raise ValueError(
            f"registered score prefix request count mismatch: {len(grouped)}"
        )
    return [
        (request_id, items, np.asarray(values, dtype=np.float32))
        for request_id, items, values in grouped
    ]


def assert_identical(left, right, label: str) -> None:
    if len(left) != len(right):
        raise ValueError(f"{label}: request count differs")
    for index, (a, b) in enumerate(zip(left, right)):
        if a[0] != b[0] or a[1] != b[1]:
            raise ValueError(f"{label}: key mismatch at request {index}")
        if not np.array_equal(a[2], b[2]):
            maximum = float(np.max(np.abs(a[2] - b[2])))
            raise ValueError(f"{label}: score mismatch at request {index}: {maximum}")


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    lock = verify_proposal_lock(config)
    source = assert_source_isolation()
    gpu = assert_gpu_binding(int(config["physical_gpu"]))
    set_determinism(int(config["seed"]))
    count = int(config["screening"]["deterministic_requests"])
    run_dir = REPO_ROOT / config["paths"]["c01_run_dir"]
    scores_path = run_dir / "scores.jsonl"
    if not scores_path.exists():
        raise FileNotFoundError("blind score file must exist before determinism audit")
    split = PackedSplit(config, "dev")
    model, _ = load_checkpoint(
        REPO_ROOT / config["paths"]["c01_model"],
        config,
        lock["candidate_hash"],
        "cuda:0",
    )
    first = collect(model, split, count)
    second = collect(model, split, count)
    registered = load_registered_prefix(scores_path, count)
    assert_identical(first, second, "repeat rescore")
    assert_identical(first, registered, "checkpoint versus registered scores")
    if score_bytes(first) != score_bytes(second) or score_bytes(first) != score_bytes(registered):
        raise ValueError("serialized deterministic score bytes differ")
    score_count = sum(len(row[1]) for row in first)
    report = {
        "candidate_hash": lock["candidate_hash"],
        "checkpoint_sha256": sha256_file(REPO_ROOT / config["paths"]["c01_model"]),
        "float_encoding": "IEEE-754 binary32 network byte order",
        "gpu": gpu,
        "key_mismatches": 0,
        "max_absolute_difference": 0.0,
        "qrel_files_read": False,
        "request_count": count,
        "score_count": score_count,
        "scores_sha256": sha256_file(scores_path),
        "serialized_bytes_identical": True,
        "source_isolation": source,
        "test_files_read": False,
    }
    artifact_path = REPO_ROOT / config["paths"]["c01_artifacts"] / "determinism_first1000.json"
    run_path = run_dir / "determinism_first1000.json"
    for path in (artifact_path, run_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite determinism evidence: {path}")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps({"event": "determinism_passed", **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
