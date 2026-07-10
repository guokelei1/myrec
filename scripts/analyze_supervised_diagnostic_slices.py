#!/usr/bin/env python
"""Analyze pre-registered D1 diagnostic slices from shared evaluator outputs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.supervised_diagnostics import PackedRequestData
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


RUN_NAMES = {
    "d1q": "d1q_supervised_query_dev",
    "d1m": "d1m_mean_history_residual_dev",
    "d1a": "d1a_query_attn_residual_dev",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/analysis/supervised_motivation_diagnostics.yaml",
    )
    parser.add_argument(
        "--final-config",
        default="configs/analysis/supervised_motivation_diagnostics_final.yaml",
    )
    parser.add_argument("--seed", type=int, default=20260708)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--affinity-batch-size", type=int, default=2048)
    parser.add_argument(
        "--output",
        default="reports/pps_supervised_diagnostics_slices.json",
    )
    return parser.parse_args()


def run_id(variant: str, seed: int) -> str:
    return f"20260710_kuaisearch_{RUN_NAMES[variant]}_s{seed}"


def load_metric(path: Path) -> dict[str, float]:
    return {
        str(row["request_id"]): float(row["ndcg@10"])
        for row in iter_jsonl(path)
    }


def semantic_affinity(
    data: PackedRequestData,
    query_embeddings: torch.Tensor,
    item_embeddings: torch.Tensor,
    batch_size: int,
    device: str,
) -> np.ndarray:
    result = np.full(len(data), np.nan, dtype=np.float32)
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(len(data), batch_start + batch_size)
        indices = list(range(batch_start, batch_end))
        lengths = np.asarray(
            [
                int(data.history_offsets[index + 1] - data.history_offsets[index])
                for index in indices
            ],
            dtype=np.int64,
        )
        max_length = int(lengths.max(initial=0))
        if max_length == 0:
            continue
        history_indices = np.zeros((len(indices), max_length), dtype=np.int64)
        mask = np.zeros((len(indices), max_length), dtype=bool)
        for row, index in enumerate(indices):
            start = int(data.history_offsets[index])
            end = int(data.history_offsets[index + 1])
            length = end - start
            if length:
                history_indices[row, :length] = data.history_embedding_indices[start:end]
                mask[row, :length] = True
        query_index = torch.from_numpy(
            np.asarray(data.query_indices[batch_start:batch_end], dtype=np.int64)
        ).to(device)
        history_index = torch.from_numpy(history_indices).to(device)
        query = F.normalize(query_embeddings[query_index].float(), dim=-1, eps=1e-6)
        history = F.normalize(
            item_embeddings[history_index].float(), dim=-1, eps=1e-6
        )
        similarities = torch.einsum("bd,bhd->bh", query, history)
        similarities.masked_fill_(~torch.from_numpy(mask).to(device), -torch.inf)
        values = similarities.max(dim=1).values.cpu().numpy()
        values[lengths == 0] = np.nan
        result[batch_start:batch_end] = values
    return result


def slice_metrics(
    request_ids: list[str],
    masks: dict[str, np.ndarray],
    metrics: dict[str, dict[str, float]],
) -> dict:
    arrays = {
        variant: np.asarray([values[request_id] for request_id in request_ids])
        for variant, values in metrics.items()
    }
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        if count == 0:
            result[name] = {"request_count": 0}
            continue
        means = {variant: float(values[mask].mean()) for variant, values in arrays.items()}
        d1m_d1q = arrays["d1m"][mask] - arrays["d1q"][mask]
        d1a_d1q = arrays["d1a"][mask] - arrays["d1q"][mask]
        d1a_d1m = arrays["d1a"][mask] - arrays["d1m"][mask]
        result[name] = {
            "request_count": count,
            "mean_ndcg@10": means,
            "paired_delta": {
                "d1m_minus_d1q": float(d1m_d1q.mean()),
                "d1a_minus_d1q": float(d1a_d1q.mean()),
                "d1a_minus_d1m": float(d1a_d1m.mean()),
            },
            "positive_request_fraction": {
                "d1m_minus_d1q": float(np.mean(d1m_d1q > 0)),
                "d1a_minus_d1q": float(np.mean(d1a_d1q > 0)),
                "d1a_minus_d1m": float(np.mean(d1a_d1m > 0)),
            },
        }
    return result


def main() -> int:
    args = parse_args()
    with Path(args.base_config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    data_root = Path(config["materialized_data"]["output_dir"])
    train = PackedRequestData.load(data_root, "train")
    dev = PackedRequestData.load(data_root, "dev")
    embedding_root = Path(config["embedding_artifact"]["root"])
    query_embeddings = torch.from_numpy(
        np.array(
            np.load(embedding_root / config["embedding_artifact"]["query_embeddings"], mmap_mode="r"),
            copy=True,
        )
    ).to(args.device)
    item_embeddings = torch.from_numpy(
        np.array(
            np.load(embedding_root / config["embedding_artifact"]["item_embeddings"], mmap_mode="r"),
            copy=True,
        )
    ).to(args.device)
    train_affinity = semantic_affinity(
        train, query_embeddings, item_embeddings, args.affinity_batch_size, args.device
    )
    dev_affinity = semantic_affinity(
        dev, query_embeddings, item_embeddings, args.affinity_batch_size, args.device
    )
    train_nonempty_affinity = train_affinity[np.isfinite(train_affinity)]
    affinity_cuts = np.quantile(train_nonempty_affinity, [0.25, 0.5, 0.75])

    standardized_dir = Path(config["standardized_dir"])
    train_query_counts = Counter(
        str(row["query"]) for row in iter_jsonl(standardized_dir / "records_train.jsonl")
    )
    dev_queries = [
        str(row["query"]) for row in iter_jsonl(standardized_dir / "records_dev.jsonl")
    ]
    if len(dev_queries) != len(dev):
        raise AssertionError(f"dev query count mismatch: {len(dev_queries)} != {len(dev)}")
    query_frequency = np.asarray(
        [train_query_counts[query] for query in dev_queries], dtype=np.int64
    )

    history_length = np.diff(dev.history_offsets).astype(np.int64)
    candidate_overlap = np.zeros(len(dev), dtype=bool)
    for index in range(len(dev)):
        candidate_start = int(dev.candidate_offsets[index])
        candidate_end = int(dev.candidate_offsets[index + 1])
        history_start = int(dev.history_offsets[index])
        history_end = int(dev.history_offsets[index + 1])
        candidate_overlap[index] = bool(
            set(
                dev.candidate_embedding_indices[candidate_start:candidate_end]
            ).intersection(
                dev.history_embedding_indices[history_start:history_end]
            )
        )

    masks = {
        "history_empty": history_length == 0,
        "history_present": history_length > 0,
        "history_length_1": history_length == 1,
        "history_length_2_5": (history_length >= 2) & (history_length <= 5),
        "history_length_6_20": (history_length >= 6) & (history_length <= 20),
        "history_length_21_50": (history_length >= 21) & (history_length <= 50),
        "candidate_history_overlap_zero": (history_length > 0) & ~candidate_overlap,
        "candidate_history_overlap_positive": (history_length > 0) & candidate_overlap,
        "affinity_q1": np.isfinite(dev_affinity) & (dev_affinity <= affinity_cuts[0]),
        "affinity_q2": np.isfinite(dev_affinity) & (dev_affinity > affinity_cuts[0]) & (dev_affinity <= affinity_cuts[1]),
        "affinity_q3": np.isfinite(dev_affinity) & (dev_affinity > affinity_cuts[1]) & (dev_affinity <= affinity_cuts[2]),
        "affinity_q4": np.isfinite(dev_affinity) & (dev_affinity > affinity_cuts[2]),
        "query_frequency_unseen": query_frequency == 0,
        "query_frequency_1_4": (query_frequency >= 1) & (query_frequency <= 4),
        "query_frequency_5_plus": query_frequency >= 5,
    }

    metrics = {
        variant: load_metric(
            Path("runs") / run_id(variant, args.seed) / "per_request_metrics.jsonl"
        )
        for variant in RUN_NAMES
    }
    for variant, values in metrics.items():
        if set(values) != set(dev.request_ids):
            raise AssertionError(f"per-request coverage mismatch for {variant}")

    report = {
        "analysis_id": config["analysis_id"],
        "affinity_definition": "maximum frozen BGE cosine between query and history events",
        "affinity_train_scope": "retained positive train requests with non-empty history",
        "affinity_train_request_count": int(len(train_nonempty_affinity)),
        "affinity_train_quartile_cut_points": [float(value) for value in affinity_cuts],
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "final_config_path": args.final_config,
        "final_config_sha256": sha256_file(args.final_config),
        "headline_gate_effect": "none; descriptive pre-registered slices only",
        "qrels_read_directly": False,
        "seed": args.seed,
        "slices": slice_metrics(dev.request_ids, masks, metrics),
        "source": "shared evaluator per_request_metrics.jsonl",
        "test_read": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
