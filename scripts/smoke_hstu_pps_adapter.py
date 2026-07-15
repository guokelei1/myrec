#!/usr/bin/env python
"""Exercise HSTU/SASRec PPS adapters on real label-free standardized records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "baselines" / "hstu"))

from myrec.baselines.hstu_pps_adapter import (
    HSTUPPSRanker,
    collate_sequence_requests,
)
from myrec.baselines.frozen_text_features import FrozenTextFeatureStore
from myrec.baselines.representative_sequence_adapter import (
    TrainVocabulary,
    build_sequence_request,
)
from myrec.utils.jsonl import iter_jsonl, write_json


def _hash_feature(text: str, dim: int):
    """Deterministic mechanics-only feature; never used for a scientific run."""

    import numpy as np

    values = bytearray()
    counter = 0
    while len(values) < dim:
        values.extend(hashlib.sha256(f"{counter}:{text}".encode()).digest())
        counter += 1
    array = np.frombuffer(bytes(values[:dim]), dtype=np.uint8).astype("float32")
    return (array - 127.5) / 127.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-budget", type=int, default=8)
    parser.add_argument("--content-dim", type=int, default=32)
    parser.add_argument("--feature-store")
    args = parser.parse_args()

    import torch

    root = Path(args.standardized_dir)
    vocabulary = TrainVocabulary.fit_file(root / "records_train.jsonl")
    visible_records = []
    for record in iter_jsonl(root / "records_dev.jsonl"):
        visible_records.append(record)
        if len(visible_records) == 2:
            break
    if len(visible_records) < 2:
        raise ValueError("smoke test requires two visible development records")
    true_records = visible_records
    null_records = [dict(record, history=[]) for record in visible_records]
    wrong_records = []
    for index, record in enumerate(visible_records):
        donor = visible_records[1 - index]
        wrong = copy.deepcopy(record)
        wrong["history"] = copy.deepcopy(donor.get("history", []))
        # Donor times can exceed this request's target. Move them backward while
        # preserving order; this is mechanics-only and never a provenance claim.
        target_ts = int(wrong["ts"])
        for offset, event in enumerate(reversed(wrong["history"]), start=1):
            event["ts"] = target_ts - offset
        wrong["history"].sort(key=lambda event: int(event["ts"]))
        wrong_records.append(wrong)
    conditions = {
        "true": [
            build_sequence_request(r, vocabulary, history_budget=args.history_budget)
            for r in true_records
        ],
        "null": [
            build_sequence_request(r, vocabulary, history_budget=args.history_budget)
            for r in null_records
        ],
        "wrong": [
            build_sequence_request(r, vocabulary, history_budget=args.history_budget)
            for r in wrong_records
        ],
    }
    candidate_identity_equal = all(
        tuple(c.raw_item_id for c in conditions["true"][i].candidates)
        == tuple(c.raw_item_id for c in conditions[condition][i].candidates)
        for condition in ("null", "wrong")
        for i in range(2)
    )
    if args.feature_store:
        feature_store = FrozenTextFeatureStore(args.feature_store)
        feature_lookup = feature_store
        content_dim = feature_store.dimension
        feature_boundary = feature_store.metadata["feature_contract"]
    else:
        feature_lookup = lambda text: _hash_feature(text, args.content_dim)
        content_dim = args.content_dim
        feature_boundary = "deterministic_hash_mechanics_only"
    max_sequence_length = args.history_budget + 1
    batches = {
        name: collate_sequence_requests(
            requests,
            feature_lookup,
            content_dim=content_dim,
            max_sequence_length=max_sequence_length,
        ).to(args.device)
        for name, requests in conditions.items()
    }
    torch.manual_seed(20260715)
    torch.cuda.manual_seed_all(20260715)
    checks = []
    for architecture in ("hstu", "sasrec"):
        model = HSTUPPSRanker(
            architecture=architecture,
            num_item_ids=vocabulary.num_item_embeddings,
            num_event_ids=vocabulary.num_event_embeddings,
            content_dim=content_dim,
            embedding_dim=32,
            max_sequence_length=max_sequence_length,
            num_blocks=1,
            num_heads=1,
            dropout_rate=0.0,
        ).to(args.device)
        model.train()
        scores = {name: model(batch) for name, batch in batches.items()}
        loss = sum(value.square().mean() for value in scores.values())
        loss.backward()
        gradients = [
            p.grad for p in model.parameters() if p.requires_grad and p.grad is not None
        ]
        checks.append(
            {
                "architecture": architecture,
                "finite_scores": all(
                    bool(torch.isfinite(value).all()) for value in scores.values()
                ),
                "finite_gradients": all(
                    bool(torch.isfinite(value).all()) for value in gradients
                ),
                "nonzero_gradient_tensor_count": sum(
                    int(bool((value != 0).any())) for value in gradients
                ),
                "score_shapes": {
                    name: list(value.shape) for name, value in scores.items()
                },
                "true_null_score_difference": bool(
                    (scores["true"] - scores["null"]).abs().max() > 0
                ),
            }
        )
    passed = candidate_identity_equal and all(
        check["finite_scores"]
        and check["finite_gradients"]
        and check["nonzero_gradient_tensor_count"] > 0
        and check["true_null_score_difference"]
        for check in checks
    )
    result = {
        "schema_version": 1,
        "decision": "pass" if passed else "fail",
        "scientific_evidence": False,
        "feature_boundary": feature_boundary,
        "feature_store": args.feature_store,
        "qrels_read": False,
        "standardized_dir": str(root),
        "candidate_identity_equal_across_conditions": candidate_identity_equal,
        "null_sequence_lengths": [
            len(request.past_item_ids) for request in conditions["null"]
        ],
        "checks": checks,
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
