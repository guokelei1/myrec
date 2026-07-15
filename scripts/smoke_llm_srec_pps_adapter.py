#!/usr/bin/env python
"""Mechanics-only LLM-SRec smoke test with frozen local Qwen on real records."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.llm_srec_adapter import (
    FrozenQwenLLMSRecEncoder,
    LLMSRecRetrievalHead,
)
from myrec.baselines.representative_sequence_adapter import (
    TrainVocabulary,
    build_sequence_request,
)
from myrec.utils.jsonl import iter_jsonl, write_json


def _hash_feature(text: str, dim: int):
    import torch

    values = bytearray()
    counter = 0
    while len(values) < dim * 2:
        values.extend(hashlib.sha256(f"{counter}:{text}".encode()).digest())
        counter += 1
    raw = torch.frombuffer(values[: dim * 2], dtype=torch.int16).clone().float()
    return raw / 32768.0


def _cf_history_batch(requests, dim: int, device: str):
    import torch

    width = max((request.retained_history_count for request in requests), default=0)
    values = torch.zeros(len(requests), width, dim, device=device)
    mask = torch.zeros(len(requests), width, dtype=torch.bool, device=device)
    for row, request in enumerate(requests):
        for column, text in enumerate(
            request.past_content_texts[: request.retained_history_count]
        ):
            values[row, column] = _hash_feature(text, dim).to(device)
            mask[row, column] = True
    return values, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--model", default="models/huggingface/Qwen3-Reranker-0.6B")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--history-budget", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=512)
    args = parser.parse_args()

    import torch
    import transformers

    root = Path(args.standardized_dir)
    vocabulary = TrainVocabulary.fit_file(root / "records_train.jsonl")
    records = []
    for record in iter_jsonl(root / "records_dev.jsonl"):
        if record.get("history"):
            records.append(record)
        if len(records) == 2:
            break
    if len(records) < 2:
        raise ValueError("smoke test requires two history-present dev records")
    null_records = [dict(record, history=[]) for record in records]
    wrong_records = []
    for index, record in enumerate(records):
        wrong = copy.deepcopy(record)
        wrong["history"] = copy.deepcopy(records[1 - index]["history"])
        target_ts = int(wrong["ts"])
        for offset, event in enumerate(reversed(wrong["history"]), start=1):
            event["ts"] = target_ts - offset
        wrong["history"].sort(key=lambda event: int(event["ts"]))
        wrong_records.append(wrong)
    conditions = {
        "true": [
            build_sequence_request(r, vocabulary, history_budget=args.history_budget)
            for r in records
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
    cf_dim = 64
    encoder = FrozenQwenLLMSRecEncoder(
        model_name_or_path=args.model,
        cf_item_dim=cf_dim,
        max_length=args.max_length,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    encoder.train()
    head = LLMSRecRetrievalHead(
        llm_dim=encoder.llm_dim,
        cf_dim=cf_dim,
        projection_dim=128,
        hidden_dim=256,
    ).to(args.device)

    true_cf, true_mask = _cf_history_batch(conditions["true"], cf_dim, args.device)
    true_users = encoder.encode_users(conditions["true"], true_cf, true_mask)
    with torch.no_grad():
        null_cf, null_mask = _cf_history_batch(
            conditions["null"], cf_dim, args.device
        )
        wrong_cf, wrong_mask = _cf_history_batch(
            conditions["wrong"], cf_dim, args.device
        )
        null_users = encoder.encode_users(conditions["null"], null_cf, null_mask)
        wrong_users = encoder.encode_users(conditions["wrong"], wrong_cf, wrong_mask)

    candidates = [
        candidate
        for request in conditions["true"]
        for candidate in request.candidates[:2]
    ]
    cf_items = torch.stack(
        [_hash_feature(candidate.content_text, cf_dim) for candidate in candidates]
    ).to(args.device)
    item_vectors = encoder.encode_items(candidates, cf_items).reshape(
        len(records), 2, encoder.llm_dim
    )
    cf_users = torch.stack(
        [_hash_feature(request.request_id, cf_dim) for request in conditions["true"]]
    ).to(args.device)
    scores, losses = head.losses(
        llm_user=true_users,
        llm_items=item_vectors,
        cf_user=cf_users,
        positive_indices=torch.zeros(len(records), dtype=torch.long, device=args.device),
        candidate_mask=torch.ones(len(records), 2, dtype=torch.bool, device=args.device),
    )
    losses.total.backward()
    backbone_gradients = [
        parameter.grad
        for parameter in encoder.backbone.parameters()
        if parameter.grad is not None
    ]
    trainable_gradients = [
        parameter.grad
        for module in (encoder, head)
        for parameter in module.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    candidate_identity_equal = all(
        tuple(c.raw_item_id for c in conditions["true"][index].candidates)
        == tuple(c.raw_item_id for c in conditions[name][index].candidates)
        for name in ("null", "wrong")
        for index in range(len(records))
    )
    passed = (
        candidate_identity_equal
        and bool(torch.isfinite(scores).all())
        and bool(torch.isfinite(losses.total))
        and not backbone_gradients
        and trainable_gradients
        and all(bool(torch.isfinite(gradient).all()) for gradient in trainable_gradients)
        and bool((true_users - null_users).abs().max() > 0)
        and bool((true_users - wrong_users).abs().max() > 0)
    )
    result = {
        "schema_version": 1,
        "decision": "pass" if passed else "fail",
        "scientific_evidence": False,
        "qrels_read": False,
        "feature_boundary": "deterministic_hash_cf_mechanics_only",
        "standardized_dir": str(root),
        "model": args.model,
        "candidate_identity_equal_across_conditions": candidate_identity_equal,
        "backbone_parameter_count": sum(
            parameter.numel() for parameter in encoder.backbone.parameters()
        ),
        "backbone_gradient_tensor_count": len(backbone_gradients),
        "trainable_gradient_tensor_count": len(trainable_gradients),
        "finite_scores": bool(torch.isfinite(scores).all()),
        "finite_loss": bool(torch.isfinite(losses.total)),
        "true_null_user_difference": bool((true_users - null_users).abs().max() > 0),
        "true_wrong_user_difference": bool((true_users - wrong_users).abs().max() > 0),
        "score_shape": list(scores.shape),
        "package_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
