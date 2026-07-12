from __future__ import annotations

import torch
from transformers import AutoTokenizer

from cpdlr.model import PrefixDeltaRanker
from cpdlr.tokenization import PrefixTokenizer


def _config() -> dict[str, object]:
    return {
        "model": {
            "backbone": "BAAI/bge-small-zh-v1.5",
            "local_files_only": True,
            "d2_initialization_checkpoint": None,
            "lora_rank": 2,
            "lora_alpha": 4.0,
            "lora_dropout": 0.0,
            "lora_layers": 1,
            "delta_clip": 1.0,
            "tangent_epsilon": 1.0e-6,
        }
    }


def test_shared_parameter_identity_and_deterministic_candidate_logit() -> None:
    torch.manual_seed(7)
    model = PrefixDeltaRanker(_config(), mode="paired_delta")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-zh-v1.5", local_files_only=True, use_fast=False
    )
    prefix = PrefixTokenizer(tokenizer, 64, 8, 20, 2, 8)
    record = {"request_id": "r", "query": "蓝色衬衫", "history": []}
    candidate = {"item_id": "9", "title": "蓝衬衫", "brand": "甲", "cat": ["服装"]}
    inputs = prefix.batch_encode([(record, candidate)], "factual", 20260708)
    with torch.inference_mode():
        first = model.score(inputs)
        second = model.score(inputs)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert len(model.lora_modules) == 2
    assert model.trainable_parameter_count() < model.total_parameter_count()
