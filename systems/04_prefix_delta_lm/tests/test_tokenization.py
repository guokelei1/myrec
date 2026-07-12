from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from cpdlr.tokenization import PrefixTokenizer


@pytest.fixture(scope="module")
def encoder() -> PrefixTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        "BAAI/bge-small-zh-v1.5", local_files_only=True, use_fast=False
    )
    return PrefixTokenizer(tokenizer, 96, 12, 28, 3, 10)


def _candidate(item_id: str = "42") -> dict[str, object]:
    return {
        "item_id": item_id,
        "title": "蓝色棉质衬衫",
        "brand": "样例品牌",
        "cat": ["服装", "衬衫"],
    }


def test_empty_factual_and_null_prefixes_are_byte_identical(encoder: PrefixTokenizer) -> None:
    record = {"request_id": "r0", "query": "蓝色衬衫", "history": []}
    factual = encoder.encode(record, _candidate(), "factual", 20260708)
    null = encoder.encode(record, _candidate(), "null", 20260708)
    assert factual == null


def test_fixed_candidate_identity_is_encoded_deterministically(encoder: PrefixTokenizer) -> None:
    record = {"request_id": "r1", "query": "衬衫", "history": []}
    first = encoder.encode(record, _candidate("42"), "null", 20260708)
    repeat = encoder.encode(record, _candidate("42"), "null", 20260708)
    other = encoder.encode(record, _candidate("43"), "null", 20260708)
    assert first == repeat
    assert first["input_ids"] != other["input_ids"]
    assert len(first["input_ids"]) == 96


def test_query_mask_and_shuffle_change_only_the_intended_prefix(encoder: PrefixTokenizer) -> None:
    history = [
        {**_candidate("1"), "event": "click", "ts": 1},
        {**_candidate("2"), "event": "purchase", "ts": 2},
        {**_candidate("3"), "event": "click", "ts": 3},
    ]
    record = {"request_id": "r2", "query": "衬衫", "history": history}
    factual = encoder.encode(record, _candidate(), "factual", 20260708)
    masked = encoder.encode(record, _candidate(), "query_masked", 20260708)
    shuffled = encoder.encode(record, _candidate(), "shuffled", 20260708)
    assert factual["input_ids"] != masked["input_ids"]
    assert factual["input_ids"] != shuffled["input_ids"]
