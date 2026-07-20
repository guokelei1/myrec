import hashlib

from myrec.mechanism.fold_qrels import _canonical_json
from myrec.mechanism.representation_probe import normalize_query, normalized_query_fold


def test_fold_assignment_is_frozen_sha256_mod_two():
    for query in ("abc", " A B c ", "query-two", "商品 搜索"):
        expected = int(
            hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest(), 16
        ) % 2
        assert normalized_query_fold(query) == expected


def test_qrels_serialization_is_canonical():
    assert _canonical_json({"request_id": "r", "relevance": {"b": 2, "a": 1}}) == (
        '{"relevance":{"a":1,"b":2},"request_id":"r"}'
    )
