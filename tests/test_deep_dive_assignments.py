from __future__ import annotations

from myrec.mechanism.deep_dive_assignments import (
    _audit_rows,
    _context_budget,
    _exact_token_span,
    _index_donors,
    _match_target,
    _tie_hash,
    materialize_fixed_candidate_sample,
)
from myrec.utils.jsonl import write_jsonl


class _WhitespaceTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


def _event(item_id: str, ts: int, title: str = "x"):
    return {
        "item_id": item_id,
        "ts": ts,
        "event": "click",
        "query": "q",
        "title": title,
        "brand": "b",
        "cat": ["c"],
    }


def test_wrong_user_mapping_preserves_count_and_excludes_items():
    donors = [
        {
            "request_id": "d0",
            "user_id": "u0",
            "history": [_event("forbidden", 1), _event("d0b", 2)],
        },
        {
            "request_id": "d1",
            "user_id": "u1",
            "history": [_event("d1a", 1), _event("d1b", 2)],
        },
    ]
    target = {
        "request_id": "r",
        "user_id": "recipient",
        "ts": 10,
        "history": [_event("old", 3), _event("old2", 4)],
        "candidate_ids": {"forbidden"},
    }
    buckets, lengths = _index_donors(donors, _WhitespaceTokenizer())
    row = _match_target(
        target,
        donor_buckets=buckets,
        donor_lengths=lengths,
        tokenizer=_WhitespaceTokenizer(),
    )
    assert row["eligible"] is True
    assert row["donor_request_id"] == "d1"
    assert len(row["history"]) == 2
    _audit_rows([row], [target])


def test_wrong_user_no_history_is_frozen_ineligible():
    target = {
        "request_id": "r",
        "user_id": "u",
        "ts": 10,
        "history": [],
        "candidate_ids": {"c"},
    }
    row = _match_target(
        target,
        donor_buckets={},
        donor_lengths={},
        tokenizer=_WhitespaceTokenizer(),
    )
    assert row["eligible"] is False
    assert row["match_type"] == "recipient_no_visible_history"


def test_tie_hash_is_identity_stable():
    assert _tie_hash("r", "d") == _tie_hash("r", "d")
    assert _tie_hash("r", "d") != _tie_hash("r", "e")


def test_exact_token_span_rejects_delimiter_crossing():
    offsets = [(0, 2), (2, 5), (5, 8)]
    assert _exact_token_span(offsets, 2, 8) == (1, 3)
    assert _exact_token_span(offsets, 3, 8) is None
    assert _exact_token_span(offsets, 2, 7) is None


def test_context_budget_matches_frozen_allocation_rule():
    assert _context_budget(20, 10, 40) == 30
    assert _context_budget(50, 10, 40) == 30
    assert _context_budget(10, 50, 40) == 10


def test_fixed_candidate_sample_is_qrels_blind_and_excludes_recurrence(tmp_path):
    records = tmp_path / "records.jsonl"
    write_jsonl(
        records,
        [
            {
                "request_id": "r0",
                "user_id": "u0",
                "ts": 10,
                "query": "q",
                "history": [_event("seen", 1)],
                "candidates": [
                    {"item_id": "seen", "title": "s", "brand": "b", "cat": []},
                    {"item_id": "new", "title": "n", "brand": "b", "cat": []},
                ],
            }
        ],
    )
    report = materialize_fixed_candidate_sample(
        records, tmp_path / "out", sample_rows=1
    )
    assert report["selected_candidate_rows"] == 1
    assert report["qrels_read"] is False
    row = __import__("json").loads(
        (tmp_path / "out" / "candidate_rows.jsonl").read_text().strip()
    )
    assert row["candidate_item_id"] == "new"
