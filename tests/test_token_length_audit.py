from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from myrec.baselines.motivation_v12_contracts import (  # noqa: E402
    ModelRecord,
    build_prompt_sections,
    encode_instructrec_selection_prompt,
    encode_prompt_sections,
    instructrec_template_index,
    serialize_history,
)
from myrec.baselines.motivation_v12_ranker import _answer_target_tokens  # noqa: E402
from myrec.mechanism.token_length_audit import (  # noqa: E402
    ASSIGNMENT_MANIFEST_SHA256,
    PROBE_MANIFEST_SHA256,
    RequestTokenMeasurement,
    measure_request_tokens,
    summarize_measurement_comparisons,
)
from myrec.utils.hashing import sha256_file  # noqa: E402


class CharacterTokenizer:
    pad_token_id = 0
    eos_token = "<eos>"

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(value) for value in text]


def _event(item_id: str, title: str) -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": "brand",
        "cat": ["root", "leaf"],
        "event": "click",
        "query": "old query",
        "ts": 1,
    }


def _candidate(item_id: str, title: str) -> dict:
    return {
        "item_id": item_id,
        "title": title,
        "brand": "brand",
        "cat": ["root", "leaf"],
    }


def _record() -> ModelRecord:
    return ModelRecord(
        request_id="request-a",
        query="current query",
        history=(_event("h1", "first history"), _event("h2", "second history")),
        candidates=(
            _candidate("c1", "first candidate"),
            _candidate("c2", "second candidate"),
        ),
    )


def _measurement(
    total: int,
    visible: int,
    *,
    truncated: int = 0,
    overlength: int = 0,
    boundary: int = 0,
) -> RequestTokenMeasurement:
    return RequestTokenMeasurement(
        total_prompt_tokens=total,
        raw_total_prompt_tokens=total + overlength,
        visible_history_tokens=visible,
        prompt_instances=1,
        truncated_prompt_instances=truncated,
        raw_overlength_prompt_instances=overlength,
        at_max_boundary_prompt_instances=boundary,
    )


def test_hand_calculated_paired_distributions_and_rates():
    baseline = [
        _measurement(10, 1),
        _measurement(20, 2, truncated=1, overlength=1),
        _measurement(30, 3),
        _measurement(40, 4),
        _measurement(50, 5),
    ]
    intervention = [
        _measurement(15, 2),
        _measurement(18, 2, truncated=1, overlength=1),
        _measurement(30, 1),
        _measurement(50, 8, truncated=1, overlength=1, boundary=1),
        _measurement(45, 5),
    ]
    result = summarize_measurement_comparisons(baseline, intervention)

    delta = result["total_prompt_tokens"]["delta"]
    assert delta == {
        "count": 5,
        "mean": 1.6,
        "median": 0,
        "p90": 10,
        "p99": 10,
        "min": -5,
        "max": 10,
    }
    absolute = result["total_prompt_tokens"]["absolute_delta"]
    assert absolute["mean"] == 4.4
    assert absolute["median"] == 5
    assert absolute["p90"] == 10
    visible_delta = result["visible_history_tokens"]["delta"]
    assert visible_delta["mean"] == 0.6
    assert visible_delta["median"] == 0
    assert visible_delta["p90"] == 4
    rates = result["truncation_and_overlength"]
    assert rates["baseline"]["request_truncation_rate"] == 0.2
    assert rates["intervention"]["request_truncation_rate"] == 0.4
    assert rates["rate_delta"]["request_truncation_rate"] == 0.2
    assert rates["request_truncation_transitions"] == {
        "unchanged_not_truncated": 3,
        "newly_truncated": 1,
        "no_longer_truncated": 0,
        "unchanged_truncated": 1,
    }


def test_pointwise_lengths_equal_formal_encoder_for_q0_q2_q3():
    tokenizer = CharacterTokenizer()
    record = _record()
    history = list(record.history)
    for method_id in (
        "q0_qwen3_reranker_06b",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    ):
        config = {"training": {"history_budget": 6, "max_length": 512}}
        observed = measure_request_tokens(
            method_id, config, tokenizer, record, history
        )
        reserve = 0
        if method_id == "q3_tallrec_generalqwen":
            reserve = max(
                len(_answer_target_tokens(tokenizer, "Yes")),
                len(_answer_target_tokens(tokenizer, "No")),
            )
        expected = []
        for candidate in record.candidates:
            sections = build_prompt_sections(
                method_id,
                record,
                candidate,
                history=history,
                history_budget=6,
            )
            expected.append(
                len(
                    encode_prompt_sections(
                        tokenizer,
                        sections,
                        max_length=512 - reserve,
                    )
                )
            )
        assert observed.total_prompt_tokens == sum(expected)
        assert observed.prompt_instances == len(record.candidates)
        assert observed.visible_history_tokens == len(
            serialize_history(history, history_budget=6)
        )


def test_q1_length_equals_formal_slate_encoder():
    tokenizer = CharacterTokenizer()
    record = _record()
    history = list(record.history)
    config = {
        "training": {
            "history_budget": 6,
            "max_length": 1024,
            "max_target_length": 96,
            "context_token_budget": 256,
            "seed": 20260714,
        }
    }
    observed = measure_request_tokens(
        "q1_instructrec_generalqwen", config, tokenizer, record, history
    )
    template_index = instructrec_template_index(record.request_id, seed=20260714)
    expected, _responses, _audit = encode_instructrec_selection_prompt(
        tokenizer,
        record,
        record.candidates,
        history=history,
        history_budget=6,
        template_index=template_index,
        max_length=1024 - 96,
        context_token_budget=256,
        max_target_length=96,
    )
    assert observed.total_prompt_tokens == len(expected)
    assert observed.prompt_instances == 1
    assert observed.visible_history_tokens == len(
        serialize_history(history, history_budget=6)
    )


def test_frozen_probe_hash_and_assignment_binding_are_well_formed():
    root = Path(__file__).resolve().parents[1]
    assert sha256_file(root / "experiments/motivation/probe_manifest.yaml") == (
        PROBE_MANIFEST_SHA256
    )
    assert len(ASSIGNMENT_MANIFEST_SHA256) == 64
    int(ASSIGNMENT_MANIFEST_SHA256, 16)
