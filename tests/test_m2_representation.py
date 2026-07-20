from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import myrec.mechanism.representation_probe as representation_probe_module
from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.representation_evaluator import (
    _audit_candidate_and_request_manifests,
)
from myrec.mechanism.representation_probe import (
    MECHANISM_PROBE_MANIFEST_SHA256,
    M2_HIDDEN_STATE_INDICES,
    MechanicalPositionError,
    audit_activation_bundle,
    build_preference_labels,
    fit_linear_readout,
    instrument_pointwise_prompt,
    load_m2_probe_manifest,
    normalize_query,
    normalized_query_fold,
    representation_holdout,
    select_train_probe_records,
    write_activation_shard,
)
from myrec.mechanism.representation_runtime import extract_m2_activations
from myrec.utils.hashing import sha256_file, sha256_text


class CharacterTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [self._id(value) for value in text]

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        return {
            "input_ids": self.encode(text),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }

    def _id(self, value: str) -> int:
        if value not in self._ids:
            self._ids[value] = len(self._ids) + 1
        return self._ids[value]


class ByteFallbackTokenizer(CharacterTokenizer):
    """Tiny tokenizer with three subtokens sharing one Unicode offset."""

    def _pieces(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for index, value in enumerate(text):
            if value == "鞋":
                for piece in range(3):
                    ids.append(self._id(f"{value}-byte-{piece}"))
                    offsets.append((index, index + 1))
            else:
                ids.append(self._id(value))
                offsets.append((index, index + 1))
        return ids, offsets

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return self._pieces(text)[0]

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        ids, offsets = self._pieces(text)
        return {"input_ids": ids, "offset_mapping": offsets}


def _raw_record(request_id: str, query: str = "轻薄电脑") -> dict[str, object]:
    return {
        "request_id": request_id,
        "query": query,
        "history": [
            {
                "item_id": f"h-{request_id}",
                "title": "旧款电脑",
                "brand": "甲牌",
                "cat": ["数码", "电脑"],
                "event": "click",
                "query": "办公电脑",
                "ts": 1,
            }
        ],
        "candidates": [
            {
                "item_id": f"a-{request_id}",
                "title": "轻薄本",
                "brand": "甲牌",
                "cat": ["数码", "电脑"],
            },
            {
                "item_id": f"b-{request_id}",
                "title": "游戏本",
                "brand": "乙牌",
                "cat": ["数码", "游戏本"],
            },
        ],
    }


def test_representation_manifest_audit_hashes_raw_query_before_prompt_strip(
    tmp_path: Path,
) -> None:
    raw = _raw_record("r-whitespace", query=" 黄金")
    record = sanitize_record_for_model(raw)
    assert record.query == "黄金"
    item_ids = [str(row["item_id"]) for row in raw["candidates"]]
    candidate_path = tmp_path / "candidate_manifest.json"
    request_path = tmp_path / "request_manifest.json"
    candidate_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "dev",
                        "request_id": raw["request_id"],
                        "candidate_item_ids": item_ids,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    request_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "split": "dev",
                        "request_id": raw["request_id"],
                        "candidate_item_ids_sha256": sha256_text(
                            json.dumps(item_ids, separators=(",", ":"))
                        ),
                        "query_sha256": sha256_text(str(raw["query"])),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert _audit_candidate_and_request_manifests(
        candidate_path, request_path, [record], [raw]
    ) == {"r-whitespace": item_ids}


@pytest.mark.parametrize(
    "method_id",
    ("q2_recranker_generalqwen", "q3_tallrec_generalqwen"),
)
def test_instrumented_prompt_has_exact_causal_positions(method_id: str) -> None:
    tokenizer = CharacterTokenizer()
    record = sanitize_record_for_model(_raw_record("r1"))
    prompt = instrument_pointwise_prompt(
        tokenizer,
        method_id,
        record,
        record.candidates[0],
        history=record.history,
        history_budget=6,
        max_length=2048,
    )
    assert prompt.query_end < prompt.history_summary_end < prompt.candidate_start
    assert prompt.candidate_start <= prompt.candidate_readout
    assert prompt.candidate_readout == len(prompt.token_ids) - 1
    assert prompt.token_ids[prompt.query_end] == tokenizer.encode(record.query)[-1]


@pytest.mark.parametrize(
    "method_id",
    ("q2_recranker_generalqwen", "q3_tallrec_generalqwen"),
)
def test_query_end_uses_last_byte_fallback_piece(method_id: str) -> None:
    tokenizer = ByteFallbackTokenizer()
    record = sanitize_record_for_model(_raw_record("r-byte", query="鞋"))
    prompt = instrument_pointwise_prompt(
        tokenizer,
        method_id,
        record,
        record.candidates[0],
        history=record.history,
        history_budget=6,
        max_length=2048,
    )
    query_ids = tokenizer.encode(record.query)
    assert list(prompt.token_ids[prompt.query_end - 2 : prompt.query_end + 1]) == query_ids
    assert prompt.query_end < prompt.history_summary_end < prompt.candidate_start
    assert prompt.candidate_start <= prompt.candidate_readout


@pytest.mark.parametrize(
    ("offsets", "char_end", "expected"),
    (
        ([(0, 5), (5, 6), (6, 8), (7, 8), (8, 9)], 8, 3),
        ([(0, 7), (7, 13), (13, 14), (14, 16), (15, 16), (15, 16), (16, 17)], 16, 5),
    ),
)
def test_span_end_uses_last_overlapping_offset_piece(
    offsets: list[tuple[int, int]], char_end: int, expected: int
) -> None:
    assert (
        representation_probe_module._token_covering_span_end(
            offsets, char_end, name="query_end"
        )
        == expected
    )


def test_span_end_without_covering_offset_fails_closed() -> None:
    with pytest.raises(MechanicalPositionError) as error:
        representation_probe_module._token_covering_span_end(
            [(0, 1), (2, 3)], 2, name="query_end"
        )
    assert error.value.code == "offset_endpoint_unresolved"


def test_span_end_with_noncontiguous_covering_offsets_fails_closed() -> None:
    with pytest.raises(MechanicalPositionError) as error:
        representation_probe_module._token_covering_span_end(
            [(0, 2), (2, 3), (1, 2)], 2, name="query_end"
        )
    assert error.value.code == "offset_endpoint_unresolved"


@pytest.mark.parametrize(
    "offsets",
    (
        [(0, 2), (1, 3)],
        [(1, 2), (0, 2)],
    ),
)
def test_span_end_with_incoherent_covering_group_fails_closed(
    offsets: list[tuple[int, int]],
) -> None:
    with pytest.raises(MechanicalPositionError) as error:
        representation_probe_module._token_covering_span_end(
            offsets, 2, name="query_end"
        )
    assert error.value.code == "offset_endpoint_unresolved"


def test_null_history_position_uses_no_history_marker_end() -> None:
    tokenizer = CharacterTokenizer()
    record = sanitize_record_for_model(_raw_record("r1"))
    prompt = instrument_pointwise_prompt(
        tokenizer,
        "q2_recranker_generalqwen",
        record,
        record.candidates[0],
        history=[],
        history_budget=6,
        max_length=2048,
    )
    assert prompt.token_ids[prompt.history_summary_end] == tokenizer.encode(
        "[NO_HISTORY]"
    )[-1]
    assert prompt.history_summary_end < prompt.candidate_start


def test_history_truncation_is_mechanical_failure_without_fallback() -> None:
    tokenizer = CharacterTokenizer()
    raw = _raw_record("r1")
    raw["history"][0]["title"] = "很长" * 200  # type: ignore[index]
    record = sanitize_record_for_model(raw)
    with pytest.raises(MechanicalPositionError) as error:
        instrument_pointwise_prompt(
            tokenizer,
            "q2_recranker_generalqwen",
            record,
            record.candidates[0],
            history=record.history,
            history_budget=6,
            max_length=512,
        )
    assert error.value.code == "history_endpoint_truncated"


def test_train_selection_is_label_free_full_slate_disjoint_and_stable() -> None:
    rows = [_raw_record(f"r{index}") for index in range(8)]
    rows[0]["history"] = []
    rows[1]["history"][0]["item_id"] = rows[1]["candidates"][0]["item_id"]  # type: ignore[index]
    selected_a, audit_a = select_train_probe_records(rows, limit=3)
    selected_b, audit_b = select_train_probe_records(reversed(rows), limit=3)
    assert [row.request_id for row in selected_a] == [row.request_id for row in selected_b]
    assert audit_a["skipped_no_history"] == 1
    assert audit_a["skipped_candidate_history_overlap"] == 1
    assert audit_a["selected_request_ids_sha256"] == audit_b[
        "selected_request_ids_sha256"
    ]


def test_normalized_query_cluster_never_crosses_split_or_fold() -> None:
    left = "Ａ B\tC"
    # Normalization is exactly casefold + whitespace removal; it intentionally
    # does not introduce an unregistered NFKC step.
    assert normalize_query(left) == "ａbc"
    assert representation_holdout(left) == representation_holdout("ＡBC")
    assert normalized_query_fold(left) == normalized_query_fold("ＡBC")


def test_linear_readout_matches_hand_separable_classes() -> None:
    matrix = np.asarray(
        [[-3.0, 0.0], [-2.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        dtype=np.float64,
    )
    labels = np.asarray(["left", "left", "right", "right"])
    readout = fit_linear_readout(matrix, labels)
    assert readout.predict(matrix).tolist() == labels.tolist()
    assert np.all(readout.scale > 0)


def test_preference_labels_use_first_max_gain_and_train_frequency() -> None:
    records = [
        sanitize_record_for_model(_raw_record("r1")),
        sanitize_record_for_model(_raw_record("r2")),
        sanitize_record_for_model(_raw_record("r3")),
    ]
    qrels = {
        "r1": {"a-r1": 2.0, "b-r1": 2.0},
        "r2": {"b-r2": 2.0},
        "r3": {"a-r3": 1.0},
    }
    labels, audit = build_preference_labels(
        records, qrels, max_classes=64, min_frequency=1
    )
    assert labels["brand"] == ["甲牌", "乙牌", "甲牌"]
    assert labels["category"] == ["电脑", "游戏本", "电脑"]
    assert audit["brand"]["frequency"] == {"乙牌": 2, "甲牌": 2}
    assert audit["brand"]["frequency_unit"] == "positive_candidate_occurrence"


def test_activation_bundle_audit_checks_full_identity_and_qrels_flag(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    records = [
        sanitize_record_for_model(_raw_record("r1")),
        sanitize_record_for_model(_raw_record("r2")),
    ]
    request = np.arange(2 * 2 * 5 * 3, dtype=np.float32).reshape(2, 2, 5, 3)
    candidate = np.arange(4 * 5 * 3, dtype=np.float32).reshape(4, 5, 3)
    shard = write_activation_shard(
        root / "shards" / "shard_00000.npz",
        request_ids=["r1", "r2"],
        normalized_queries=[normalize_query(row.query) for row in records],
        request_activations=request,
        candidate_offsets=[0, 2, 4],
        candidate_ids=["a-r1", "b-r1", "a-r2", "b-r2"],
        candidate_activations=candidate,
    )
    metadata = {
        "schema_version": 1,
        "analysis_stage": "m2_representation",
        "bundle_role": "dev_representation",
        "condition_id": "full",
        "method_id": "q2_recranker_generalqwen",
        "qrels_read": False,
        "request_positions": ["query_end", "history_summary_end"],
        "candidate_positions": ["candidate_readout"],
        "hidden_state_indices": list(M2_HIDDEN_STATE_INDICES),
        "preference_classifier_position": "history_summary_end",
        "candidate_text_visible_to_preference_classifier": False,
        "activation_passes": {
            "positions_share_same_forward": False,
            "request_level_query_history": {
                "context": "prompt_only",
                "causal_before_candidate_text": True,
            },
            "candidate_readout_donor": {
                "context": "prompt_only_frozen_scoring_kernel",
            },
        },
        "mechanism_probe_manifest": {
            "sha256": MECHANISM_PROBE_MANIFEST_SHA256,
            "expected_sha256": MECHANISM_PROBE_MANIFEST_SHA256,
            "verified": True,
        },
        "result_eligible": True,
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    index = {
        "schema_version": 1,
        "metadata_sha256": sha256_file(root / "metadata.json"),
        "request_count": 2,
        "candidate_count": 4,
        "shards": [shard],
    }
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    audited = audit_activation_bundle(
        root,
        expected_records=records,
        expected_role="dev_representation",
        expected_condition="full",
    )
    assert audited.request_ids == ("r1", "r2")
    assert audited.candidate_count == 4
    metadata["qrels_read"] = True
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    index["metadata_sha256"] = sha256_file(root / "metadata.json")
    (root / "index.json").write_text(json.dumps(index), encoding="utf-8")
    with pytest.raises(ValueError, match="qrels_read"):
        audit_activation_bundle(
            root,
            expected_records=records,
            expected_role="dev_representation",
            expected_condition="full",
        )


def test_extractor_api_has_no_qrels_argument() -> None:
    import inspect

    assert "qrels" not in inspect.signature(extract_m2_activations).parameters


def test_frozen_manifest_normalizes_unquoted_yaml_null_condition() -> None:
    identity = load_m2_probe_manifest()
    assert identity["sha256"] == MECHANISM_PROBE_MANIFEST_SHA256
    assert identity["verified"] is True


def test_train_qrels_bytes_are_untouched_when_activation_audit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    standardized = tmp_path / "standardized"
    standardized.mkdir()
    records_path = standardized / "records_train.jsonl"
    qrels_path = standardized / "qrels_train.jsonl"
    dataset_manifest_path = standardized / "manifest.json"
    records_path.write_text("{}\n", encoding="utf-8")
    qrels_path.write_text("supervised-bytes\n", encoding="utf-8")
    dataset_manifest_path.write_text("{}\n", encoding="utf-8")
    frozen = {
        "records_train_sha256": sha256_file(records_path),
        "qrels_train_sha256": sha256_file(qrels_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
    }
    monkeypatch.setattr(
        representation_probe_module,
        "load_m2_probe_manifest",
        lambda: {"frozen_inputs": frozen},
    )
    monkeypatch.setattr(
        representation_probe_module,
        "select_train_probe_records",
        lambda _rows: ([], {}),
    )
    events: list[str] = []

    def fail_activation_audit(*_args: object, **_kwargs: object) -> object:
        events.append("activation_bundle_audit")
        raise RuntimeError("activation audit failed")

    monkeypatch.setattr(
        representation_probe_module,
        "audit_activation_bundle",
        fail_activation_audit,
    )
    original_sha256_file = representation_probe_module.sha256_file

    def guarded_sha256_file(path: str | Path) -> str:
        if Path(path) == qrels_path:
            events.append("qrels_sha256")
            raise AssertionError("qrels_train was read before activation audit succeeded")
        return original_sha256_file(path)

    monkeypatch.setattr(
        representation_probe_module, "sha256_file", guarded_sha256_file
    )
    with pytest.raises(RuntimeError, match="activation audit failed"):
        representation_probe_module.fit_train_representation_probes(
            standardized,
            tmp_path / "activation_bundle",
            tmp_path / "probe",
            expected_method_id="q2_recranker_generalqwen",
        )
    assert events == ["activation_bundle_audit"]
