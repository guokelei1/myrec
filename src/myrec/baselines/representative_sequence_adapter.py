"""Dataset-independent sequence inputs for HSTU and LLM-SRec baselines.

This module deliberately does not build a model or read qrels. It turns the
shared standardized record into the label-free sequence/candidate information
used by both representative architecture adapters. The current request query
is appended as the final causal token, so an empty-history counterfactual still
has one valid token for sequence encoders such as the official HSTU code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from myrec.utils.jsonl import iter_jsonl, write_json
from myrec.utils.hashing import sha256_file


PAD_ID = 0
OOV_ID = 1
QUERY_TOKEN_ID = 2
FIRST_ITEM_ID = 3

PAD_EVENT_ID = 0
OOV_EVENT_ID = 1
QUERY_EVENT_ID = 2
FIRST_EVENT_ID = 3


def _text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _category_text(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return " > ".join(part for part in (_text(v) for v in value) if part)
    return _text(value)


def serialize_item_content(row: Mapping[str, Any]) -> str:
    """Serialize only visible item/context fields, never labels or positions."""

    parts: list[str] = []
    prior_query = _text(row.get("query"))
    title = _text(row.get("title"))
    brand = _text(row.get("brand"))
    category = _category_text(row.get("cat"))
    event = _text(row.get("event"))
    if prior_query:
        parts.append(f"query: {prior_query}")
    if title:
        parts.append(f"title: {title}")
    if brand:
        parts.append(f"brand: {brand}")
    if category:
        parts.append(f"category: {category}")
    if event:
        parts.append(f"event: {event}")
    return " | ".join(parts) or "[NO_ITEM_TEXT]"


@dataclass(frozen=True)
class TrainVocabulary:
    """Train-only categorical vocabulary with fixed special IDs."""

    item_to_id: dict[str, int]
    event_to_id: dict[str, int]

    @classmethod
    def fit(cls, records: Iterable[Mapping[str, Any]]) -> "TrainVocabulary":
        item_values: set[str] = set()
        event_values: set[str] = set()
        seen_request_ids: set[str] = set()
        count = 0
        for record in records:
            count += 1
            request_id = _required_text(record, "request_id")
            if request_id in seen_request_ids:
                raise ValueError(f"duplicate train request_id={request_id}")
            seen_request_ids.add(request_id)
            for row in _rows(record, "history") + _rows(record, "candidates"):
                item_values.add(_required_text(row, "item_id"))
            for row in _rows(record, "history"):
                event = _text(row.get("event"))
                if event:
                    event_values.add(event)
        if count == 0:
            raise ValueError("cannot fit sequence vocabulary from zero records")
        return cls(
            item_to_id={
                value: index
                for index, value in enumerate(sorted(item_values), start=FIRST_ITEM_ID)
            },
            event_to_id={
                value: index
                for index, value in enumerate(
                    sorted(event_values), start=FIRST_EVENT_ID
                )
            },
        )

    @classmethod
    def fit_file(cls, records_train_path: str | Path) -> "TrainVocabulary":
        return cls.fit(iter_jsonl(records_train_path))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrainVocabulary":
        special_items = payload.get("special_item_ids", {})
        special_events = payload.get("special_event_ids", {})
        if special_items != {"pad": PAD_ID, "oov": OOV_ID, "query": QUERY_TOKEN_ID}:
            raise ValueError("sequence vocabulary has incompatible special item IDs")
        if special_events != {
            "pad": PAD_EVENT_ID,
            "oov": OOV_EVENT_ID,
            "query": QUERY_EVENT_ID,
        }:
            raise ValueError("sequence vocabulary has incompatible special event IDs")
        return cls(
            item_to_id={str(k): int(v) for k, v in payload["item_to_id"].items()},
            event_to_id={str(k): int(v) for k, v in payload["event_to_id"].items()},
        )

    def item_id(self, raw_item_id: Any) -> int:
        return self.item_to_id.get(_text(raw_item_id), OOV_ID)

    def event_id(self, raw_event: Any) -> int:
        return self.event_to_id.get(_text(raw_event), OOV_EVENT_ID)

    @property
    def num_item_embeddings(self) -> int:
        return max(self.item_to_id.values(), default=QUERY_TOKEN_ID) + 1

    @property
    def num_event_embeddings(self) -> int:
        return max(self.event_to_id.values(), default=QUERY_EVENT_ID) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "special_item_ids": {
                "pad": PAD_ID,
                "oov": OOV_ID,
                "query": QUERY_TOKEN_ID,
            },
            "special_event_ids": {
                "pad": PAD_EVENT_ID,
                "oov": OOV_EVENT_ID,
                "query": QUERY_EVENT_ID,
            },
            "item_to_id": self.item_to_id,
            "event_to_id": self.event_to_id,
        }

    def write(self, output_path: str | Path) -> None:
        write_json(output_path, self.to_dict())


@dataclass(frozen=True)
class SequenceCandidate:
    raw_item_id: str
    item_id: int
    content_text: str


@dataclass(frozen=True)
class SequenceRequest:
    request_id: str
    query: str
    target_timestamp: int
    past_raw_item_ids: tuple[str, ...]
    past_item_ids: tuple[int, ...]
    past_event_ids: tuple[int, ...]
    past_timestamps: tuple[int, ...]
    past_content_texts: tuple[str, ...]
    candidates: tuple[SequenceCandidate, ...]
    retained_history_count: int
    original_history_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_sequence_request(
    record: Mapping[str, Any],
    vocabulary: TrainVocabulary,
    *,
    history_budget: int,
) -> SequenceRequest:
    """Build one causal sequence without consulting candidate labels.

    The retained history is the most recent ``history_budget`` events in the
    record's declared chronological order. The current query is always the
    final sequence token at the target request timestamp. Consequently a null
    history record yields a valid length-one sequence rather than a fabricated
    historical event.
    """

    if history_budget < 0:
        raise ValueError("history_budget must be non-negative")
    request_id = _required_text(record, "request_id")
    query = _required_text(record, "query")
    target_timestamp = _required_int(record, "ts")
    history = _rows(record, "history")
    candidates = _rows(record, "candidates")
    if not candidates:
        raise ValueError(f"request_id={request_id}: candidate slate is empty")

    previous_timestamp: int | None = None
    for index, event in enumerate(history):
        timestamp = _required_int(event, "ts")
        if timestamp > target_timestamp:
            raise ValueError(
                f"request_id={request_id}: future history at index={index}: "
                f"{timestamp}>{target_timestamp}"
            )
        if previous_timestamp is not None and timestamp < previous_timestamp:
            raise ValueError(
                f"request_id={request_id}: history is not chronological at index={index}"
            )
        previous_timestamp = timestamp

    retained = history[-history_budget:] if history_budget else []
    past_item_ids = [vocabulary.item_id(row["item_id"]) for row in retained]
    past_raw_item_ids = [_required_text(row, "item_id") for row in retained]
    past_event_ids = [vocabulary.event_id(row.get("event")) for row in retained]
    past_timestamps = [_required_int(row, "ts") for row in retained]
    past_content_texts = [serialize_item_content(row) for row in retained]

    # The query token is the causal target context. HSTU therefore never
    # receives a zero-length sequence, including in the null-history condition.
    past_item_ids.append(QUERY_TOKEN_ID)
    past_event_ids.append(QUERY_EVENT_ID)
    past_timestamps.append(target_timestamp)
    past_content_texts.append(f"query: {query}")

    candidate_rows: list[SequenceCandidate] = []
    seen_candidate_ids: set[str] = set()
    for row in candidates:
        raw_item_id = _required_text(row, "item_id")
        if raw_item_id in seen_candidate_ids:
            raise ValueError(
                f"request_id={request_id}: duplicate candidate item_id={raw_item_id}"
            )
        seen_candidate_ids.add(raw_item_id)
        candidate_rows.append(
            SequenceCandidate(
                raw_item_id=raw_item_id,
                item_id=vocabulary.item_id(raw_item_id),
                content_text=serialize_item_content(row),
            )
        )

    return SequenceRequest(
        request_id=request_id,
        query=query,
        target_timestamp=target_timestamp,
        past_raw_item_ids=tuple(past_raw_item_ids),
        past_item_ids=tuple(past_item_ids),
        past_event_ids=tuple(past_event_ids),
        past_timestamps=tuple(past_timestamps),
        past_content_texts=tuple(past_content_texts),
        candidates=tuple(candidate_rows),
        retained_history_count=len(retained),
        original_history_count=len(history),
    )


def audit_standardized_sequence_inputs(
    standardized_dir: str | Path,
    output_path: str | Path,
    *,
    history_budget: int,
    splits: Sequence[str] = ("train", "dev"),
) -> dict[str, Any]:
    """Audit the shared adapter on real records without opening any qrels."""

    root = Path(standardized_dir)
    manifest_path = root / "manifest.json"
    train_path = root / "records_train.jsonl"
    if not manifest_path.exists() or not train_path.exists():
        raise FileNotFoundError("standardized manifest and records_train.jsonl are required")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    vocabulary = TrainVocabulary.fit_file(train_path)
    split_results: dict[str, Any] = {}
    for split in splits:
        records_path = root / f"records_{split}.jsonl"
        if not records_path.exists():
            continue
        stats = {
            "candidate_count": 0,
            "candidate_oov_count": 0,
            "history_oov_count": 0,
            "null_history_request_count": 0,
            "request_count": 0,
            "retained_history_event_count": 0,
            "records_sha256": sha256_file(records_path),
        }
        for record in iter_jsonl(records_path):
            request = build_sequence_request(
                record, vocabulary, history_budget=history_budget
            )
            stats["request_count"] += 1
            stats["null_history_request_count"] += int(
                request.retained_history_count == 0
            )
            stats["retained_history_event_count"] += request.retained_history_count
            stats["history_oov_count"] += sum(
                item_id == OOV_ID for item_id in request.past_item_ids[:-1]
            )
            stats["candidate_count"] += len(request.candidates)
            stats["candidate_oov_count"] += sum(
                candidate.item_id == OOV_ID for candidate in request.candidates
            )
        split_results[split] = stats
    result = {
        "schema_version": 1,
        "dataset_id": manifest.get("dataset_id"),
        "dataset_version": manifest.get("dataset_version"),
        "standardized_dir": str(root),
        "manifest_sha256": sha256_file(manifest_path),
        "history_budget": history_budget,
        "qrels_read": False,
        "adapter_contract": "history_events_then_current_query_token_v1",
        "vocabulary": {
            "fit_split": "train",
            "item_count": len(vocabulary.item_to_id),
            "event_to_id": vocabulary.event_to_id,
            "special_item_ids": {
                "pad": PAD_ID,
                "oov": OOV_ID,
                "query": QUERY_TOKEN_ID,
            },
        },
        "splits": split_results,
        "decision": "pass" if split_results else "fail_no_requested_split",
    }
    write_json(output_path, result)
    return result


def _rows(record: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = record.get(key, [])
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{key} must be a list")
    rows: list[Mapping[str, Any]] = []
    for index, row in enumerate(value):
        if not isinstance(row, Mapping):
            raise ValueError(f"{key}[{index}] must be an object")
        rows.append(row)
    return rows


def _required_text(row: Mapping[str, Any], key: str) -> str:
    value = _text(row.get(key))
    if not value:
        raise ValueError(f"missing non-empty {key}")
    return value


def _required_int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer timestamp")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer timestamp") from exc
