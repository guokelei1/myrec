"""Batch 2b unified train-interaction export.

The exporter is deliberately train-only: it accepts exactly
``records_train.jsonl`` and refuses dev/test/qrels paths. Official baselines can
then consume one shared interaction artifact instead of each reconstructing
training histories with slightly different boundaries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


ALLOWED_EVENTS = {"click", "purchase"}
EVENT_PRIORITY = {"click": 0, "purchase": 1}
FORBIDDEN_TRAIN_INPUT_NAMES = {
    "records_dev.jsonl",
    "records_test.jsonl",
    "qrels_dev.jsonl",
    "qrels_test.jsonl",
}


@dataclass
class _Interaction:
    user_id: str
    item_id: str
    event_time: int
    event_type: str
    request_id: str
    sources: set[str] = field(default_factory=set)

    def merge(self, event_type: str, source: str, request_id: str) -> None:
        if EVENT_PRIORITY[event_type] > EVENT_PRIORITY[self.event_type]:
            self.event_type = event_type
        self.sources.add(source)
        if not self.request_id:
            self.request_id = request_id

    def to_json(self) -> dict[str, Any]:
        return {
            "event_time": self.event_time,
            "event_type": self.event_type,
            "item_id": self.item_id,
            "request_id": self.request_id,
            "sources": sorted(self.sources),
            "user_id": self.user_id,
        }


def export_batch2b_train_interactions(
    standardized_dir: str | Path,
    output_path: str | Path = "artifacts/batch2b/interactions_train.jsonl",
    report_path: str | Path | None = "reports/pps_batch2b_interactions_train_manifest.json",
) -> dict[str, Any]:
    """Export unified Batch 2b interactions from the standardized train split."""

    standardized_dir = Path(standardized_dir)
    return export_train_interactions_from_path(
        records_train_path=standardized_dir / "records_train.jsonl",
        output_path=output_path,
        report_path=report_path,
        standardized_dir=standardized_dir,
    )


def export_train_interactions_from_path(
    records_train_path: str | Path,
    output_path: str | Path,
    report_path: str | Path | None = None,
    standardized_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export interactions from a single train-record JSONL file.

    The output is sorted by ``(user_id, event_time, item_id)`` and de-duplicated
    by ``(user_id, item_id, event_time)``. If duplicate rows disagree on event
    type, ``purchase`` wins over ``click``.
    """

    records_train_path = Path(records_train_path)
    output_path = Path(output_path)
    report = _build_interactions(records_train_path)
    interactions = report.pop("_interactions")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        for interaction in sorted(
            interactions.values(),
            key=lambda row: (row.user_id, row.event_time, row.item_id),
        ):
            handle.write(json.dumps(interaction.to_json(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    tmp_path.replace(output_path)

    records_sha = sha256_file(records_train_path)
    output_sha = sha256_file(output_path)
    manifest = {
        "artifact_path": str(output_path),
        "artifact_sha256": output_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_assertions": {
            "accepted_input": "records_train.jsonl",
            "forbidden_inputs": sorted(FORBIDDEN_TRAIN_INPUT_NAMES),
            "train_only_path_assertion": True,
        },
        "input_path": str(records_train_path),
        "input_sha256": records_sha,
        "qrels_read": False,
        "schema": {
            "event_time": "integer timestamp copied from train history ts or train request ts",
            "event_type": "click|purchase; purchase wins duplicate conflicts",
            "item_id": "candidate/history item id string",
            "request_id": "example train request that contributed the interaction",
            "sources": "history and/or request_positive",
            "user_id": "user id string",
        },
        "sort_key": ["user_id", "event_time", "item_id"],
        "standardized_dir": str(standardized_dir) if standardized_dir else None,
        **report,
    }
    if report_path is not None:
        write_json(report_path, manifest)
    return manifest


def _build_interactions(records_train_path: Path) -> dict[str, Any]:
    _assert_records_train_path(records_train_path)
    interactions: dict[tuple[str, str, int], _Interaction] = {}
    stats = {
        "duplicate_keys": 0,
        "history_click_events": 0,
        "history_purchase_events": 0,
        "request_click_positive_events": 0,
        "request_purchase_positive_events": 0,
        "request_positive_events": 0,
        "train_candidate_rows": 0,
        "train_records": 0,
        "unique_items": 0,
        "unique_users": 0,
    }
    users: set[str] = set()
    items: set[str] = set()
    with records_train_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stats["train_records"] += 1
            user_id = _required_str(record, "user_id", line_no)
            request_id = _required_str(record, "request_id", line_no)
            request_time = _coerce_time(record.get("ts"), "ts", line_no)
            users.add(user_id)

            for event in record.get("history") or []:
                item_id = _required_str(event, "item_id", line_no)
                event_type = _event_type(event.get("event"), line_no)
                event_time = _coerce_time(event.get("ts", event.get("event_time")), "history.ts", line_no)
                source = "history"
                _add_interaction(interactions, user_id, item_id, event_time, event_type, request_id, source, stats)
                items.add(item_id)
                if event_type == "purchase":
                    stats["history_purchase_events"] += 1
                else:
                    stats["history_click_events"] += 1

            for candidate in record.get("candidates") or []:
                stats["train_candidate_rows"] += 1
                clicked = int(candidate.get("clicked", 0) or 0)
                purchased = int(candidate.get("purchased", 0) or 0)
                if not clicked and not purchased:
                    continue
                item_id = _required_str(candidate, "item_id", line_no)
                event_type = "purchase" if purchased else "click"
                _add_interaction(
                    interactions,
                    user_id,
                    item_id,
                    request_time,
                    event_type,
                    request_id,
                    "request_positive",
                    stats,
                )
                items.add(item_id)
                stats["request_positive_events"] += 1
                if purchased:
                    stats["request_purchase_positive_events"] += 1
                else:
                    stats["request_click_positive_events"] += 1

    stats["unique_interactions"] = len(interactions)
    stats["unique_items"] = len(items)
    stats["unique_users"] = len(users)
    stats["_interactions"] = interactions
    return stats


def _add_interaction(
    interactions: dict[tuple[str, str, int], _Interaction],
    user_id: str,
    item_id: str,
    event_time: int,
    event_type: str,
    request_id: str,
    source: str,
    stats: dict[str, int],
) -> None:
    key = (user_id, item_id, event_time)
    existing = interactions.get(key)
    if existing is None:
        interactions[key] = _Interaction(
            user_id=user_id,
            item_id=item_id,
            event_time=event_time,
            event_type=event_type,
            request_id=request_id,
            sources={source},
        )
    else:
        stats["duplicate_keys"] += 1
        existing.merge(event_type=event_type, source=source, request_id=request_id)


def _assert_records_train_path(path: Path) -> None:
    if path.name in FORBIDDEN_TRAIN_INPUT_NAMES or path.name != "records_train.jsonl":
        raise ValueError(
            "Batch 2b interaction export is train-only and accepts exactly records_train.jsonl; "
            f"got {path}"
        )
    if not path.exists():
        raise FileNotFoundError(path)


def _required_str(row: dict[str, Any], key: str, line_no: int) -> str:
    value = row.get(key)
    if value is None or value == "":
        raise ValueError(f"records_train.jsonl:{line_no}: missing {key}")
    return str(value)


def _coerce_time(value: Any, key: str, line_no: int) -> int:
    if value is None or value == "":
        raise ValueError(f"records_train.jsonl:{line_no}: missing {key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"records_train.jsonl:{line_no}: invalid {key}: {value!r}") from exc


def _event_type(value: Any, line_no: int) -> str:
    event_type = str(value)
    if event_type not in ALLOWED_EVENTS:
        raise ValueError(f"records_train.jsonl:{line_no}: unsupported history event {value!r}")
    return event_type
