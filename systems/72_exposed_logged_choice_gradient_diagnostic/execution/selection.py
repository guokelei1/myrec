from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def structural_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "user_id": str(row["user_id"]),
        "ts": int(row["ts"]),
        "query": str(row.get("query", "")),
        "candidate_ids": [str(candidate["item_id"]) for candidate in row.get("candidates", [])],
        "history": [
            {"ts": int(event["ts"]), "item_id": str(event["item_id"])}
            for event in row.get("history", [])
        ],
    }


def iter_structural_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            yield structural_record(json.loads(line))


def load_json_map(path: str | Path) -> dict[str, int]:
    return {str(key): int(value) for key, value in json.loads(Path(path).read_text()).items()}


def candidate_key_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        value = [record["request_id"], *record["candidate_ids"]]
        digest.update(json.dumps(value, separators=(",", ":")).encode()); digest.update(b"\n")
    return digest.hexdigest()
