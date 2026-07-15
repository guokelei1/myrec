"""Build the dataset-independent request identity manifest from visible records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from myrec.utils.hashing import sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


def materialize_request_manifest(
    record_files: Iterable[tuple[str, str | Path]],
    output_path: str | Path,
    *,
    dataset_version: str,
) -> dict:
    entries = []
    seen = set()
    for split, path in record_files:
        for record in iter_jsonl(path):
            request_id = str(record["request_id"])
            if request_id in seen:
                raise ValueError(f"duplicate request_id={request_id}")
            seen.add(request_id)
            candidate_ids = [str(row["item_id"]) for row in record["candidates"]]
            entries.append({
                "split": split,
                "request_id": request_id,
                "query_sha256": sha256_text(str(record["query"])),
                "candidate_item_ids_sha256": sha256_text(
                    json.dumps(candidate_ids, separators=(",", ":"))
                ),
            })
    if not entries:
        raise ValueError("request manifest cannot be empty")
    result = {"dataset_version": dataset_version, "entries": entries}
    write_json(output_path, result)
    return result
