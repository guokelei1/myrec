"""Materialize label-free structural request subsets for the frozen C04 gate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import (
    assert_candidate_manifest,
    assert_label_free_record,
    iter_jsonl,
    sha256_file,
    write_json,
)


def materialize_structural_subsets(
    config: dict[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    candidate_hash = assert_candidate_manifest(
        config["paths"]["candidate_manifest"], config["candidate_manifest_sha256"]
    )
    subsets = {
        "no_history": [],
        "nonrepeat_history_present": [],
        "repeat_present": [],
    }
    for record in iter_jsonl(config["paths"]["records_dev"]):
        assert_label_free_record(record)
        request_id = str(record["request_id"])
        history = list(record.get("history", []))
        if not history:
            subsets["no_history"].append(request_id)
            continue
        history_ids = {str(event["item_id"]) for event in history}
        candidate_ids = {str(candidate["item_id"]) for candidate in record["candidates"]}
        if history_ids.intersection(candidate_ids):
            subsets["repeat_present"].append(request_id)
        else:
            subsets["nonrepeat_history_present"].append(request_id)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, request_ids in subsets.items():
        path = target / f"{name}_request_ids.txt"
        with path.open("w", encoding="utf-8") as handle:
            for request_id in sorted(request_ids):
                handle.write(request_id + "\n")
        files[name] = {"path": str(path), "sha256": sha256_file(path)}
    report = {
        "candidate_manifest_sha256": candidate_hash,
        "counts": {name: len(values) for name, values in subsets.items()},
        "files": files,
        "qrels_read": False,
        "records_dev_sha256": sha256_file(config["paths"]["records_dev"]),
        "test_read": False,
    }
    write_json(target / "structural_subsets.json", report)
    return report
