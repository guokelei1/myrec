#!/usr/bin/env python
"""Prepare label-free TEM adapter views for V1.1 true/null/wrong scoring."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import iter_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standardized-dir", required=True)
    parser.add_argument("--assignments-dir", required=True)
    parser.add_argument("--output-parent", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    standardized_dir = Path(args.standardized_dir)
    assignments_dir = Path(args.assignments_dir)
    output_parent = Path(args.output_parent)
    output_parent.mkdir(parents=True, exist_ok=True)
    base_records = {
        str(row["request_id"]): row
        for row in iter_jsonl(standardized_dir / "records_confirmation.jsonl")
    }
    if not base_records:
        raise ValueError("empty V1.1 confirmation records")
    train_path = standardized_dir / "records_train.jsonl"
    manifest = json.loads((standardized_dir / "manifest.json").read_text(encoding="utf-8"))

    outputs = {}
    for condition in ("true", "null", "wrong"):
        assignment_path = assignments_dir / f"{condition}.jsonl"
        assignments = {
            str(row["request_id"]): row
            for row in iter_jsonl(assignment_path)
        }
        if set(assignments) != set(base_records):
            raise ValueError(f"{condition} assignment coverage mismatch")
        root = output_parent / condition
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(f"TEM adapter view is not empty: {root}")
        root.mkdir(parents=True, exist_ok=False)
        shutil.copy2(train_path, root / "records_train.jsonl")
        with (root / "records_dev.jsonl").open("w", encoding="utf-8") as handle:
            for request_id in sorted(base_records):
                record = dict(base_records[request_id])
                record["history"] = assignments[request_id]["history"]
                record["masks"] = dict(record["masks"])
                record["masks"]["history_present"] = bool(record["history"])
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        shutil.copy2(standardized_dir / "candidate_manifest.json", root / "candidate_manifest.json")
        shutil.copy2(standardized_dir / "request_manifest.json", root / "request_manifest.json")
        adapter_manifest = {
            "dataset_id": manifest["dataset_id"],
            "dataset_version": manifest["dataset_version"],
            "history_condition": condition,
            "standardized_dir": str(standardized_dir),
            "records_train_sha256": sha256_file(train_path),
            "records_dev_sha256": sha256_file(root / "records_dev.jsonl"),
            "assignment_sha256": sha256_file(assignment_path),
            "candidate_manifest_sha256": sha256_file(root / "candidate_manifest.json"),
            "request_manifest_sha256": sha256_file(root / "request_manifest.json"),
            "qrels_read": False,
            "test_opened": False,
        }
        (root / "manifest.json").write_text(
            json.dumps(adapter_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        outputs[condition] = adapter_manifest
    print(json.dumps(outputs, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
