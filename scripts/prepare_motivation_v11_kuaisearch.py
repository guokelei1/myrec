#!/usr/bin/env python
"""Build the V1.1 KuaiSearch train extension without changing V1 confirmation."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.data.contracts import audit_standardized_file  # noqa: E402
from myrec.data.kuaisearch_scout import build_kuaisearch_lite_scout  # noqa: E402
from myrec.utils.hashing import sha256_file, sha256_text  # noqa: E402
from myrec.utils.jsonl import iter_jsonl, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default="data/raw/kuaisearch_full")
    parser.add_argument(
        "--v1-dir",
        default="data/standardized/kuaisearch/full_confirm_preceding10k_v1",
    )
    parser.add_argument(
        "--staging-dir",
        default="data/interim/kuaisearch/motivation_v11_train_window40k",
    )
    parser.add_argument(
        "--output-dir",
        default="data/standardized/kuaisearch/full_confirm_preceding40k_v11",
    )
    parser.add_argument(
        "--report-path",
        default="reports/pps_motivation_v11_kuaisearch_data_admission.json",
    )
    parser.add_argument("--dataset-version", default="full_confirm_preceding40k_v11")
    parser.add_argument("--max-requests", type=int, default=40_000)
    parser.add_argument("--dev-fraction", type=float, default=0.20)
    parser.add_argument("--max-history-len", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    v1_dir = Path(args.v1_dir)
    staging_dir = Path(args.staging_dir)
    output_dir = Path(args.output_dir)
    report_path = Path(args.report_path)
    for path in (v1_dir,):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise FileExistsError(f"staging directory is not empty: {staging_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")

    v1_records = v1_dir / "records_confirmation.jsonl"
    v1_qrels = v1_dir / "qrels_confirmation.jsonl"
    v1_candidates = _read_json(v1_dir / "candidate_manifest.json")
    v1_request_manifest = _read_json(v1_dir / "request_manifest.json")
    confirmation_records = list(iter_jsonl(v1_records))
    if not confirmation_records:
        raise ValueError("V1 confirmation records are empty")
    confirmation_min_ts = min(int(row["ts"]) for row in confirmation_records)
    confirmation_sessions = {str(row["session_id"]) for row in confirmation_records}

    staging_report = staging_dir.parent / f"{staging_dir.name}_manifest.json"
    staging = build_kuaisearch_lite_scout(
        raw_dir=args.raw_dir,
        output_dir=staging_dir,
        report_path=staging_report,
        dataset_version=f"{args.dataset_version}_staging",
        max_requests=args.max_requests,
        dev_fraction=args.dev_fraction,
        max_history_len=args.max_history_len,
        include_history_query=True,
        evaluation_split="dev",
        end_before_time=confirmation_min_ts,
    )

    staged_records = {
        split: staging_dir / f"records_{split}.jsonl"
        for split in ("train", "dev")
    }
    staged_qrels = {
        split: staging_dir / f"qrels_{split}.jsonl"
        for split in ("train", "dev")
    }
    selected_records = [
        *iter_jsonl(staged_records["train"]),
        *iter_jsonl(staged_records["dev"]),
    ]
    overlap_sessions = sorted(
        {str(row["session_id"]) for row in selected_records} & confirmation_sessions
    )
    if overlap_sessions:
        raise ValueError(
            "expanded KuaiSearch source window overlaps confirmation sessions: "
            f"{overlap_sessions[:5]}"
        )
    if any(int(row["ts"]) >= confirmation_min_ts for row in selected_records):
        raise ValueError("expanded training window is not strictly before confirmation")
    if {str(row["request_id"]) for row in selected_records} & {
        str(row["request_id"]) for row in confirmation_records
    }:
        raise ValueError("expanded training window overlaps confirmation requests")

    output_dir.mkdir(parents=True, exist_ok=False)
    for split in ("train", "dev"):
        shutil.copy2(staged_records[split], output_dir / f"records_{split}.jsonl")
        shutil.copy2(staged_qrels[split], output_dir / f"qrels_{split}.jsonl")
    shutil.copy2(v1_records, output_dir / "records_confirmation.jsonl")
    shutil.copy2(v1_qrels, output_dir / "qrels_confirmation.jsonl")

    staged_candidates = _read_json(staging_dir / "candidate_manifest.json")
    staged_requests = _read_json(staging_dir / "request_manifest.json")
    staged_candidate_entries = [
        entry for entry in staged_candidates["entries"] if entry["split"] in {"train", "dev"}
    ]
    staged_request_entries = [
        entry for entry in staged_requests["entries"] if entry["split"] in {"train", "dev"}
    ]
    v1_confirmation_candidates = [
        entry for entry in v1_candidates["entries"] if entry["split"] == "confirmation"
    ]
    v1_confirmation_requests = [
        entry for entry in v1_request_manifest["entries"] if entry["split"] == "confirmation"
    ]
    if len(v1_confirmation_candidates) != len(confirmation_records):
        raise ValueError("V1 candidate manifest does not cover confirmation")
    if len(v1_confirmation_requests) != len(confirmation_records):
        raise ValueError("V1 request manifest does not cover confirmation")

    combined_candidates = {
        "dataset_version": args.dataset_version,
        "entries": [*staged_candidate_entries, *v1_confirmation_candidates],
    }
    combined_requests = {
        "dataset_version": args.dataset_version,
        "entries": [*staged_request_entries, *v1_confirmation_requests],
    }
    write_json(output_dir / "candidate_manifest.json", combined_candidates)
    write_json(output_dir / "request_manifest.json", combined_requests)

    train_audit = audit_standardized_file(output_dir / "records_train.jsonl", "train")
    dev_audit = audit_standardized_file(output_dir / "records_dev.jsonl", "dev")
    confirmation_audit = audit_standardized_file(
        output_dir / "records_confirmation.jsonl", "confirmation"
    )
    final_confirmation_candidates = [
        entry for entry in combined_candidates["entries"] if entry["split"] == "confirmation"
    ]
    if final_confirmation_candidates != v1_confirmation_candidates:
        raise ValueError("V1 confirmation candidate projection changed")
    if sha256_file(output_dir / "records_confirmation.jsonl") != sha256_file(v1_records):
        raise ValueError("V1 confirmation records changed")
    if sha256_file(output_dir / "qrels_confirmation.jsonl") != sha256_file(v1_qrels):
        raise ValueError("V1 confirmation qrels changed")

    manifest = {
        "schema_version": 1,
        "dataset_id": "kuaisearch",
        "dataset_version": args.dataset_version,
        "evidence_mode": "frozen_v1_1_train_extension_with_v1_confirmation",
        "protocol": "experiments/motivation_v1_1/protocol.md",
        "source": {
            "raw_dir": args.raw_dir,
            "source_splits_used": ["train"],
            "confirmation_min_timestamp": confirmation_min_ts,
            "staging_manifest": str(staging_report),
            "staging_manifest_sha256": sha256_file(staging_report),
        },
        "selection": {
            "max_requests_before_internal_dev_split": args.max_requests,
            "dev_fraction": args.dev_fraction,
            "max_history_len": args.max_history_len,
            "strictly_before_confirmation": True,
            "confirmation_session_overlap": len(overlap_sessions),
            "train_requests": train_audit["request_count"],
            "internal_dev_requests": dev_audit["request_count"],
            "confirmation_requests": confirmation_audit["request_count"],
        },
        "confirmation_lock": {
            "v1_dir": str(v1_dir),
            "v1_records_sha256": sha256_file(v1_records),
            "v1_qrels_sha256": sha256_file(v1_qrels),
            "v1_candidate_manifest_sha256": sha256_file(v1_dir / "candidate_manifest.json"),
            "v1_request_manifest_sha256": sha256_file(v1_dir / "request_manifest.json"),
            "confirmation_candidate_projection_sha256": _canonical_hash(
                v1_confirmation_candidates
            ),
            "confirmation_request_projection_sha256": _canonical_hash(
                v1_confirmation_requests
            ),
            "copied_byte_identical": True,
        },
        "label_isolation": {
            "train_records_have_source_labels": True,
            "dev_records_label_free": True,
            "confirmation_records_label_free": True,
            "confirmation_qrels_present_but_not_opened_by_materializer": True,
        },
        "outputs": {
            "records_train": train_audit,
            "records_dev": dev_audit,
            "records_confirmation": confirmation_audit,
            "candidate_manifest": {
                "path": str(output_dir / "candidate_manifest.json"),
                "sha256": sha256_file(output_dir / "candidate_manifest.json"),
            },
            "request_manifest": {
                "path": str(output_dir / "request_manifest.json"),
                "sha256": sha256_file(output_dir / "request_manifest.json"),
            },
        },
        "admission_passed": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(report_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_hash(value: object) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    raise SystemExit(main())
