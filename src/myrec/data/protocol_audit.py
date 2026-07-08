"""C1 protocol audit for standardized PPS datasets."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from myrec.eval.evaluator import evaluate_run
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


DEFAULT_CANARY_RUN_IDS = {
    "random": "20260708_kuaisearch_random_c1",
    "label_shuffle": "20260708_kuaisearch_label_shuffle_c1",
    "positive_title_leak": "20260708_kuaisearch_positive_title_leak_c1",
}


def audit_c1_protocol(
    standardized_dir: str | Path,
    report_path: str | Path,
    runs_dir: str | Path = "runs",
    tmp_dir: str | Path = "tmp/c1_protocol_audit",
    canary_run_ids: dict[str, str] | None = None,
    run_unit_tests: bool = True,
) -> dict[str, Any]:
    standardized_dir = Path(standardized_dir)
    report_path = Path(report_path)
    runs_dir = Path(runs_dir)
    canary_run_ids = canary_run_ids or DEFAULT_CANARY_RUN_IDS

    manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    split_manifest_path = standardized_dir / "split_manifest.json"
    c0_report_path = Path("reports/pps_c0_data_audit.json")

    manifest = _read_json(manifest_path)
    record_scans = {
        split: _scan_records(standardized_dir / f"records_{split}.jsonl", split)
        for split in ("train", "dev", "test")
    }
    qrel_scans = {
        split: _scan_qrels(standardized_dir / f"qrels_{split}.jsonl")
        for split in ("dev", "test")
    }
    candidate_scan = _scan_candidate_manifest(candidate_manifest_path)
    split_manifest_scan = _scan_split_manifest(split_manifest_path)
    file_checks = _build_file_checks(manifest, manifest_path, record_scans, qrel_scans)
    split_summary = _build_split_summary(record_scans, qrel_scans)

    checks = {
        "split_manifest": _check_split_manifest(record_scans, split_manifest_scan),
        "candidate_manifest": _check_candidate_manifest(record_scans, candidate_scan),
        "label_isolation": _check_label_isolation(record_scans),
        "qrels_consistency": _check_qrels_consistency(qrel_scans, candidate_scan),
        "history_causality": _check_history_causality(record_scans),
        "metric_unit_tests": _run_metric_unit_tests(run_unit_tests),
        "candidate_hash_assertion": _run_candidate_hash_check(
            standardized_dir=standardized_dir,
            candidate_manifest_path=candidate_manifest_path,
            tmp_dir=tmp_dir,
        ),
        "canaries": _check_canaries(runs_dir, canary_run_ids),
    }

    report = {
        "report_id": "pps_c1_protocol",
        "dataset_id": manifest.get("dataset_id", "kuaisearch"),
        "dataset_version": manifest.get("dataset_version", "v0_lite"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "standardized_dir": str(standardized_dir),
        "standardized_manifest_path": str(manifest_path),
        "standardized_manifest_sha256": sha256_file(manifest_path),
        "candidate_manifest_path": str(candidate_manifest_path),
        "candidate_manifest_sha256": candidate_scan["sha256"],
        "history_source": manifest.get("history_source", {}),
        "c0_report": _c0_summary(c0_report_path),
        "split_summary": split_summary,
        "files": file_checks,
        "checks": checks,
    }
    report["overall_status"] = (
        "passed" if all(check["status"] == "passed" for check in checks.values()) else "failed"
    )
    write_json(report_path, report)
    return report


def _scan_records(path: Path, split: str) -> dict[str, Any]:
    digest = hashlib.sha256()
    request_ids: set[str] = set()
    user_ids: set[str] = set()
    candidate_counts: list[int] = []
    history_lengths: list[int] = []
    text_coverages: list[float] = []
    label_field_candidates = 0
    clicked_candidates = 0
    purchased_candidates = 0
    clicked_requests = 0
    purchased_requests = 0
    history_future_violations = 0
    history_future_examples = []

    with path.open("rb") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            request_id = str(row["request_id"])
            request_ids.add(request_id)
            user_ids.add(str(row["user_id"]))
            request_ts = int(row["ts"])
            candidates = row.get("candidates", [])
            history = row.get("history", [])
            candidate_counts.append(len(candidates))
            history_lengths.append(len(history))
            text_coverages.append(float(row.get("masks", {}).get("text_coverage", 0.0)))

            row_clicked = 0
            row_purchased = 0
            for candidate in candidates:
                if "clicked" in candidate or "purchased" in candidate:
                    label_field_candidates += 1
                row_clicked += int(candidate.get("clicked", 0) or 0)
                row_purchased += int(candidate.get("purchased", 0) or 0)
            if row_clicked:
                clicked_requests += 1
            if row_purchased:
                purchased_requests += 1
            clicked_candidates += row_clicked
            purchased_candidates += row_purchased

            for event in history:
                event_ts = int(event["ts"])
                if event_ts >= request_ts:
                    history_future_violations += 1
                    if len(history_future_examples) < 10:
                        history_future_examples.append(
                            {
                                "event_item_id": str(event["item_id"]),
                                "event_ts": event_ts,
                                "line_no": line_no,
                                "request_id": request_id,
                                "request_ts": request_ts,
                                "split": split,
                            }
                        )

    total_candidates = sum(candidate_counts)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "request_ids": request_ids,
        "user_count": len(user_ids),
        "request_count": len(request_ids),
        "candidate_count": total_candidates,
        "candidate_count_distribution": _summarize_int(candidate_counts),
        "history_length_distribution": _summarize_int(history_lengths),
        "text_coverage_distribution": _summarize_float(text_coverages),
        "label_field_candidates": label_field_candidates,
        "clicked_candidates": clicked_candidates,
        "purchased_candidates": purchased_candidates,
        "clicked_requests": clicked_requests,
        "purchased_requests": purchased_requests,
        "clicked_candidate_rate": clicked_candidates / total_candidates if total_candidates else 0.0,
        "purchased_candidate_rate": purchased_candidates / total_candidates if total_candidates else 0.0,
        "clicked_request_rate": clicked_requests / len(request_ids) if request_ids else 0.0,
        "purchased_request_rate": purchased_requests / len(request_ids) if request_ids else 0.0,
        "history_future_violations": history_future_violations,
        "history_future_examples": history_future_examples,
    }


def _scan_qrels(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    request_ids: set[str] = set()
    clicked_by_request: dict[str, set[str]] = {}
    purchased_by_request: dict[str, set[str]] = {}
    clicked_items = 0
    purchased_items = 0
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            row = json.loads(raw_line)
            request_id = str(row["request_id"])
            clicked = set(str(item_id) for item_id in row.get("clicked", []))
            purchased = set(str(item_id) for item_id in row.get("purchased", []))
            request_ids.add(request_id)
            clicked_by_request[request_id] = clicked
            purchased_by_request[request_id] = purchased
            clicked_items += len(clicked)
            purchased_items += len(purchased)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "request_ids": request_ids,
        "request_count": len(request_ids),
        "clicked_by_request": clicked_by_request,
        "purchased_by_request": purchased_by_request,
        "clicked_items": clicked_items,
        "purchased_items": purchased_items,
        "clicked_requests": sum(1 for values in clicked_by_request.values() if values),
        "purchased_requests": sum(1 for values in purchased_by_request.values() if values),
    }


def _scan_candidate_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    by_split: dict[str, dict[str, Any]] = {}
    candidate_sets: dict[str, dict[str, set[str]]] = {"dev": {}, "test": {}}
    for entry in manifest["entries"]:
        split = str(entry["split"])
        request_id = str(entry["request_id"])
        item_ids = [str(item_id) for item_id in entry["candidate_item_ids"]]
        stats = by_split.setdefault(
            split,
            {
                "request_ids": set(),
                "candidate_counts": [],
                "candidate_count": 0,
            },
        )
        stats["request_ids"].add(request_id)
        stats["candidate_counts"].append(len(item_ids))
        stats["candidate_count"] += len(item_ids)
        if split in candidate_sets:
            candidate_sets[split][request_id] = set(item_ids)
    for stats in by_split.values():
        stats["request_count"] = len(stats["request_ids"])
        stats["candidate_count_distribution"] = _summarize_int(stats["candidate_counts"])
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "by_split": by_split,
        "candidate_sets": candidate_sets,
    }


def _scan_split_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "request_ids": {split: set(str(request_id) for request_id in request_ids) for split, request_ids in manifest.items()},
        "counts": {split: len(request_ids) for split, request_ids in manifest.items()},
    }


def _build_split_summary(
    record_scans: dict[str, dict[str, Any]],
    qrel_scans: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    summary = {}
    for split, scan in record_scans.items():
        total_candidates = scan["candidate_count"]
        row = {
            "request_count": scan["request_count"],
            "user_count": scan["user_count"],
            "candidate_count": total_candidates,
            "candidate_count_distribution": scan["candidate_count_distribution"],
            "history_length_distribution": scan["history_length_distribution"],
            "text_coverage_distribution": scan["text_coverage_distribution"],
        }
        if split == "train":
            row.update(
                {
                    "clicked_candidate_rate": scan["clicked_candidate_rate"],
                    "clicked_request_rate": scan["clicked_request_rate"],
                    "purchased_candidate_rate": scan["purchased_candidate_rate"],
                    "purchased_request_rate": scan["purchased_request_rate"],
                    "label_source": "records_train candidate labels",
                }
            )
        else:
            qrels = qrel_scans[split]
            row.update(
                {
                    "clicked_candidate_rate": qrels["clicked_items"] / total_candidates
                    if total_candidates
                    else 0.0,
                    "clicked_request_rate": qrels["clicked_requests"] / scan["request_count"]
                    if scan["request_count"]
                    else 0.0,
                    "purchased_candidate_rate": qrels["purchased_items"] / total_candidates
                    if total_candidates
                    else 0.0,
                    "purchased_request_rate": qrels["purchased_requests"] / scan["request_count"]
                    if scan["request_count"]
                    else 0.0,
                    "label_source": f"qrels_{split}.jsonl",
                }
            )
        summary[split] = row
    return summary


def _build_file_checks(
    manifest: dict[str, Any],
    manifest_path: Path,
    record_scans: dict[str, dict[str, Any]],
    qrel_scans: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    files = {
        "standardized_manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    }
    expected = manifest.get("outputs", {}).get("files", {})
    scan_sha_by_name = {
        "records_train": record_scans["train"]["sha256"],
        "records_dev": record_scans["dev"]["sha256"],
        "records_test": record_scans["test"]["sha256"],
        "qrels_dev": qrel_scans["dev"]["sha256"],
        "qrels_test": qrel_scans["test"]["sha256"],
    }
    for name, info in expected.items():
        path = Path(info["path"])
        current_sha = scan_sha_by_name.get(name)
        if current_sha is None and name in {"candidate_manifest", "split_manifest"}:
            current_sha = sha256_file(path)
        files[name] = {
            "path": str(path),
            "manifest_sha256": info.get("sha256"),
            "current_sha256": current_sha,
            "sha256_matches_manifest": current_sha == info.get("sha256") if current_sha else None,
            "rows": info.get("rows"),
            "size_bytes": info.get("size_bytes"),
        }
    return files


def _check_split_manifest(
    record_scans: dict[str, dict[str, Any]],
    split_manifest_scan: dict[str, Any],
) -> dict[str, Any]:
    mismatches = {}
    for split, scan in record_scans.items():
        expected = scan["request_ids"]
        recorded = split_manifest_scan["request_ids"].get(split, set())
        if expected != recorded:
            mismatches[split] = {
                "missing": sorted(expected - recorded)[:10],
                "extra": sorted(recorded - expected)[:10],
            }
    return {
        "status": "passed" if not mismatches else "failed",
        "evidence": {
            "split_manifest_path": split_manifest_scan["path"],
            "split_manifest_sha256": split_manifest_scan["sha256"],
            "counts": split_manifest_scan["counts"],
            "mismatches": mismatches,
        },
    }


def _check_candidate_manifest(
    record_scans: dict[str, dict[str, Any]],
    candidate_scan: dict[str, Any],
) -> dict[str, Any]:
    mismatches = {}
    by_split_evidence = {}
    for split, scan in record_scans.items():
        manifest_stats = candidate_scan["by_split"].get(split, {})
        expected = scan["request_ids"]
        recorded = manifest_stats.get("request_ids", set())
        if expected != recorded:
            mismatches[split] = {
                "missing": sorted(expected - recorded)[:10],
                "extra": sorted(recorded - expected)[:10],
            }
        if scan["candidate_count"] != manifest_stats.get("candidate_count", 0):
            mismatches.setdefault(split, {})["candidate_count"] = {
                "records": scan["candidate_count"],
                "manifest": manifest_stats.get("candidate_count", 0),
            }
        by_split_evidence[split] = {
            "record_requests": scan["request_count"],
            "manifest_requests": manifest_stats.get("request_count", 0),
            "record_candidates": scan["candidate_count"],
            "manifest_candidates": manifest_stats.get("candidate_count", 0),
            "candidate_count_distribution": manifest_stats.get("candidate_count_distribution", {}),
        }
    return {
        "status": "passed" if not mismatches else "failed",
        "evidence": {
            "candidate_manifest_path": candidate_scan["path"],
            "candidate_manifest_sha256": candidate_scan["sha256"],
            "by_split": by_split_evidence,
            "mismatches": mismatches,
        },
    }


def _check_label_isolation(record_scans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    violations = {
        split: record_scans[split]["label_field_candidates"]
        for split in ("dev", "test")
        if record_scans[split]["label_field_candidates"]
    }
    return {
        "status": "passed" if not violations else "failed",
        "evidence": {
            "method": "structured JSON scan of records_dev/test candidates",
            "violations": violations,
            "dev_label_field_candidates": record_scans["dev"]["label_field_candidates"],
            "test_label_field_candidates": record_scans["test"]["label_field_candidates"],
        },
    }


def _check_qrels_consistency(
    qrel_scans: dict[str, dict[str, Any]],
    candidate_scan: dict[str, Any],
) -> dict[str, Any]:
    errors = {}
    for split, qrels in qrel_scans.items():
        candidate_sets = candidate_scan["candidate_sets"][split]
        request_errors = []
        item_errors = []
        if set(candidate_sets) != qrels["request_ids"]:
            request_errors.append(
                {
                    "missing_qrels": sorted(set(candidate_sets) - qrels["request_ids"])[:10],
                    "extra_qrels": sorted(qrels["request_ids"] - set(candidate_sets))[:10],
                }
            )
        for request_id in sorted(qrels["request_ids"]):
            candidates = candidate_sets.get(request_id, set())
            outside = (
                qrels["clicked_by_request"].get(request_id, set())
                | qrels["purchased_by_request"].get(request_id, set())
            ) - candidates
            if outside and len(item_errors) < 10:
                item_errors.append({"request_id": request_id, "outside_candidates": sorted(outside)[:10]})
        if request_errors or item_errors:
            errors[split] = {"request_errors": request_errors, "item_errors": item_errors}
    return {
        "status": "passed" if not errors else "failed",
        "evidence": {
            "qrels_dev_sha256": qrel_scans["dev"]["sha256"],
            "qrels_test_sha256": qrel_scans["test"]["sha256"],
            "errors": errors,
        },
    }


def _check_history_causality(record_scans: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_violations = sum(scan["history_future_violations"] for scan in record_scans.values())
    return {
        "status": "passed" if total_violations == 0 else "failed",
        "evidence": {
            "rule": "every standardized history event must have event.ts < request.ts",
            "violations": {split: scan["history_future_violations"] for split, scan in record_scans.items()},
            "examples": [
                example
                for scan in record_scans.values()
                for example in scan["history_future_examples"]
            ][:10],
        },
    }


def _run_metric_unit_tests(run_unit_tests: bool) -> dict[str, Any]:
    if not run_unit_tests:
        return {"status": "skipped", "evidence": {"reason": "run_unit_tests=false"}}
    repo_root = Path(__file__).resolve().parents[3]
    command = [sys.executable, "-m", "unittest", "discover", "-s", "tests"]
    proc = subprocess.run(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    output_lines = proc.stdout.splitlines()
    return {
        "status": "passed" if proc.returncode == 0 else "failed",
        "evidence": {
            "command": " ".join(command),
            "returncode": proc.returncode,
            "output_tail": output_lines[-20:],
        },
    }


def _run_candidate_hash_check(
    standardized_dir: Path,
    candidate_manifest_path: Path,
    tmp_dir: str | Path,
) -> dict[str, Any]:
    tmp_dir = Path(tmp_dir)
    run_id = "bad_candidate_hash"
    run_dir = tmp_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir / "metadata.json",
        {
            "candidate_manifest_sha256": "bad_hash",
            "method_id": "candidate_hash_negative_control",
            "run_id": run_id,
        },
    )
    try:
        evaluate_run(
            run_id=run_id,
            split="dev",
            candidate_manifest_path=candidate_manifest_path,
            standardized_dir=standardized_dir,
            runs_dir=tmp_dir,
            dev_eval_log_path=tmp_dir / "dev_eval_log.jsonl",
        )
    except ValueError as exc:
        message = str(exc)
        passed = "candidate_manifest_sha256 mismatch" in message
        return {
            "status": "passed" if passed else "failed",
            "evidence": {"expected_failure": "candidate_manifest_sha256 mismatch", "message": message},
        }
    except Exception as exc:  # noqa: BLE001 - audit must record unexpected failure mode.
        return {
            "status": "failed",
            "evidence": {"expected_failure": "candidate_manifest_sha256 mismatch", "message": repr(exc)},
        }
    return {
        "status": "failed",
        "evidence": {"expected_failure": "candidate_manifest_sha256 mismatch", "message": "evaluator accepted bad hash"},
    }


def _check_canaries(runs_dir: Path, canary_run_ids: dict[str, str]) -> dict[str, Any]:
    metrics = {}
    missing = []
    for name, run_id in canary_run_ids.items():
        metrics_path = runs_dir / run_id / "metrics.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
            continue
        metrics[name] = _read_json(metrics_path)
    if missing:
        return {"status": "failed", "evidence": {"missing_metrics": missing, "metrics": metrics}}
    random_ndcg = float(metrics["random"]["ndcg@10"])
    shuffle_ndcg = float(metrics["label_shuffle"]["ndcg@10"])
    leak_ndcg = float(metrics["positive_title_leak"]["ndcg@10"])
    random_like_ceiling = max(random_ndcg, shuffle_ndcg)
    passed = leak_ndcg >= 0.8 and (leak_ndcg - random_like_ceiling) >= 0.5
    return {
        "status": "passed" if passed else "failed",
        "evidence": {
            "expectation": (
                "random/shuffled-label canaries remain low while positive-title leak canary surges"
            ),
            "threshold": {
                "positive_title_leak_ndcg_min": 0.8,
                "positive_title_leak_minus_random_like_min": 0.5,
            },
            "run_ids": canary_run_ids,
            "metrics": {
                name: {
                    "method_id": row["method_id"],
                    "ndcg@10": row["ndcg@10"],
                    "mrr": row["mrr"],
                    "recall@10": row["recall@10"],
                    "purchase_ndcg@10": row["purchase_ndcg@10"],
                }
                for name, row in metrics.items()
            },
        },
    }


def _c0_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "status": "missing"}
    report = _read_json(path)
    leakage = report.get("checks", {}).get("history_future_leakage", {})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "overall_status": report.get("overall_status"),
        "history_future_leakage": leakage,
    }


def _summarize_int(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
        "total": sum(ordered),
    }


def _summarize_float(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": ordered[len(ordered) // 2],
        "mean": sum(ordered) / len(ordered),
        "max": ordered[-1],
    }


def _read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)
