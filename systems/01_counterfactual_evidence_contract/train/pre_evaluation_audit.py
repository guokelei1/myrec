#!/usr/bin/env python
"""Authorize the sole shared dev evaluator call from label-free artifacts."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from train.data import iter_jsonl, sha256_file  # noqa: E402
from train.integrity import (  # noqa: E402
    CONFIG_PATH,
    assert_source_isolation,
    load_config,
    verify_proposal_lock,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    lock = verify_proposal_lock(config)
    source = assert_source_isolation()
    run_dir = REPO_ROOT / config["paths"]["c01_run_dir"]
    artifact_root = REPO_ROOT / config["paths"]["c01_artifacts"]
    internal = load_json(artifact_root / "internal_gate_report.json")
    determinism = load_json(artifact_root / "determinism_first1000.json")
    metadata = load_json(run_dir / "metadata.json")
    diagnostics = load_json(run_dir / "score_diagnostics.json")
    if not internal.get("ready_for_dev_scoring"):
        raise PermissionError("internal gate did not pass")
    if not determinism.get("serialized_bytes_identical"):
        raise PermissionError("determinism gate did not pass")
    if metadata.get("candidate_hash") != lock["candidate_hash"]:
        raise ValueError("score metadata candidate hash mismatch")
    if metadata.get("candidate_manifest_sha256") != config["paths"]["candidate_manifest_sha256"]:
        raise ValueError("score metadata candidate manifest mismatch")
    if metadata.get("qrel_files_read_by_scorer") is not False:
        raise ValueError("blind scorer isolation declaration missing")
    if diagnostics.get("no_history_max_absolute_score_difference") != 0.0:
        raise ValueError("dev no-history exact fallback failed")
    if diagnostics.get("no_history_requests") != int(config["screening"]["expected_no_history_requests"]):
        raise ValueError("dev no-history request count mismatch")

    scores_path = run_dir / "scores.jsonl"
    row_count = 0
    request_ids: set[str] = set()
    method_ids: set[str] = set()
    for row in iter_jsonl(scores_path):
        score = float(row["score"])
        if not math.isfinite(score):
            raise FloatingPointError("non-finite score in pre-evaluation audit")
        row_count += 1
        request_ids.add(str(row["request_id"]))
        method_ids.add(str(row["method_id"]))
    if row_count != int(config["screening"]["expected_score_rows"]):
        raise ValueError("pre-evaluation score row count mismatch")
    if len(request_ids) != int(config["screening"]["expected_requests"]):
        raise ValueError("pre-evaluation request count mismatch")
    if method_ids != {config["method_id"]}:
        raise ValueError(f"score method ids differ: {method_ids}")
    scores_sha = sha256_file(scores_path)
    if metadata.get("scores_sha256") != scores_sha or determinism.get("scores_sha256") != scores_sha:
        raise ValueError("score hash differs across frozen artifacts")
    for forbidden_output in (run_dir / "metrics.json", run_dir / "per_request_metrics.jsonl"):
        if forbidden_output.exists():
            raise FileExistsError(f"evaluator output already exists: {forbidden_output}")

    prior_calls = 0
    log_path = REPO_ROOT / "reports" / "dev_eval_log.jsonl"
    if log_path.exists():
        prior_calls = sum(
            1
            for row in iter_jsonl(log_path)
            if str(row.get("run_id")) == config["run_id"]
        )
    if prior_calls != 0:
        raise PermissionError(f"run already has {prior_calls} dev evaluator call(s)")
    audit = {
        "authorized_evaluator_calls_remaining": 1,
        "candidate_hash": lock["candidate_hash"],
        "candidate_manifest_sha256": config["paths"]["candidate_manifest_sha256"],
        "determinism_passed": True,
        "dev_evaluator_calls_before": prior_calls,
        "internal_gate_passed": True,
        "no_history_exact_fallback": True,
        "qrel_files_read": False,
        "request_count": len(request_ids),
        "score_row_count": row_count,
        "scores_sha256": scores_sha,
        "source_isolation": source,
        "status": "authorized_single_shared_dev_evaluator_call",
        "test_files_read": False,
    }
    for path in (artifact_root / "pre_evaluation_audit.json", run_dir / "pre_evaluation_audit.json"):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite pre-evaluation audit: {path}")
        with path.open("w", encoding="utf-8") as handle:
            json.dump(audit, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    print(json.dumps(audit, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
