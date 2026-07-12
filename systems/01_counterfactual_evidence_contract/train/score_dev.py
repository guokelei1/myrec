#!/usr/bin/env python
"""Produce the single blind C01 dev score file after the internal gate passes."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from train.data import PackedSplit, prepare_local_arrays, sha256_file  # noqa: E402
from train.engine import load_checkpoint  # noqa: E402
from train.integrity import (  # noqa: E402
    CONFIG_PATH,
    assert_gpu_binding,
    assert_source_isolation,
    environment_record,
    load_config,
    set_determinism,
    verify_proposal_lock,
)
from train.scoring import score_requests  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: object) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite blind-score evidence: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    lock = verify_proposal_lock(config)
    source_isolation = assert_source_isolation()
    gpu = assert_gpu_binding(int(config["physical_gpu"]))
    set_determinism(int(config["seed"]))
    prepare_local_arrays(config)

    internal_path = REPO_ROOT / config["paths"]["c01_artifacts"] / "internal_gate_report.json"
    internal = load_json(internal_path)
    if not internal.get("ready_for_dev_scoring"):
        raise PermissionError("internal falsifier did not authorize blind dev scoring")
    model_path = REPO_ROOT / config["paths"]["c01_model"]
    model, checkpoint = load_checkpoint(
        model_path, config, lock["candidate_hash"], "cuda:0"
    )
    config_sha = sha256_file(args.config)
    if checkpoint.get("config_sha256") != config_sha:
        raise ValueError("checkpoint/config hash mismatch")

    run_dir = REPO_ROOT / config["paths"]["c01_run_dir"]
    if run_dir.exists():
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    run_dir.mkdir(parents=True)
    shutil.copy2(args.config, run_dir / "config.yaml")
    scores_path = run_dir / "scores.jsonl"
    dev = PackedSplit(config, "dev")
    expected_requests = int(config["screening"]["expected_requests"])
    expected_rows = int(config["screening"]["expected_score_rows"])
    request_count = 0
    row_count = 0
    no_history_count = 0
    no_history_max_abs = 0.0
    repeat_count = 0
    evidence_count = 0
    admitted = 0
    eligible = 0
    started = time.monotonic()
    with scores_path.open("x", encoding="utf-8") as handle:
        for request in score_requests(
            model,
            dev,
            range(len(dev)),
            "cuda:0",
            batch_size=min(int(config["training"]["requests_per_batch"]), 24),
        ):
            request_count += 1
            repeat_count += int(request.exact_present)
            evidence_count += int(request.evidence_present)
            admitted += request.admitted_count
            eligible += request.eligible_event_count
            if not request.history_present:
                no_history_count += 1
                difference = np_abs_max(request.scores, request.base_scores)
                no_history_max_abs = max(no_history_max_abs, difference)
                if difference != 0.0:
                    raise ValueError(
                        f"no-history score contract violated for {request.request_id}: {difference}"
                    )
            for item_id, score in zip(request.candidate_item_ids, request.scores):
                value = float(score)
                if not math.isfinite(value):
                    raise FloatingPointError(
                        f"non-finite dev score for {request.request_id}/{item_id}"
                    )
                handle.write(
                    json.dumps(
                        {
                            "candidate_item_id": item_id,
                            "method_id": config["method_id"],
                            "request_id": request.request_id,
                            "score": value,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                row_count += 1
    inference_seconds = time.monotonic() - started
    if request_count != expected_requests or row_count != expected_rows:
        raise ValueError(
            f"blind score coverage mismatch: requests={request_count}/{expected_requests}, "
            f"rows={row_count}/{expected_rows}"
        )
    expected_no_history = int(config["screening"]["expected_no_history_requests"])
    if no_history_count != expected_no_history:
        raise ValueError(
            f"no-history count mismatch: {no_history_count} != {expected_no_history}"
        )
    diagnostics = {
        "admission_rate": admitted / eligible if eligible else 0.0,
        "admitted_nonexact_candidate_events": admitted,
        "eligible_nonexact_candidate_events": eligible,
        "evidence_present_requests": evidence_count,
        "inference_gpu_hours": inference_seconds / 3600.0,
        "inference_seconds": inference_seconds,
        "no_history_max_absolute_score_difference": no_history_max_abs,
        "no_history_requests": no_history_count,
        "repeat_present_requests": repeat_count,
        "requests": request_count,
        "score_rows": row_count,
    }
    write_json(run_dir / "score_diagnostics.json", diagnostics)
    scores_sha = sha256_file(scores_path)
    metadata = {
        "candidate_hash": lock["candidate_hash"],
        "candidate_id": config["candidate_id"],
        "candidate_manifest_path": config["paths"]["candidate_manifest"],
        "candidate_manifest_sha256": config["paths"]["candidate_manifest_sha256"],
        "checkpoint_path": str(model_path),
        "checkpoint_sha256": sha256_file(model_path),
        "command": (
            "CONDA_ENVS_PATH=/data/gkl/conda_envs CUDA_VISIBLE_DEVICES=0 "
            "conda run -n myrec-c01 python "
            "systems/01_counterfactual_evidence_contract/train/score_dev.py"
        ),
        "config_path": str(Path(args.config).resolve()),
        "config_sha256": config_sha,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_record(),
        "gpu": gpu,
        "internal_gate_report_path": str(internal_path),
        "internal_gate_report_sha256": sha256_file(internal_path),
        "method_id": config["method_id"],
        "qrel_files_read_by_scorer": False,
        "run_id": config["run_id"],
        "scores_sha256": scores_sha,
        "source_isolation": source_isolation,
        "split": "dev",
        "test_files_read": False,
        "true_inputs_only": True,
    }
    write_json(run_dir / "metadata.json", metadata)
    print(
        json.dumps(
            {
                "event": "blind_dev_scoring_complete",
                "diagnostics": diagnostics,
                "run_dir": str(run_dir),
                "scores_sha256": scores_sha,
            },
            sort_keys=True,
        )
    )
    return 0


def np_abs_max(left, right) -> float:
    return float(np.max(np.abs(left - right))) if len(left) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
