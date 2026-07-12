#!/usr/bin/env python
"""Execute the locked C01 train/calibration/internal-falsifier probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.cect import model_signature  # noqa: E402
from train.data import PackedSplit, prepare_local_arrays, sha256_file  # noqa: E402
from train.engine import run_internal_falsifier, save_checkpoint, train_models  # noqa: E402
from train.integrity import (  # noqa: E402
    CONFIG_PATH,
    assert_gpu_binding,
    assert_source_isolation,
    environment_record,
    load_config,
    set_determinism,
    verify_proposal_lock,
)
from train.smoke import run_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--attempt", type=int, default=1)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite runtime evidence: {path}")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            value,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        handle.write("\n")


def main() -> int:
    args = parse_args()
    os.chdir(REPO_ROOT)
    config = load_config(args.config)
    max_attempts = int(config["training"]["max_attempts"])
    if not 1 <= args.attempt <= max_attempts:
        raise ValueError(f"attempt must be in [1,{max_attempts}]")
    artifact_root = REPO_ROOT / config["paths"]["c01_artifacts"]
    attempt_path = artifact_root / f"implementation_attempt_{args.attempt}.json"
    if attempt_path.exists():
        raise FileExistsError(f"attempt already recorded: {attempt_path}")
    prior = [
        artifact_root / f"implementation_attempt_{number}.json"
        for number in range(1, args.attempt)
    ]
    if any(not path.exists() for path in prior):
        raise ValueError("implementation attempts must be contiguous")
    prior_gpu_hours = 0.0
    for path in prior:
        with path.open("r", encoding="utf-8") as handle:
            prior_record = json.load(handle)
        prior_started = datetime.fromisoformat(prior_record["started_at"])
        prior_finished = datetime.fromisoformat(prior_record["finished_at"])
        prior_gpu_hours += (prior_finished - prior_started).total_seconds() / 3600.0

    started = time.monotonic()
    attempt_record: dict[str, object] = {
        "attempt": args.attempt,
        "candidate_id": config["candidate_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
    }
    try:
        lock = verify_proposal_lock(config)
        source_isolation = assert_source_isolation()
        gpu = assert_gpu_binding(int(config["physical_gpu"]))
        set_determinism(int(config["seed"]))
        smoke = run_smoke(config, "cuda:0")
        print(json.dumps({"event": "gpu_smoke_passed", **smoke}, sort_keys=True), flush=True)

        data_manifest = prepare_local_arrays(config)
        train_split = PackedSplit(config, "train")
        contract, plain, training_log = train_models(config, train_split, "cuda:0")
        total_before_internal = prior_gpu_hours + (time.monotonic() - started) / 3600.0
        if total_before_internal > float(config["training"]["max_gpu_hours"]):
            raise RuntimeError("C01 GPU-hour budget exhausted before internal falsifier")

        config_sha = sha256_file(args.config)
        model_path = REPO_ROOT / config["paths"]["c01_model"]
        plain_path = REPO_ROOT / config["paths"]["c01_plain_model"]
        save_checkpoint(
            model_path,
            contract,
            args.config,
            lock["candidate_hash"],
            training_log,
        )
        save_checkpoint(
            plain_path,
            plain,
            args.config,
            lock["candidate_hash"],
            training_log,
        )

        internal = run_internal_falsifier(
            contract, plain, train_split, config, "cuda:0"
        )
        current_gpu_hours = (time.monotonic() - started) / 3600.0
        total_gpu_hours = prior_gpu_hours + current_gpu_hours
        if total_gpu_hours > float(config["training"]["max_gpu_hours"]):
            raise RuntimeError(
                f"C01 GPU-hour budget exceeded: {total_gpu_hours:.4f}"
            )

        report = {
            "attempt": args.attempt,
            "candidate_hash": lock["candidate_hash"],
            "candidate_id": config["candidate_id"],
            "config_path": str(Path(args.config).resolve()),
            "config_sha256": config_sha,
            "data_manifest": data_manifest,
            "environment": environment_record(),
            "gpu": gpu,
            "gpu_hours_current_attempt_through_internal": current_gpu_hours,
            "gpu_hours_prior_attempts": prior_gpu_hours,
            "gpu_hours_through_internal": total_gpu_hours,
            "internal_falsifier": internal,
            "models": {
                "contract": {
                    "path": str(model_path),
                    "sha256": sha256_file(model_path),
                    "signature": model_signature(contract),
                },
                "plain": {
                    "path": str(plain_path),
                    "sha256": sha256_file(plain_path),
                    "signature": model_signature(plain),
                },
            },
            "qrel_files_read": False,
            "ready_for_dev_scoring": bool(internal["all_passed"]),
            "smoke": smoke,
            "source_isolation": source_isolation,
            "status": "pass" if internal["all_passed"] else "stop_internal_gate_failed",
            "test_files_read": False,
            "training": training_log,
        }
        report_path = artifact_root / "internal_gate_report.json"
        write_json(report_path, report)
        attempt_record.update(
            {
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "internal_gate_report": str(report_path),
                "internal_gate_report_sha256": sha256_file(report_path),
                "status": report["status"],
            }
        )
        write_json(attempt_path, attempt_record)
        print(
            json.dumps(
                {
                    "event": "internal_falsifier_complete",
                    "gate_items": internal["gate_items"],
                    "ready_for_dev_scoring": internal["all_passed"],
                    "report": str(report_path),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0 if internal["all_passed"] else 2
    except Exception as error:
        attempt_record.update(
            {
                "error": f"{type(error).__name__}: {error}",
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "status": "implementation_error",
                "traceback": traceback.format_exc(),
            }
        )
        write_json(attempt_path, attempt_record)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
