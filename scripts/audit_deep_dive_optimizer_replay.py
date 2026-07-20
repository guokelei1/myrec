#!/usr/bin/env python3
"""Persist one immutable D7 step-500 optimizer replay binding audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from myrec.baselines.motivation_v12_ranker import _git_revision, _validate_run_id
from myrec.mechanism.optimizer_replay_binding import load_bound_step500_state
from myrec.utils.hashing import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-id",
        choices=("q2_recranker_generalqwen", "q3_tallrec_generalqwen"),
        required=True,
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    _validate_run_id(args.run_id)
    run_dir = Path(args.runs_dir) / args.run_id
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"optimizer audit run is not empty: {run_dir}")
    state, audit = load_bound_step500_state(args.method_id)
    del state
    root = Path(__file__).resolve().parents[1]
    implementation_files = [
        root / "src/myrec/mechanism/optimizer_replay_binding.py",
        root / "src/myrec/mechanism/optimizer_replay_math.py",
        root / "scripts/audit_deep_dive_optimizer_replay.py",
    ]
    payload = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d7_optimizer_binding_audit",
        "run_id": args.run_id,
        "method_id": args.method_id,
        "status": "completed",
        "evidence_mode": "mechanical_binding_audit_non_result",
        "result_eligible": False,
        "qrels_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "optimizer_steps_performed": 0,
        "binding": audit,
        "implementation_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for path in implementation_files
        ],
        "code_revision": _git_revision(),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    temporary = run_dir / f".metadata.json.tmp-{os.getpid()}"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, run_dir / "metadata.json")
    print(
        json.dumps(
            {
                "run_id": args.run_id,
                "method_id": args.method_id,
                "status": payload["status"],
                "optimizer_parameter_count": audit["optimizer_parameter_count"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
