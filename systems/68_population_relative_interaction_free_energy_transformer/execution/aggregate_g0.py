"""Aggregate the three immutable C68 seed reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import atomic_json, load_config, sha256_file, timestamp, verify_g0_lock  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/g0.yaml")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    _, lock_hash = verify_g0_lock(config)
    seeds = [int(seed) for seed in config["synthetic_G0"]["seeds"]]
    source_reports = []
    reports = []
    for seed in seeds:
        path = REPO_ROOT / args.output_root / f"seed_{seed}_report.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["seed"] != seed or report["g0_lock_sha256"] != lock_hash:
            raise RuntimeError(f"C68 seed report identity mismatch: {seed}")
        reports.append(report)
        source_reports.append({"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path)})
    target = REPO_ROOT / config["paths"]["promoted_report"]
    value = {
        "schema": "myrec.c68.g0.v1",
        "candidate_id": "c68",
        "gate": "data_free_population_relative_interaction_free_energy_G0",
        "created_at": timestamp(),
        "g0_lock_sha256": lock_hash,
        "seeds": seeds,
        "source_reports": source_reports,
        "per_seed": [
            {
                "seed": report["seed"],
                "passed": report["passed"],
                "failed_checks": report["failed_checks"],
                "evaluations": report["evaluations"],
                "mechanics": report["mechanics"],
                "elapsed_seconds": report["elapsed_seconds"],
            }
            for report in reports
        ],
        "passed": all(report["passed"] for report in reports),
        "decision": "authorize_implementation_review" if all(report["passed"] for report in reports) else "failed_G0_terminal",
        "isolation": {
            "repository_data_opened": False,
            "labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))
    print(value["decision"])


if __name__ == "__main__":
    main()
