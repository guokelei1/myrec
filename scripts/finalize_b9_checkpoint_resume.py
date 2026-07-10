#!/usr/bin/env python
"""Finalize a B9 checkpoint-resumed run without training or evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.prodsearch_adapter import (  # noqa: E402
    OFFICIAL_COMMIT,
    convert_prodsearch_ranklist,
)
from myrec.utils.hashing import sha256_file  # noqa: E402
from myrec.utils.jsonl import write_json  # noqa: E402


VALID_RE = re.compile(r"Epoch (?P<epoch>\d+): MRR:(?P<mrr>[0-9.eE+-]+)")
COPY_RE = re.compile(r"Copying .*model_epoch_(?P<epoch>\d+)\.ckpt to checkpoint .*model_best\.ckpt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baselines/b9_prodsearch.yaml")
    parser.add_argument("--model", required=True, choices=["zam", "tem"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--resume-from-epoch", type=int, required=True)
    parser.add_argument("--pre-resume-best-mrr", type=float, required=True)
    parser.add_argument("--resume-command-path", default=None)
    parser.add_argument("--resume-stdout", default=None)
    parser.add_argument("--resume-stderr", default=None)
    parser.add_argument("--cuda-visible-devices", required=True)
    parser.add_argument("--interrupted-compatibility-probe", action="store_true")
    parser.add_argument("--runs-dir", default="runs")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_training_evidence(path: Path, resume_from_epoch: int) -> dict:
    validations: list[dict[str, float | int]] = []
    copied_best_epochs: list[int] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = VALID_RE.search(line)
        if match:
            epoch = int(match.group("epoch"))
            if epoch > resume_from_epoch:
                validations.append({"epoch": epoch, "mrr": float(match.group("mrr"))})
        copy_match = COPY_RE.search(line)
        if copy_match and int(copy_match.group("epoch")) > resume_from_epoch:
            copied_best_epochs.append(int(copy_match.group("epoch")))
    if not validations:
        raise ValueError(f"no post-resume validation entries in {path}")
    completed_epochs = {int(row["epoch"]) for row in validations}
    if 20 not in completed_epochs:
        raise ValueError(f"epoch 20 validation missing from {path}")
    best = max(validations, key=lambda row: float(row["mrr"]))
    if not copied_best_epochs or copied_best_epochs[-1] != best["epoch"]:
        raise ValueError(
            f"model_best log mismatch: copied={copied_best_epochs[-1:]}, expected={best['epoch']}"
        )
    return {
        "post_resume_validations": validations,
        "best_post_resume_epoch": int(best["epoch"]),
        "best_post_resume_mrr": float(best["mrr"]),
        "completed_epoch_20": True,
    }


def main() -> int:
    args = parse_args()
    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["models"][args.model]
    run_dir = Path(args.runs_dir) / args.run_id
    official_dir = run_dir / "official"
    metadata_path = run_dir / "metadata.json"
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        raise ValueError(f"refusing to finalize an already evaluated run: {metrics_path}")
    initial_metadata = _read_json(metadata_path)
    if initial_metadata.get("status") != "failed":
        raise ValueError(f"expected failed pre-recovery metadata: {metadata_path}")
    initial_stdout = run_dir / "stdout.log"
    initial_stderr = run_dir / "stderr.log"
    for path in (initial_stdout, initial_stderr):
        if not path.exists():
            raise FileNotFoundError(path)

    training = _parse_training_evidence(
        official_dir / "train.log", args.resume_from_epoch
    )
    if training["best_post_resume_mrr"] < args.pre_resume_best_mrr:
        raise ValueError(
            "post-resume train-only best is below the preserved pre-resume checkpoint; "
            "select and rescore the preserved checkpoint explicitly"
        )

    checkpoint = official_dir / "model_best.ckpt"
    preserved_pre_resume_checkpoint = (
        official_dir / f"model_best_epoch{args.resume_from_epoch}_pre_resume.ckpt"
    )
    ranklist = official_dir / "official.ranklist"
    if (
        not checkpoint.exists()
        or not preserved_pre_resume_checkpoint.exists()
        or not ranklist.exists()
    ):
        raise FileNotFoundError(f"missing final checkpoint/ranklist under {official_dir}")

    candidate_manifest = Path(config["candidate_manifest"])
    conversion = convert_prodsearch_ranklist(
        ranklist,
        Path(config["materialized_root"]) / "dev_request_map.jsonl",
        run_dir / "scores.jsonl",
        method_id=model_config["method_id"],
        candidate_manifest_path=candidate_manifest,
        split="dev",
    )
    shutil.copyfile(config_path, run_dir / "config_snapshot.yaml")

    resume_command_path = Path(
        args.resume_command_path or run_dir / "resume_command.sh"
    )
    resume_stdout = Path(args.resume_stdout) if args.resume_stdout else run_dir / "resume_stdout.log"
    resume_stderr = Path(args.resume_stderr) if args.resume_stderr else run_dir / "resume_stderr.log"
    for path in (resume_command_path, resume_stdout, resume_stderr):
        if not path.exists():
            raise FileNotFoundError(path)

    metadata = {
        "status": "scored_not_evaluated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "official_commit": OFFICIAL_COMMIT,
        "implementation_type": config["identity_label"],
        "model": args.model,
        "method_id": model_config["method_id"],
        "seed": args.seed,
        "command": initial_metadata.get("command", []),
        "environment_name": config["environment_name"],
        "python": platform.python_version(),
        "hostname": platform.node(),
        "candidate_manifest_path": str(candidate_manifest),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "conversion": conversion,
        "selected_checkpoint": str(checkpoint),
        "selected_checkpoint_sha256": sha256_file(checkpoint),
        "training_and_initial_failure": {
            "metadata": initial_metadata,
            "stdout_path": str(initial_stdout),
            "stdout_sha256": sha256_file(initial_stdout),
            "stderr_path": str(initial_stderr),
            "stderr_sha256": sha256_file(initial_stderr),
        },
        "recovery": {
            "status": "passed",
            "strategy": "resume_from_complete_checkpoint_without_retraining_prior_epochs",
            "authorization": "user approved checkpoint resume without full restart in chat on 2026-07-10",
            "resume_from_epoch": args.resume_from_epoch,
            "pre_resume_best_mrr": args.pre_resume_best_mrr,
            "preserved_pre_resume_checkpoint": str(preserved_pre_resume_checkpoint),
            "preserved_pre_resume_checkpoint_sha256": sha256_file(
                preserved_pre_resume_checkpoint
            ),
            **training,
            "resume_command_path": str(resume_command_path),
            "resume_command_sha256": sha256_file(resume_command_path),
            "resume_command": resume_command_path.read_text(encoding="utf-8").strip(),
            "resume_working_directory": config["upstream"]["local_dir"],
            "resume_stdout_path": str(resume_stdout),
            "resume_stdout_sha256": sha256_file(resume_stdout),
            "resume_stderr_path": str(resume_stderr),
            "resume_stderr_sha256": sha256_file(resume_stderr),
            "interrupted_compatibility_probe_before_final_resume": args.interrupted_compatibility_probe,
            "implementation_fixes": [
                "explicit torch.long dtype for all-empty-history index batches",
                "dimension-specific squeeze for singleton shuffled training batches",
            ],
            "rng_state_restored": False,
            "rng_caveat": (
                "The official checkpoint stores model and optimizer state but not Python/NumPy/"
                "PyTorch RNG states. The resumed trajectory is therefore not bit-identical to an "
                "uninterrupted run; final selection uses only the upstream train-only validation."
            ),
        },
        "torch_cuda_visible_devices": args.cuda_visible_devices,
        "returncode": 0,
        "qrels_read": False,
        "records_test_read": False,
        "shared_evaluator_pending": True,
    }
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "status": metadata["status"],
                "run_id": args.run_id,
                "best_post_resume_epoch": training["best_post_resume_epoch"],
                "best_post_resume_mrr": training["best_post_resume_mrr"],
                "scores_sha256": conversion["scores_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
