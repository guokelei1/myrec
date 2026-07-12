"""One-shot executor for the locked C08 G1 learned synthetic falsifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import torch

from g1_protocol import (
    FROZEN,
    MECHANISMS,
    config_dict,
    corrupt_supported,
    evaluate_model,
    generate_split,
    initialized_models,
    item_recurrence_accuracy,
    make_batch_schedule,
    tensor_state_hash,
    train_model,
)


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution_lock(lock_path: Path) -> dict[str, object]:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    entries: dict[str, str] = lock["files_sha256"]
    aggregate_lines: list[str] = []
    for relative in sorted(entries):
        path = ROOT / relative
        if not path.is_file() or _sha256(path) != entries[relative]:
            raise RuntimeError(f"execution-lock mismatch: {relative}")
        aggregate_lines.append(f"{entries[relative]}  {relative}\n")
    aggregate = hashlib.sha256("".join(aggregate_lines).encode("utf-8")).hexdigest()
    if aggregate != lock["aggregate_sha256"]:
        raise RuntimeError("execution-lock aggregate mismatch")
    if lock["protocol_constants"] != config_dict():
        raise RuntimeError("locked constants differ from executable constants")
    return lock


def _write_json_exclusive(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_json_replace(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_seed(base_seed: int) -> dict[str, object]:
    train = generate_split(base_seed, "train")
    evaluation = generate_split(base_seed, "eval")
    repeat = evaluation.subset(torch.nonzero(evaluation.kind.eq(0)).view(-1))
    supported = evaluation.subset(torch.nonzero(evaluation.kind.eq(1)).view(-1))
    schedule = make_batch_schedule(base_seed)
    models = initialized_models(base_seed)
    initial_hashes = {name: tensor_state_hash(model) for name, model in models.items()}
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError("control initializations differ")
    parameter_counts = {
        name: sum(parameter.numel() for parameter in model.parameters())
        for name, model in models.items()
    }
    if len(set(parameter_counts.values())) != 1:
        raise RuntimeError("control parameter counts differ")

    controls: dict[str, object] = {}
    trained_models = {}
    for mechanism in MECHANISMS:
        model = models[mechanism]
        training = train_model(model, train, schedule)
        _, repeat_metrics = evaluate_model(model, repeat, base_seed)
        _, supported_metrics = evaluate_model(model, supported, base_seed)
        controls[mechanism] = {
            "training": training,
            "repeat": repeat_metrics,
            "supported_nonrepeat": supported_metrics,
            "final_state_sha256": tensor_state_hash(model),
        }
        trained_models[mechanism] = model
        print(
            json.dumps(
                {
                    "seed": base_seed,
                    "mechanism": mechanism,
                    "repeat_accuracy": repeat_metrics["accuracy"],
                    "supported_accuracy": supported_metrics["accuracy"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rwpu = trained_models["rwpu"]
    _, clean_supported = evaluate_model(rwpu, supported, base_seed)
    corruptions: dict[str, object] = {}
    for corruption in ("wrong_history", "shuffled_event", "query_mask", "disjoint"):
        corrupted = corrupt_supported(supported, base_seed, corruption)
        _, metrics = evaluate_model(rwpu, corrupted, base_seed)
        corruptions[corruption] = metrics

    empty = supported.subset(torch.arange(min(64, supported.query.shape[0])))
    empty.history = torch.zeros_like(empty.history)
    empty.history_mask = torch.zeros_like(empty.history_mask)
    rwpu.eval()
    with torch.no_grad():
        empty_scores = rwpu(**empty.as_model_inputs())
        query_only = rwpu.query_only(empty.query, empty.candidates)
    fallback_equal = torch.equal(empty_scores, query_only)
    fallback_max_error = float((empty_scores - query_only).abs().max())

    permutation_source = supported.subset(torch.arange(min(64, supported.query.shape[0])))
    permutation = torch.arange(FROZEN.candidate_count - 1, -1, -1)
    permuted = permutation_source.subset(torch.arange(permutation_source.query.shape[0]))
    permuted.candidates = permutation_source.candidates[:, permutation]
    permuted.candidate_ids = permutation_source.candidate_ids[:, permutation]
    with torch.no_grad():
        original_scores = rwpu(**permutation_source.as_model_inputs())
        permuted_scores = rwpu(**permuted.as_model_inputs())
    permutation_error = float((permuted_scores - original_scores[:, permutation]).abs().max())

    item_accuracy = item_recurrence_accuracy(repeat, base_seed)
    rwpu_repeat = controls["rwpu"]["repeat"]["accuracy"]
    rwpu_supported = controls["rwpu"]["supported_nonrepeat"]["accuracy"]
    ordinary_supported = controls["ordinary"]["supported_nonrepeat"]["accuracy"]
    best_control = max(
        controls[name]["supported_nonrepeat"]["accuracy"]
        for name in ("ordinary", "attention", "pooled_ffn")
    )
    clean_margin = clean_supported["mean_target_margin"]
    retention = {
        name: metrics["mean_target_margin"] / clean_margin
        if clean_margin != 0.0
        else float("inf")
        for name, metrics in corruptions.items()
    }
    conditions = {
        "repeat_noninferior": rwpu_repeat
        >= item_accuracy - FROZEN.repeat_noninferiority_pp,
        "supported_beats_best_control": rwpu_supported - best_control
        >= FROZEN.supported_best_control_advantage_pp,
        "supported_beats_ordinary": rwpu_supported - ordinary_supported
        >= FROZEN.supported_ordinary_advantage_pp,
        "clean_margin_positive": clean_margin > 0.0,
        "all_corruption_retention_at_most_threshold": all(
            value <= FROZEN.corruption_margin_retention_max for value in retention.values()
        ),
        "exact_empty_fallback": fallback_equal and fallback_max_error == 0.0,
        "candidate_permutation": permutation_error <= FROZEN.permutation_max_error,
        "finite_training_and_scores": True,
    }

    return {
        "seed": base_seed,
        "parameter_count": next(iter(parameter_counts.values())),
        "parameter_counts": parameter_counts,
        "shared_initial_state_sha256": next(iter(initial_hashes.values())),
        "control_initial_state_sha256": initial_hashes,
        "item_recurrence_repeat_accuracy": item_accuracy,
        "controls": controls,
        "corruptions": corruptions,
        "corruption_margin_retention": retention,
        "fallback": {
            "request_count": int(empty.query.shape[0]),
            "bitwise_equal": fallback_equal,
            "max_abs_error": fallback_max_error,
        },
        "candidate_permutation_max_abs_error": permutation_error,
        "conditions": conditions,
        "passed": all(conditions.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--execution-lock",
        default="G1_EXECUTION_LOCK.json",
        help="Locked manifest; no other protocol override is accepted.",
    )
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError('G1 requires CUDA_VISIBLE_DEVICES=""')
    if torch.cuda.is_available():
        raise RuntimeError("CUDA must be unavailable for G1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)

    lock_path = (ROOT / args.execution_lock).resolve()
    if lock_path.parent != ROOT:
        raise RuntimeError("execution lock must be candidate-local")
    lock = verify_execution_lock(lock_path)
    run_dir = ROOT / "runs" / f"g1_{lock['aggregate_sha256'][:16]}"
    if run_dir.exists():
        raise RuntimeError(f"one-shot G1 run path already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    started = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": _sha256(lock_path),
        "aggregate_sha256": lock["aggregate_sha256"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "config": config_dict(),
    }
    _write_json_exclusive(run_dir / "RUN_STARTED.json", started)

    seed_results: list[dict[str, object]] = []
    try:
        for seed in FROZEN.seeds:
            result = run_seed(seed)
            _write_json_exclusive(run_dir / f"seed_{seed}.json", result)
            seed_results.append(result)
        completed = {
            **started,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "seed_results": seed_results,
            "passed": all(result["passed"] for result in seed_results),
        }
        _write_json_exclusive(run_dir / "RUN_COMPLETE.json", completed)
        print(json.dumps({"run_dir": str(run_dir), "passed": completed["passed"]}))
        return 0
    except BaseException as error:
        _write_json_replace(
            run_dir / "RUN_FAILED.json",
            {"type": type(error).__name__, "message": str(error)},
        )
        raise


if __name__ == "__main__":
    sys.exit(main())
