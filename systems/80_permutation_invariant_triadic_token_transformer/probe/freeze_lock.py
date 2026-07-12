#!/usr/bin/env python
"""Freeze and verify the terminal C80 real-gate execution boundary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from prepare_real_gate import load_config  # noqa: E402


FRESH_FILES = (
    "request_original_indices.npy",
    "request_roles.npy",
    "candidate_offsets.npy",
    "candidate_item_positions.npy",
    "history_offsets.npy",
    "history_item_positions.npy",
    "wrong_history_offsets.npy",
    "wrong_history_item_positions.npy",
    "query_token_ids.npy",
    "query_attention_mask.npy",
    "item_token_ids.npy",
    "item_attention_mask.npy",
    "requests.jsonl",
    "items.jsonl",
    "token_manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def source_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config["paths"]
    keys = (
        "protocol",
        "proposal",
        "module",
        "prepare_script",
        "freeze_script",
        "run_script",
        "summarize_script",
    )
    output = {"config": config_path}
    output.update({key: ROOT / paths[key] for key in keys})
    output["upstream_token_module"] = ROOT / "src/myrec/analysis/token_history_observability.py"
    output["upstream_prepare_helpers"] = ROOT / "scripts/prepare_amazon_token_history_observability.py"
    candidate_root = ROOT / "systems/80_permutation_invariant_triadic_token_transformer"
    output.update(
        {
            "readme": candidate_root / "README.md",
            "mechanism_fingerprint": candidate_root / "notes/mechanism_fingerprint.md",
            "nearest_neighbors": candidate_root / "notes/nearest_neighbors.md",
            "preimplementation_review": candidate_root / "notes/preimplementation_review.md",
            "model_tests": candidate_root / "tests/test_model.py",
        }
    )
    return output


def input_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    fresh = ROOT / paths["fresh_root"]
    fit = ROOT / paths["fit_token_root"]
    snapshot = ROOT / paths["bge_snapshot"]
    output = {
        "records_train_blind": ROOT / paths["records_train_blind"],
        "records_train_labels": ROOT / paths["records_train"],
        "c38_selection": ROOT / paths["c38_selection"],
        "fit_manifest": fit / "token_manifest.json",
        "fit_requests": fit / "requests.jsonl",
        "fit_labels": ROOT / paths["fit_labels"],
        "external_hso_report": ROOT / "reports/pps_amazon_token_history_observability_v1.json",
        "backbone_config": snapshot / "config.json",
        "backbone_weights": snapshot / "model.safetensors",
        "tokenizer": snapshot / "tokenizer.json",
    }
    output.update({f"fresh_{name}": fresh / name for name in FRESH_FILES})
    for base_seed in sorted({int(value) for value in config["training"]["base_seed_map"].values()}):
        output[f"base_checkpoint_{base_seed}"] = (
            ROOT / paths["base_checkpoint_root"] / f"seed_{base_seed}.pt"
        )
        output[f"base_report_{base_seed}"] = fit / f"seed_{base_seed}_report.json"
    return output


def current_hashes(paths: dict[str, Path]) -> dict[str, str]:
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("C80 lock inputs missing: " + ", ".join(missing))
    return {key: sha256_file(path) for key, path in paths.items()}


def freeze(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    lock_path = ROOT / config["paths"]["execution_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    lock = {
        "candidate_id": "c80",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_three_seed_five_mode_terminal_real_gate",
        "source_sha256": current_hashes(source_paths(config, config_path)),
        "input_sha256": current_hashes(input_paths(config)),
        "outcome_boundary": {
            "fit_labels_reused_after_lock": True,
            "fresh_labels_before_all_scores": False,
            "dev_test_qrels": False,
            "successor_after_c80": False,
            "mandatory_c01_c80_retrospective": True,
        },
    }
    atomic_json(lock_path, lock)
    return lock, sha256_file(lock_path)


def verify_lock(
    config: dict[str, Any], config_path: Path
) -> tuple[dict[str, Any], str]:
    lock_path = ROOT / config["paths"]["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if current_hashes(source_paths(config, config_path)) != lock["source_sha256"]:
        raise RuntimeError("C80 source changed after execution lock")
    if current_hashes(input_paths(config)) != lock["input_sha256"]:
        raise RuntimeError("C80 input changed after execution lock")
    return lock, sha256_file(lock_path)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    lock, digest = (
        verify_lock(config, config_path)
        if args.verify
        else freeze(config, config_path)
    )
    print(json.dumps({"decision": lock["decision"], "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
