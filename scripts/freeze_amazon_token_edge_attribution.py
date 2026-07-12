#!/usr/bin/env python
"""Freeze and verify the post-outcome Amazon token-edge attribution audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402


SOURCE_KEYS = ("protocol", "module", "freeze_script", "run_script", "summarize_script")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("token-edge config must be a mapping")
    if any(bool(value["authorization"][key]) for key in ("retraining", "checkpoint_selection", "dev", "test", "qrels")):
        raise PermissionError("token-edge audit requests unauthorized action")
    return value


def source_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    output = {"config": config_path}
    output.update({key: ROOT / config["paths"][key] for key in SOURCE_KEYS})
    return output


def input_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    root = ROOT / paths["token_root"]
    checkpoint_root = ROOT / paths["checkpoint_root"]
    output = {
        "upstream_config": ROOT / paths["upstream_config"],
        "upstream_execution_lock": ROOT / paths["upstream_execution_lock"],
        "upstream_report": ROOT / paths["upstream_report"],
        "token_manifest": root / "token_manifest.json",
        "request_indices": root / "request_original_indices.npy",
        "request_roles": root / "request_roles.npy",
        "candidate_offsets": root / "candidate_offsets.npy",
        "candidate_positions": root / "candidate_item_positions.npy",
        "history_offsets": root / "history_offsets.npy",
        "history_positions": root / "history_item_positions.npy",
        "wrong_offsets": root / "wrong_history_offsets.npy",
        "wrong_positions": root / "wrong_history_item_positions.npy",
        "query_ids": root / "query_token_ids.npy",
        "query_mask": root / "query_attention_mask.npy",
        "item_ids": root / "item_token_ids.npy",
        "item_mask": root / "item_attention_mask.npy",
        "requests": root / "requests.jsonl",
        "items": root / "items.jsonl",
    }
    for seed in config["seeds"]:
        output[f"checkpoint_{seed}"] = checkpoint_root / f"seed_{seed}.pt"
        output[f"scores_{seed}"] = root / f"seed_{seed}_scores.npz"
        output[f"seed_report_{seed}"] = root / f"seed_{seed}_report.json"
    return output


def verify_lock(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    lock_path = ROOT / config["paths"]["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    source = {key: sha256_file(path) for key, path in source_paths(config, config_path).items()}
    inputs = {key: sha256_file(path) for key, path in input_paths(config).items()}
    if source != lock["source_sha256"]:
        raise RuntimeError("token-edge source changed after lock")
    if inputs != lock["input_sha256"]:
        raise RuntimeError("token-edge input changed after lock")
    return lock, sha256_file(lock_path)


def freeze(config: dict[str, Any], config_path: Path) -> None:
    lock_path = ROOT / config["paths"]["execution_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_weight_frozen_post_outcome_attention_edge_attribution",
        "modes": list(config["modes"]),
        "source_sha256": {
            key: sha256_file(path) for key, path in source_paths(config, config_path).items()
        },
        "input_sha256": {
            key: sha256_file(path) for key, path in input_paths(config).items()
        },
        "boundary": {
            "retraining": False,
            "checkpoint_selection": False,
            "same_open_reserve": True,
            "dev_test_qrels": False,
        },
    }
    atomic_json(lock_path, lock)
    print(json.dumps({"path": str(lock_path), "sha256": sha256_file(lock_path)}, sort_keys=True))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.verify:
        _, digest = verify_lock(config, config_path)
        print(json.dumps({"verified": True, "sha256": digest}, sort_keys=True))
    else:
        freeze(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
