#!/usr/bin/env python
"""Build fixed D2s true- and wrong-history scores without retraining."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.baselines.core import write_static_mixture_scores
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


def main() -> int:
    base_config_path = Path("configs/analysis/d2s_static_full_control.yaml")
    final_config_path = Path("configs/analysis/d2s_static_full_control_final.yaml")
    with base_config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with final_config_path.open("r", encoding="utf-8") as handle:
        final_config = yaml.safe_load(handle)
    beta = float(final_config["final_scoring"]["beta"])
    candidate_manifest = config["dev_sources"]["candidate_manifest"]

    results = {}
    for seed_value in config["seeds"]:
        seed = int(seed_value)
        d2p_run_id = config["dev_sources"]["d2p_run_pattern"].format(seed=seed)
        d2p_scores = Path("runs") / d2p_run_id / "scores.jsonl"
        history_conditions = {
            "true": config["dev_sources"]["true_b0b_run"],
            "wrong": config["dev_sources"]["wrong_b0b_run_pattern"].format(
                seed=seed
            ),
        }
        for condition, history_run_id in history_conditions.items():
            run_id = f"20260710_kuaisearch_d2s_static_{condition}_history_dev_s{seed}"
            metadata = write_static_mixture_scores(
                query_scores_path=d2p_scores,
                history_scores_path=Path("runs") / history_run_id / "scores.jsonl",
                query_run_id=d2p_run_id,
                history_run_id=history_run_id,
                run_id=run_id,
                method_id=f"d2s_static_{condition}_history",
                alpha=beta,
                candidate_manifest_path=candidate_manifest,
                config_path=final_config_path,
            )
            metadata.update(
                {
                    "analysis_id": config["analysis_id"],
                    "base_config_path": str(base_config_path),
                    "base_config_sha256": sha256_file(base_config_path),
                    "beta": beta,
                    "final_config_path": str(final_config_path),
                    "final_config_sha256": sha256_file(final_config_path),
                    "history_condition": condition,
                    "qrels_read": False,
                    "seed": seed,
                    "test_read": False,
                }
            )
            write_json(Path("runs") / run_id / "metadata.json", metadata)
            results[run_id] = metadata
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
