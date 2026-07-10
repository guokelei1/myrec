#!/usr/bin/env python
"""Correctly recalibrate D2p alpha with internal-train popularity only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.finetuned_query_tower import (
    build_model,
    calibrate_popularity_alpha,
    load_tokens,
)
from myrec.analysis.supervised_diagnostics import PackedRequestData
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


def main() -> int:
    config_path = Path("configs/analysis/finetuned_nonpersonalized_control.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    run_dir = Path("runs/20260710_kuaisearch_d2t_calibrate_s20260708")
    summary_path = run_dir / "train_summary.json"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    checkpoint_path = Path(summary["checkpoint_path"])
    data = PackedRequestData.load(config["packed_data_dir"], "train")
    input_ids, attention_mask = load_tokens(config, "train")
    with (Path(config["packed_data_dir"]) / "manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    cut = int(manifest["internal_calibration"]["cut_request_index"])
    validation_indices = np.arange(cut, len(data), dtype=np.int64)
    device = "cuda:0"
    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    popularity_path = (
        Path(config["packed_data_dir"]) / "item_log_click_internal_train.npy"
    )
    corrected = calibrate_popularity_alpha(
        model,
        data,
        input_ids,
        attention_mask,
        validation_indices,
        np.load(popularity_path, mmap_mode="r"),
        [float(value) for value in config["internal_calibration"]["d2p_alpha_grid"]],
        int(config["training"]["max_requests_per_batch"]),
        int(config["training"]["max_padded_candidates_per_batch"]),
        device,
    )
    invalid = summary["alpha_calibration"]
    summary["alpha_calibration"] = corrected
    summary["invalid_alpha_calibration"] = {
        "result": invalid,
        "reason": (
            "used full-train click counts containing internal-validation clicks; "
            "invalidated before any D2 dev evaluation"
        ),
    }
    summary["alpha_popularity_path"] = str(popularity_path)
    summary["alpha_popularity_sha256"] = sha256_file(popularity_path)
    write_json(summary_path, summary)
    correction = {
        "analysis_id": config["analysis_id"],
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "corrected_alpha_calibration": corrected,
        "invalid_alpha_calibration": invalid,
        "popularity_path": str(popularity_path),
        "popularity_sha256": sha256_file(popularity_path),
        "qrels_read": False,
        "test_read": False,
    }
    write_json(run_dir / "alpha_recalibration.json", correction)
    print(json.dumps(correction, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
