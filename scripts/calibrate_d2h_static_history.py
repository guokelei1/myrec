#!/usr/bin/env python
"""Select the D2h static text/history alpha on internal train only."""

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
    calibrate_history_alpha,
    load_tokens,
)
from myrec.analysis.supervised_diagnostics import PackedRequestData
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


def main() -> int:
    config_path = Path("configs/analysis/d2h_static_history_control.yaml")
    d2_config_path = Path("configs/analysis/finetuned_nonpersonalized_control.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with d2_config_path.open("r", encoding="utf-8") as handle:
        d2_config = yaml.safe_load(handle)
    data = PackedRequestData.load(config["packed_data_dir"], "train")
    input_ids, attention_mask = load_tokens(d2_config, "train")
    with (Path(config["packed_data_dir"]) / "manifest.json").open(
        "r", encoding="utf-8"
    ) as handle:
        manifest = json.load(handle)
    cut = int(manifest["internal_calibration"]["cut_request_index"])
    checkpoint_path = Path(config["d2_calibration_checkpoint"])
    device = "cuda:0"
    model = build_model(d2_config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    result = calibrate_history_alpha(
        model,
        data,
        input_ids,
        attention_mask,
        np.arange(cut, len(data), dtype=np.int64),
        [float(value) for value in config["alpha_grid"]],
        int(d2_config["training"]["max_requests_per_batch"]),
        int(d2_config["training"]["max_padded_candidates_per_batch"]),
        device,
    )
    report = {
        "analysis_id": config["analysis_id"],
        "calibration": result,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "feature_scope": "D2t plus causal B0b on final 10 percent retained train",
        "qrels_read": False,
        "test_read": False,
    }
    write_json("reports/pps_d2h_train_only_calibration.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
