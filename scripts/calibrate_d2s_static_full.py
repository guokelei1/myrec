#!/usr/bin/env python
"""Select the D2s D2p/history beta on internal train only."""

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
    calibrate_d2p_history_beta,
    load_tokens,
)
from myrec.analysis.supervised_diagnostics import PackedRequestData
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


def main() -> int:
    config_path = Path("configs/analysis/d2s_static_full_control.yaml")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    d2_config_path = Path(config["d2_config"])
    with d2_config_path.open("r", encoding="utf-8") as handle:
        d2_config = yaml.safe_load(handle)

    packed_dir = Path(config["packed_data_dir"])
    data = PackedRequestData.load(packed_dir, "train")
    input_ids, attention_mask = load_tokens(d2_config, "train")
    with (packed_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    cut = int(manifest["internal_calibration"]["cut_request_index"])

    checkpoint_path = Path(config["d2_calibration_checkpoint"])
    device = "cuda:0"
    model = build_model(d2_config, device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    item_log_click = np.load(
        packed_dir / config["d2p_calibration_popularity"], mmap_mode="r"
    )
    result = calibrate_d2p_history_beta(
        model=model,
        data=data,
        input_ids=input_ids,
        attention_mask=attention_mask,
        request_indices=np.arange(cut, len(data), dtype=np.int64),
        item_log_click=item_log_click,
        d2p_alpha=float(config["d2p_alpha"]),
        beta_grid=[float(value) for value in config["beta_grid"]],
        max_requests=int(d2_config["training"]["max_requests_per_batch"]),
        max_candidates=int(
            d2_config["training"]["max_padded_candidates_per_batch"]
        ),
        device=device,
    )
    report = {
        "analysis_id": config["analysis_id"],
        "calibration": result,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "feature_scope": (
            "frozen D2p plus causal B0b on final 10 percent retained train; "
            "popularity counts use first 90 percent internal train only"
        ),
        "qrels_read": False,
        "test_read": False,
    }
    write_json("reports/pps_d2s_train_only_calibration.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
