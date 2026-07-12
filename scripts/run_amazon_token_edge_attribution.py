#!/usr/bin/env python
"""Score one frozen token-HSO checkpoint under registered attention masks."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_amazon_token_edge_attribution import load_config, verify_lock  # noqa: E402
from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_edge_attribution import (  # noqa: E402
    additive_attention_mask,
    pack_candidate_with_segments,
)
from myrec.analysis.token_history_observability import (  # noqa: E402
    TokenHistoryCrossEncoder,
    TokenHistoryData,
)
from prepare_amazon_token_history_observability import load_config as load_upstream_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def assert_device(config: dict[str, Any], seed: int, name: str) -> torch.device:
    physical = int(config["resources"]["seed_to_physical_gpu"][str(seed)])
    if name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("token-edge physical GPU registration differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("token-edge audit requires exactly one visible GPU")
    return torch.device(name)


def edge_scores(
    model: TokenHistoryCrossEncoder,
    data: TokenHistoryData,
    request_index: int,
    candidates: np.ndarray,
    *,
    scenario: str,
    mode: str,
    upstream: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    rows = []
    batch_size = int(config["evaluation"]["encoded_candidates_per_batch"])
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(candidates), batch_size):
            packed = [
                pack_candidate_with_segments(
                    data,
                    request_index,
                    int(candidate),
                    scenario=scenario,
                    token_config=upstream["tokens"],
                )
                for candidate in candidates[start : start + batch_size]
            ]
            ids = torch.from_numpy(np.asarray([row[0] for row in packed], dtype=np.int64)).to(device)
            valid = torch.from_numpy(np.asarray([row[1] for row in packed], dtype=bool)).to(device)
            segments = torch.from_numpy(np.asarray([row[2] for row in packed], dtype=np.int8)).to(device)
            bias = additive_attention_mask(valid, segments, mode)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.backbone(input_ids=ids.long(), attention_mask=bias)
                values = model.score_head(output.last_hidden_state[:, 0]).squeeze(-1)
            rows.append(values.float().cpu().numpy())
    return np.concatenate(rows).astype(np.float32, copy=False)


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.seed not in [int(value) for value in config["seeds"]]:
        raise ValueError("token-edge seed differs")
    _, lock_hash = verify_lock(config, config_path)
    device = assert_device(config, args.seed, args.device)
    seed_all(args.seed)
    paths = config["paths"]
    upstream = load_upstream_config(ROOT / paths["upstream_config"])
    token_root = ROOT / paths["token_root"]
    output_root = ROOT / paths["artifact_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    score_path = output_root / f"seed_{args.seed}_scores.npz"
    report_path = output_root / f"seed_{args.seed}_report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(score_path)
    data = TokenHistoryData(token_root)
    model = TokenHistoryCrossEncoder(
        ROOT / paths["bge_snapshot"],
        score_head_bias=bool(upstream["model"]["score_head_bias"]),
    ).to(device)
    checkpoint_path = ROOT / paths["checkpoint_root"] / f"seed_{args.seed}.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if int(checkpoint["seed"]) != args.seed:
        raise RuntimeError("token-edge checkpoint seed differs")
    model.load_state_dict(checkpoint["model_state"], strict=True)

    flat: dict[str, list[np.ndarray]] = {}
    scenarios_by_mode = {
        mode: (("true",) if mode == "history_isolated" else ("true", "wrong"))
        for mode in config["modes"]
    }
    for mode, scenarios in scenarios_by_mode.items():
        for scenario in scenarios:
            flat[f"{mode}_{scenario}"] = []
    offsets = [0]
    deterministic_error = 0.0
    permutation_error = 0.0
    checked = False
    for index_value in data.reserve_indices:
        index = int(index_value)
        candidates = data.candidates(index)
        for mode, scenarios in scenarios_by_mode.items():
            for scenario in scenarios:
                flat[f"{mode}_{scenario}"].append(
                    edge_scores(
                        model,
                        data,
                        index,
                        candidates,
                        scenario=scenario,
                        mode=mode,
                        upstream=upstream,
                        config=config,
                        device=device,
                    )
                )
        if not checked:
            reference = flat["history_isolated_true"][-1]
            repeated = edge_scores(
                model, data, index, candidates, scenario="true", mode="history_isolated",
                upstream=upstream, config=config, device=device,
            )
            reversed_values = edge_scores(
                model, data, index, candidates[::-1].copy(), scenario="true",
                mode="history_isolated", upstream=upstream, config=config, device=device,
            )[::-1]
            deterministic_error = float(np.max(np.abs(reference - repeated)))
            permutation_error = float(np.max(np.abs(reference - reversed_values)))
            checked = True
        offsets.append(offsets[-1] + len(candidates))
    arrays = {
        key: np.concatenate(value).astype(np.float32, copy=False)
        for key, value in flat.items()
    }
    arrays["request_indices"] = np.asarray(data.reserve_indices, dtype=np.int64)
    arrays["offsets"] = np.asarray(offsets, dtype=np.int64)
    temporary = score_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    temporary.replace(score_path)

    original = np.load(token_root / f"seed_{args.seed}_scores.npz", allow_pickle=False)
    isolation_error = float(np.max(np.abs(arrays["history_isolated_true"] - original["null"])))
    checks = {
        "checkpoint_hash": sha256_file(checkpoint_path)
        == json.loads((token_root / f"seed_{args.seed}_report.json").read_text(encoding="utf-8"))["checkpoint"]["sha256"],
        "candidate_hash": data.candidate_hash(data.reserve_indices)
        == json.loads((token_root / "token_manifest.json").read_text(encoding="utf-8"))["candidate_hash_reserve"],
        "same_request_indices": np.array_equal(original["request_indices"], data.reserve_indices),
        "same_offsets": np.array_equal(original["offsets"], arrays["offsets"]),
        "deterministic": deterministic_error <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": permutation_error <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "history_isolation_matches_null": isolation_error <= float(config["evaluation"]["isolation_null_tolerance"]),
        "no_retraining": True,
        "dev_test_qrels_closed": True,
    }
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "weight_frozen_attention_edge_scoring",
        "seed": args.seed,
        "execution_lock_sha256": lock_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "scores": {"path": str(score_path.relative_to(ROOT)), "sha256": sha256_file(score_path)},
        "numeric": {
            "deterministic_max_abs": deterministic_error,
            "candidate_permutation_max_abs": permutation_error,
            "history_isolation_null_max_abs": isolation_error,
        },
        "checks": checks,
        "passed_mechanics": all(checks.values()),
    }
    atomic_json(report_path, report)
    print(json.dumps({"seed": args.seed, "passed": report["passed_mechanics"], **report["numeric"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
