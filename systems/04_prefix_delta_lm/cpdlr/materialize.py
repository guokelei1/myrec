"""Create the train/internal C04 probe slice and its train-only D2p anchor."""

from __future__ import annotations

import time
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .io import (
    assert_candidate_manifest,
    assert_train_only_path,
    iter_jsonl,
    sha256_file,
    stable_hash,
    write_json,
    write_jsonl,
)


def _zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - values.mean()) / np.sqrt(float(values.var()) + 1e-6)


def _select_indices(request_ids: list[str], start: int, stop: int, count: int, seed: int) -> list[int]:
    candidates = list(range(start, stop))
    candidates.sort(key=lambda index: stable_hash("c04_probe", seed, request_ids[index]))
    return sorted(candidates[:count])


def _teacher_scores(
    config: dict[str, Any], selected_indices: list[int], device: str
) -> dict[str, dict[str, float]]:
    # Shared source is imported read-only.  It implements the already frozen D2p
    # train-only anchor; no candidate-local evaluator or qrels path is involved.
    from myrec.analysis.finetuned_query_tower import (
        build_model,
        iter_query_batches,
        load_tokens,
    )
    from myrec.analysis.supervised_diagnostics import PackedRequestData

    anchor_cfg = config["anchor"]
    import yaml

    with Path(anchor_cfg["d2_config"]).open("r", encoding="utf-8") as handle:
        d2_config = yaml.safe_load(handle)
    data = PackedRequestData.load(d2_config["packed_data_dir"], "train")
    input_ids, attention_mask = load_tokens(d2_config, "train")
    model = build_model(d2_config, device)
    checkpoint = torch.load(
        anchor_cfg["d2_checkpoint"], map_location="cpu", weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    popularity = np.load(
        Path(d2_config["packed_data_dir"]) / "item_log_click_full_train.npy",
        mmap_mode="r",
    )
    alpha = float(anchor_cfg["d2p_alpha"])
    result: dict[str, dict[str, float]] = {}
    with torch.inference_mode():
        for batch in iter_query_batches(
            data,
            np.asarray(selected_indices, dtype=np.int64),
            int(anchor_cfg["max_requests_per_batch"]),
            int(anchor_cfg["max_padded_candidates_per_batch"]),
            0,
            False,
        ):
            indices = batch["request_indices"]
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=str(device).startswith("cuda"),
            ):
                raw = model(
                    torch.from_numpy(
                        np.asarray(input_ids[indices], dtype=np.int64)
                    ).to(device),
                    torch.from_numpy(
                        np.asarray(attention_mask[indices], dtype=np.int64)
                    ).to(device),
                    torch.from_numpy(batch["candidate_indices"]).to(device),
                    torch.from_numpy(batch["candidate_mask"]).to(device),
                )
            raw = raw.float().cpu().numpy()
            for row_index, request_index in enumerate(indices):
                request_index = int(request_index)
                count = int(batch["candidate_mask"][row_index].sum())
                item_indices = batch["candidate_indices"][row_index, :count]
                mixed = alpha * _zscore(raw[row_index, :count]) + (1.0 - alpha) * _zscore(
                    np.asarray(popularity[item_indices])
                )
                result[data.request_ids[request_index]] = {
                    str(item_id): float(score)
                    for item_id, score in zip(
                        batch["candidate_item_ids"][row_index, :count], mixed
                    )
                }
    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return result


def _sample_candidates(
    record: dict[str, Any], teacher: dict[str, float], limit: int, seed: int
) -> list[dict[str, Any]]:
    history_ids = {str(event.get("item_id")) for event in record.get("history", [])}
    candidates = list(record["candidates"])
    positives = [
        index
        for index, candidate in enumerate(candidates)
        if int(candidate.get("clicked", 0)) > 0
    ]
    if not positives:
        raise ValueError(f"materialized train request has no clicked positive: {record['request_id']}")
    repeats = [
        index
        for index, candidate in enumerate(candidates)
        if str(candidate["item_id"]) in history_ids
    ]
    required = []
    for index in positives + repeats:
        if index not in required:
            required.append(index)
    if len(required) > limit:
        required = required[:limit]
        if not any(index in positives for index in required):
            required[-1] = positives[0]
    remaining = [index for index in range(len(candidates)) if index not in required]
    remaining.sort(
        key=lambda index: stable_hash(
            "c04_negative", seed, record["request_id"], candidates[index]["item_id"]
        )
    )
    selected = required + remaining[: max(limit - len(required), 0)]
    selected.sort()
    rows = []
    for index in selected:
        candidate = candidates[index]
        item_id = str(candidate["item_id"])
        if item_id not in teacher:
            raise ValueError(
                f"D2p anchor missing candidate: request={record['request_id']} item={item_id}"
            )
        rows.append(
            {
                "anchor_score": float(teacher[item_id]),
                "brand": str(candidate.get("brand", "")),
                "cat": list(candidate.get("cat", [])),
                "exact_repeat": item_id in history_ids,
                "item_id": item_id,
                "label": int(candidate.get("clicked", 0)),
                "seller": str(candidate.get("seller", "")),
                "title": str(candidate.get("title", "")),
            }
        )
    return rows


def _assign_wrong_histories(rows: list[dict[str, Any]], seed: int) -> None:
    donors = sorted(
        (row for row in rows if row.get("history")),
        key=lambda row: (int(row["ts"]), str(row["request_id"])),
    )
    if len(donors) < 2:
        raise ValueError("not enough history-present requests for wrong-user controls")
    donor_times = [int(row["ts"]) for row in donors]
    for target in rows:
        eligible_count = bisect_right(donor_times, int(target["ts"]))
        if not eligible_count:
            target["wrong_history"] = []
            continue
        target_key = stable_hash("c04_donor", seed, target["request_id"])
        start = int(target_key[:12], 16) % eligible_count
        donor = None
        for offset in range(eligible_count):
            candidate = donors[(start + offset) % eligible_count]
            if str(candidate["user_id"]) != str(target["user_id"]):
                donor = candidate
                break
        target["wrong_history"] = list(donor["history"]) if donor is not None else []


def materialize_probe(config: dict[str, Any], config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    paths = config["paths"]
    assert_train_only_path(paths["records_train"])
    candidate_hash = assert_candidate_manifest(
        paths["candidate_manifest"], config["candidate_manifest_sha256"]
    )
    from myrec.analysis.supervised_diagnostics import PackedRequestData

    packed = PackedRequestData.load(paths["packed_data_dir"], "train")
    packed_manifest_path = Path(paths["packed_data_dir"]) / "manifest.json"
    import json

    with packed_manifest_path.open("r", encoding="utf-8") as handle:
        packed_manifest = json.load(handle)
    cut = int(packed_manifest["internal_calibration"]["cut_request_index"])
    seed = int(config["seed"])
    material = config["materialization"]
    train_indices = _select_indices(
        packed.request_ids, 0, cut, int(material["train_requests"]), seed
    )
    internal_indices = _select_indices(
        packed.request_ids,
        cut,
        len(packed),
        int(material["internal_requests"]),
        seed,
    )
    selected_indices = train_indices + internal_indices
    teacher = _teacher_scores(config, selected_indices, device)
    split_by_request = {
        packed.request_ids[index]: "train" for index in train_indices
    }
    split_by_request.update(
        {packed.request_ids[index]: "internal" for index in internal_indices}
    )
    selected: list[dict[str, Any]] = []
    for record in iter_jsonl(paths["records_train"]):
        request_id = str(record["request_id"])
        split = split_by_request.get(request_id)
        if split is None:
            continue
        row = {
            "candidates": _sample_candidates(
                record,
                teacher[request_id],
                int(material["candidates_per_request"]),
                seed,
            ),
            "history": list(record.get("history", [])),
            "query": str(record.get("query", "")),
            "request_id": request_id,
            "split": split,
            "ts": int(record.get("ts", 0)),
            "user_id": str(record.get("user_id", "")),
        }
        selected.append(row)
    if len(selected) != len(selected_indices):
        raise ValueError(
            f"selected train record coverage mismatch: {len(selected)} != {len(selected_indices)}"
        )
    for split in ("train", "internal"):
        _assign_wrong_histories(
            [row for row in selected if row["split"] == split], seed
        )
    output_dir = Path(paths["probe_data_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    counts = {}
    for split in ("train", "internal"):
        output_path = output_dir / f"{split}_examples.jsonl"
        count = write_jsonl(
            output_path,
            (row for row in selected if row["split"] == split),
        )
        output_paths[split] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
        }
        counts[split] = count
    report = {
        "candidate_id": config["candidate_id"],
        "candidate_manifest_sha256": candidate_hash,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "elapsed_seconds": time.time() - started,
        "output_files": output_paths,
        "qrels_read": False,
        "request_counts": counts,
        "seed": seed,
        "selection": {
            "candidate_limit": int(material["candidates_per_request"]),
            "internal_source_range": [cut, len(packed)],
            "rule": "lowest deterministic SHA256 keys within frozen train/internal ranges",
            "train_source_range": [0, cut],
        },
        "source_records_sha256": sha256_file(paths["records_train"]),
        "test_read": False,
        "teacher": {
            "alpha": float(config["anchor"]["d2p_alpha"]),
            "checkpoint": config["anchor"]["d2_checkpoint"],
            "checkpoint_sha256": sha256_file(config["anchor"]["d2_checkpoint"]),
            "definition": "frozen train-trained D2t plus full-train popularity, within-request z scores",
        },
    }
    write_json(output_dir / "manifest.json", report)
    return report
