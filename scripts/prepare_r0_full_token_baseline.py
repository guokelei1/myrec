#!/usr/bin/env python
"""Materialize and freeze label-safe inputs for the ordinary R0 full-token baseline."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from transformers import AutoTokenizer
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.analysis.full_token_baseline import (  # noqa: E402
    build_donor_index,
    choose_fresh_wrong_donor,
    clicked_labels,
    item_text,
)
from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402


ARRAY_NAMES = (
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
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("materialize", "freeze"), required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("R0 full-token config must be a mapping")
    for key in ("test", "test_qrels", "c80_fresh_labels"):
        if bool(config["authorization"][key]):
            raise PermissionError(f"unauthorized R0 full-token boundary: {key}")
    return config


def load_records(path: Path, *, labels_allowed: bool) -> list[dict[str, Any]]:
    if "qrels" in path.name.lower() or "test" in path.name.lower():
        raise PermissionError(f"unauthorized R0 full-token input: {path}")
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if not labels_allowed and any(
                "clicked" in candidate or "purchased" in candidate
                for candidate in row["candidates"]
            ):
                raise PermissionError("blind R0 full-token input contains candidate labels")
            records.append(row)
    return records


def save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    with path.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def tokenize_texts(
    tokenizer: Any, texts: list[str], *, max_length: int, batch_size: int
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.empty((len(texts), max_length), dtype=np.int32)
    masks = np.empty((len(texts), max_length), dtype=bool)
    for start in range(0, len(texts), batch_size):
        stop = min(start + batch_size, len(texts))
        encoded = tokenizer(
            texts[start:stop],
            add_special_tokens=False,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="np",
        )
        ids[start:stop] = encoded["input_ids"].astype(np.int32)
        masks[start:stop] = encoded["attention_mask"].astype(bool)
    return ids, masks


def candidate_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            [str(row["request_id"]), *[str(value["item_id"]) for value in row["candidates"]]],
            separators=(",", ":"),
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def materialize(config: dict[str, Any]) -> None:
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    manifest_path = root / "token_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    train = load_records(ROOT / paths["records_train"], labels_allowed=True)
    dev = load_records(ROOT / paths["records_dev"], labels_allowed=False)
    fit_pairs = [
        (index, row)
        for index, row in enumerate(train)
        if any(value > 0 for value in clicked_labels(row))
    ]

    pool = [*train, *dev]
    selection = config["selection"]
    boundaries = [int(value) for value in selection["history_length_bins"]]
    donor_index = build_donor_index(pool, boundaries)
    wrong_by_dev: dict[int, tuple[int, int, int]] = {}
    ratios = []
    history_present = 0
    for dev_index, row in enumerate(dev):
        if not row["history"]:
            continue
        history_present += 1
        pool_index = len(train) + dev_index
        donor, slice_start, slice_stop, ratio = choose_fresh_wrong_donor(
            pool,
            donor_index,
            pool_index,
            boundaries,
            seed=int(selection["wrong_seed"]),
            freshness_ratio_max=float(selection["freshness_ratio_max"]),
            search_back=int(selection["donor_search_back"]),
        )
        if donor is not None:
            wrong_by_dev[dev_index] = (donor, int(slice_start), int(slice_stop))
            ratios.append(float(ratio))
    coverage = len(wrong_by_dev) / history_present if history_present else 1.0
    if coverage < float(config["evaluation"]["require_wrong_donor_coverage"]):
        raise RuntimeError(f"wrong donor coverage too low: {coverage:.6f}")

    selected: list[tuple[str, int, dict[str, Any]]] = [
        ("train", original, row) for original, row in fit_pairs
    ] + [("dev", index, row) for index, row in enumerate(dev)]
    text_by_id: dict[str, str] = {}
    conflicts = 0
    for split, original, row in selected:
        items = [*row["candidates"], *row["history"]]
        if split == "dev" and original in wrong_by_dev:
            donor, slice_start, slice_stop = wrong_by_dev[original]
            items.extend(pool[donor]["history"][slice_start:slice_stop])
        for item in items:
            identifier = str(item["item_id"])
            text = item_text(item)
            previous = text_by_id.get(identifier)
            if previous is not None and previous != text:
                conflicts += 1
                text = max((previous, text), key=lambda value: (len(value), value))
            text_by_id[identifier] = text
    item_ids = sorted(text_by_id)
    item_position = {identifier: position for position, identifier in enumerate(item_ids)}

    candidate_offsets = [0]
    candidate_positions = []
    history_offsets = [0]
    history_positions = []
    wrong_offsets = [0]
    wrong_positions = []
    query_texts = []
    request_rows = []
    label_request_indices = []
    label_offsets = [0]
    labels = []
    original_indices = []
    roles = []
    for position, (split, original, row) in enumerate(selected):
        candidates = [item_position[str(value["item_id"])] for value in row["candidates"]]
        history = [item_position[str(value["item_id"])] for value in row["history"]]
        wrong_history = (
            [
                item_position[str(value["item_id"])]
                for value in pool[wrong_by_dev[original][0]]["history"][
                    wrong_by_dev[original][1] : wrong_by_dev[original][2]
                ]
            ]
            if split == "dev" and original in wrong_by_dev
            else []
        )
        candidate_positions.extend(candidates)
        history_positions.extend(history)
        wrong_positions.extend(wrong_history)
        candidate_offsets.append(len(candidate_positions))
        history_offsets.append(len(history_positions))
        wrong_offsets.append(len(wrong_positions))
        query_texts.append(str(row["query"]))
        roles.append(0 if split == "train" else 1)
        original_indices.append(original if split == "train" else -(original + 1))
        request_rows.append(
            {
                "position": position,
                "source_split": split,
                "source_index": original,
                "request_id": str(row["request_id"]),
                "user_id": str(row["user_id"]),
            }
        )
        if split == "train":
            row_labels = clicked_labels(row)
            label_request_indices.append(position)
            labels.extend(row_labels)
            label_offsets.append(len(labels))

    requests_path = root / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for row in request_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    items_path = root / "items.jsonl"
    with items_path.open("w", encoding="utf-8") as handle:
        for position, identifier in enumerate(item_ids):
            handle.write(
                json.dumps(
                    {"position": position, "item_id": identifier, "text": text_by_id[identifier]},
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )

    arrays = {
        "request_original_indices.npy": np.asarray(original_indices, dtype=np.int64),
        "request_roles.npy": np.asarray(roles, dtype=np.int8),
        "candidate_offsets.npy": np.asarray(candidate_offsets, dtype=np.int64),
        "candidate_item_positions.npy": np.asarray(candidate_positions, dtype=np.int32),
        "history_offsets.npy": np.asarray(history_offsets, dtype=np.int64),
        "history_item_positions.npy": np.asarray(history_positions, dtype=np.int32),
        "wrong_history_offsets.npy": np.asarray(wrong_offsets, dtype=np.int64),
        "wrong_history_item_positions.npy": np.asarray(wrong_positions, dtype=np.int32),
    }
    files = {name: save_array(root, name, value) for name, value in arrays.items()}
    labels_path = root / "fit_labels.npz"
    with labels_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=np.asarray(label_request_indices, dtype=np.int64),
            offsets=np.asarray(label_offsets, dtype=np.int64),
            labels=np.asarray(labels, dtype=np.float32),
        )
    files["fit_labels.npz"] = {
        "path": str(labels_path.relative_to(ROOT)),
        "sha256": sha256_file(labels_path),
    }

    tokenizer = AutoTokenizer.from_pretrained(ROOT / paths["bge_snapshot"], local_files_only=True)
    tokens = config["tokens"]
    query_ids, query_masks = tokenize_texts(
        tokenizer,
        query_texts,
        max_length=int(tokens["query_tokens"]),
        batch_size=int(tokens["tokenizer_batch_size"]),
    )
    item_limit = max(
        int(tokens["candidate_tokens"]),
        max(
            int(config["tokens"]["history_item_tokens"]),
            *[
                int(trial.get("tokens", {}).get("history_item_tokens", 0))
                for trial in config["trials"].values()
            ],
        ),
    )
    item_ids_array, item_masks = tokenize_texts(
        tokenizer,
        [text_by_id[identifier] for identifier in item_ids],
        max_length=item_limit,
        batch_size=int(tokens["tokenizer_batch_size"]),
    )
    for name, value in (
        ("query_token_ids.npy", query_ids),
        ("query_attention_mask.npy", query_masks),
        ("item_token_ids.npy", item_ids_array),
        ("item_attention_mask.npy", item_masks),
    ):
        files[name] = save_array(root, name, value)
    files["requests.jsonl"] = {
        "path": str(requests_path.relative_to(ROOT)),
        "sha256": sha256_file(requests_path),
    }
    files["items.jsonl"] = {
        "path": str(items_path.relative_to(ROOT)),
        "sha256": sha256_file(items_path),
    }
    ratios_array = np.asarray(ratios, dtype=np.float64)
    manifest = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "kuaisearch_train_and_blind_dev_full_token_materialization",
        "fit_requests": len(fit_pairs),
        "dev_requests": len(dev),
        "dev_users": len({str(row["user_id"]) for row in dev}),
        "items": len(item_ids),
        "candidate_rows": len(candidate_positions),
        "history_rows": len(history_positions),
        "wrong_history_rows": len(wrong_positions),
        "wrong_donor": {
            "history_present_requests": history_present,
            "matched_requests": len(wrong_by_dev),
            "coverage": coverage,
            "same_user": 0,
            "freshness_ratio_max": float(ratios_array.max()) if len(ratios_array) else None,
            "freshness_ratio_median": float(np.median(ratios_array)) if len(ratios_array) else None,
        },
        "candidate_hash_dev": candidate_hash(dev),
        "candidate_manifest_sha256": sha256_file(ROOT / paths["candidate_manifest"]),
        "item_text_conflicts_resolved": conflicts,
        "special_tokens": {
            "cls_token_id": int(tokenizer.cls_token_id),
            "sep_token_id": int(tokenizer.sep_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
        },
        "label_boundary": {
            "train_labels_compacted": True,
            "dev_records_label_free": True,
            "dev_qrels_read": False,
            "test_read": False,
            "c80_fresh_labels_read": False,
        },
        "files": files,
    }
    atomic_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "fit_requests": len(fit_pairs),
                "dev_requests": len(dev),
                "items": len(item_ids),
                "wrong_coverage": coverage,
            },
            sort_keys=True,
        )
    )


def freeze(config: dict[str, Any], config_path: Path) -> None:
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    lock_path = ROOT / paths["execution_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    manifest_path = root / "token_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    input_sha = {
        "config": sha256_file(config_path),
        "module": sha256_file(ROOT / "src/myrec/analysis/full_token_baseline.py"),
        "token_module": sha256_file(ROOT / "src/myrec/analysis/token_history_observability.py"),
        "prepare_script": sha256_file(Path(__file__)),
        "run_script": sha256_file(ROOT / "scripts/run_r0_full_token_baseline.py"),
        "budget": sha256_file(
            ROOT / "experiments/problem_discovery/r0_full_token_trial_budget.yaml"
        ),
        "records_train": sha256_file(ROOT / paths["records_train"]),
        "records_dev": sha256_file(ROOT / paths["records_dev"]),
        "candidate_manifest": sha256_file(ROOT / paths["candidate_manifest"]),
        "token_manifest": sha256_file(manifest_path),
    }
    for name, row in manifest["files"].items():
        path = ROOT / row["path"]
        actual = sha256_file(path)
        if actual != row["sha256"]:
            raise RuntimeError(f"materialized artifact hash differs: {name}")
        input_sha[f"artifact_{name}"] = actual
    lock = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_registered_ordinary_full_token_trials",
        "input_sha256": input_sha,
        "registered_trials": config["trials"],
        "outcome_boundary": {
            "train_labels": True,
            "blind_dev_scoring": True,
            "dev_qrels_shared_evaluator_only": True,
            "test": False,
            "c80_fresh_labels": False,
        },
    }
    atomic_json(lock_path, lock)
    print(json.dumps({"lock": str(lock_path.relative_to(ROOT)), "sha256": sha256_file(lock_path)}))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.stage == "materialize":
        materialize(config)
    else:
        freeze(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
