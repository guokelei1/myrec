#!/usr/bin/env python
"""Materialize, freeze, and stage the Amazon full-token HSO probe."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402


SOURCE_KEYS = ("protocol", "module", "prepare_script", "run_script", "summarize_script")
TOKEN_ARTIFACTS = (
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
    parser.add_argument("--stage", choices=("prepare", "freeze", "stage-labels"), required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("token HSO config must be a mapping")
    for key in ("dev", "test", "qrels"):
        if bool(value["authorization"][key]):
            raise PermissionError(f"token HSO unauthorized split: {key}")
    return value


def load_blind(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if any("clicked" in value or "purchased" in value for value in row["candidates"]):
                raise PermissionError("token HSO blind input contains candidate labels")
            records.append(row)
    return records


def has_repeat(record: dict[str, Any]) -> bool:
    candidates = {str(value["item_id"]) for value in record["candidates"]}
    history = {str(value["item_id"]) for value in record["history"]}
    return bool(candidates & history)


def length_bin(length: int, boundaries: list[int]) -> int:
    for boundary in boundaries:
        if length <= boundary:
            return int(boundary)
    return int(boundaries[-1])


def item_text(item: dict[str, Any]) -> str:
    categories = item.get("cat")
    category = (
        " > ".join(str(value) for value in categories if value)
        if isinstance(categories, list)
        else ""
    )
    return " ".join(
        value.strip()
        for value in (
            str(item.get("title") or ""),
            str(item.get("brand") or ""),
            category,
        )
        if value.strip()
    )


def choose_wrong_donor(
    records: list[dict[str, Any]],
    bins: dict[int, list[int]],
    target_index: int,
    boundaries: list[int],
    seed: int,
) -> int:
    target = records[target_index]
    target_bin = length_bin(len(target["history"]), boundaries)
    candidates = [
        index
        for index in bins[target_bin]
        if str(records[index]["user_id"]) != str(target["user_id"])
    ]
    if not candidates:
        raise ValueError("token HSO has no wrong donor in history bin")
    return min(
        candidates,
        key=lambda index: hashlib.sha256(
            f"{seed}:token-wrong:{target['request_id']}:{records[index]['request_id']}".encode()
        ).digest(),
    )


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


def prepare(config: dict[str, Any]) -> None:
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    manifest_path = root / "token_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    records = load_blind(ROOT / paths["records_train_blind"])
    selection = json.loads((ROOT / paths["c38_selection"]).read_text(encoding="utf-8"))
    fit_original = [
        int(index)
        for index in selection["roles"][config["selection"]["fit_role"]]["indices"]
        if not has_repeat(records[int(index)])
    ]
    fit_users = {str(records[index]["user_id"]) for index in fit_original}
    reserve_candidates = [
        int(index)
        for index in selection["unused_indices"]
        if not has_repeat(records[int(index)])
        and str(records[int(index)]["user_id"]) not in fit_users
    ]
    reserve_original = reserve_candidates[: int(config["selection"]["reserve_requests"])]
    if len(reserve_original) != int(config["selection"]["reserve_requests"]):
        raise ValueError("token HSO reserve is too small after exclusions")
    boundaries = [int(value) for value in config["selection"]["history_length_bins"]]
    bins: defaultdict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        bins[length_bin(len(record["history"]), boundaries)].append(index)
    wrong_seed = int(config["selection"]["wrong_seed"])
    wrong = {
        index: choose_wrong_donor(records, bins, index, boundaries, wrong_seed)
        for index in reserve_original
    }
    selected_original = [*fit_original, *reserve_original]
    roles = np.asarray([0] * len(fit_original) + [1] * len(reserve_original), dtype=np.int8)

    text_by_id: dict[str, str] = {}
    conflicts = 0
    for original in selected_original:
        record = records[original]
        items = [*record["candidates"], *record["history"]]
        if original in wrong:
            items.extend(records[wrong[original]]["history"])
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
    request_rows = []
    query_texts = []
    for position, original in enumerate(selected_original):
        record = records[original]
        candidates = [item_position[str(value["item_id"])] for value in record["candidates"]]
        history = [item_position[str(value["item_id"])] for value in record["history"]]
        wrong_history = (
            [
                item_position[str(value["item_id"])]
                for value in records[wrong[original]]["history"]
            ]
            if original in wrong
            else []
        )
        candidate_positions.extend(candidates)
        history_positions.extend(history)
        wrong_positions.extend(wrong_history)
        candidate_offsets.append(len(candidate_positions))
        history_offsets.append(len(history_positions))
        wrong_offsets.append(len(wrong_positions))
        query_texts.append(str(record["query"]))
        request_rows.append(
            {
                "position": position,
                "original_index": original,
                "request_id": str(record["request_id"]),
                "user_id": str(record["user_id"]),
                "role": "fit" if position < len(fit_original) else "reserve",
            }
        )
    requests_path = root / "requests.jsonl"
    with requests_path.open("w", encoding="utf-8") as handle:
        for row in request_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
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
        "request_original_indices.npy": np.asarray(selected_original, dtype=np.int64),
        "request_roles.npy": roles,
        "candidate_offsets.npy": np.asarray(candidate_offsets, dtype=np.int64),
        "candidate_item_positions.npy": np.asarray(candidate_positions, dtype=np.int32),
        "history_offsets.npy": np.asarray(history_offsets, dtype=np.int64),
        "history_item_positions.npy": np.asarray(history_positions, dtype=np.int32),
        "wrong_history_offsets.npy": np.asarray(wrong_offsets, dtype=np.int64),
        "wrong_history_item_positions.npy": np.asarray(wrong_positions, dtype=np.int32),
    }
    files = {name: save_array(root, name, value) for name, value in arrays.items()}
    tokenizer = AutoTokenizer.from_pretrained(ROOT / paths["bge_snapshot"], local_files_only=True)
    token = config["tokens"]
    query_ids, query_masks = tokenize_texts(
        tokenizer,
        query_texts,
        max_length=int(token["query_tokens"]),
        batch_size=int(token["tokenizer_batch_size"]),
    )
    item_token_limit = max(int(token["candidate_tokens"]), int(token["history_item_tokens"]))
    item_tokens, item_masks = tokenize_texts(
        tokenizer,
        [text_by_id[identifier] for identifier in item_ids],
        max_length=item_token_limit,
        batch_size=int(token["tokenizer_batch_size"]),
    )
    for name, value in (
        ("query_token_ids.npy", query_ids),
        ("query_attention_mask.npy", query_masks),
        ("item_token_ids.npy", item_tokens),
        ("item_attention_mask.npy", item_masks),
    ):
        files[name] = save_array(root, name, value)
    files["requests.jsonl"] = {"path": str(requests_path.relative_to(ROOT)), "sha256": sha256_file(requests_path)}
    files["items.jsonl"] = {"path": str(items_path.relative_to(ROOT)), "sha256": sha256_file(items_path)}
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "label_free_reserve_selection_and_tokenization",
        "fit_requests": len(fit_original),
        "fit_repeat_removed": len(selection["roles"][config["selection"]["fit_role"]]["indices"]) - len(fit_original),
        "reserve_requests": len(reserve_original),
        "reserve_pool_after_exclusions": len(reserve_candidates),
        "reserve_original_indices": reserve_original,
        "fit_users": len(fit_users),
        "reserve_users": len({str(records[index]["user_id"]) for index in reserve_original}),
        "fit_reserve_user_overlap": len(
            fit_users & {str(records[index]["user_id"]) for index in reserve_original}
        ),
        "items": len(item_ids),
        "candidate_rows": len(candidate_positions),
        "history_rows": len(history_positions),
        "wrong_history_rows": len(wrong_positions),
        "wrong_same_user": int(
            sum(
                str(records[index]["user_id"]) == str(records[donor]["user_id"])
                for index, donor in wrong.items()
            )
        ),
        "item_text_conflicts_resolved": conflicts,
        "candidate_hash_reserve": candidate_hash(records, reserve_original),
        "label_access": {
            "records_train_blind": True,
            "records_train_labels": False,
            "dev_test_qrels": False,
        },
        "special_tokens": {
            "cls_token_id": int(tokenizer.cls_token_id),
            "sep_token_id": int(tokenizer.sep_token_id),
            "pad_token_id": int(tokenizer.pad_token_id),
        },
        "files": files,
    }
    atomic_json(manifest_path, report)
    print(json.dumps({key: report[key] for key in ("fit_requests", "reserve_requests", "items", "candidate_rows", "history_rows", "wrong_same_user", "candidate_hash_reserve")}, sort_keys=True))


def candidate_hash(records: list[dict[str, Any]], indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        record = records[int(index)]
        payload = json.dumps(
            [str(record["request_id"]), *[str(value["item_id"]) for value in record["candidates"]]],
            separators=(",", ":"),
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def source_paths(config: dict[str, Any], config_path: Path) -> dict[str, Path]:
    paths = config["paths"]
    output = {"config": config_path}
    output.update({key: ROOT / paths[key] for key in SOURCE_KEYS})
    return output


def input_paths(config: dict[str, Any]) -> dict[str, Path]:
    paths = config["paths"]
    snapshot = ROOT / paths["bge_snapshot"]
    output = {
        "records_train_blind": ROOT / paths["records_train_blind"],
        "records_train": ROOT / paths["records_train"],
        "c38_selection": ROOT / paths["c38_selection"],
        "c38_report": ROOT / paths["c38_report"],
        "c41_report": ROOT / paths["c41_report"],
        "c42_report": ROOT / paths["c42_report"],
        "backbone_config": snapshot / "config.json",
        "backbone_weights": snapshot / "model.safetensors",
        "tokenizer": snapshot / "tokenizer.json",
        "tokenizer_config": snapshot / "tokenizer_config.json",
        "vocab": snapshot / "vocab.txt",
    }
    root = ROOT / paths["artifact_root"]
    output.update({f"artifact_{name}": root / name for name in TOKEN_ARTIFACTS})
    return output


def freeze(config: dict[str, Any], config_path: Path) -> None:
    paths = config["paths"]
    lock_path = ROOT / paths["execution_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    lock = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "decision": "authorize_three_seed_full_token_fit_and_reserve_probe",
        "source_sha256": {key: sha256_file(path) for key, path in source_paths(config, config_path).items()},
        "input_sha256": {key: sha256_file(path) for key, path in input_paths(config).items()},
        "outcome_boundary": {
            "fit_labels": True,
            "reserve_labels_before_all_scores": False,
            "dev_test_qrels": False,
        },
    }
    atomic_json(lock_path, lock)
    print(json.dumps({"path": str(lock_path), "sha256": sha256_file(lock_path)}, sort_keys=True))


def verify_lock(config: dict[str, Any], config_path: Path) -> tuple[dict[str, Any], str]:
    lock_path = ROOT / config["paths"]["execution_lock"]
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if {key: sha256_file(path) for key, path in source_paths(config, config_path).items()} != lock["source_sha256"]:
        raise RuntimeError("token HSO source changed after lock")
    if {key: sha256_file(path) for key, path in input_paths(config).items()} != lock["input_sha256"]:
        raise RuntimeError("token HSO input changed after lock")
    return lock, sha256_file(lock_path)


def stage_labels(config: dict[str, Any], config_path: Path) -> None:
    verify_lock(config, config_path)
    paths = config["paths"]
    root = ROOT / paths["artifact_root"]
    output_path = root / "fit_labels.npz"
    report_path = root / "fit_label_manifest.json"
    if output_path.exists() or report_path.exists():
        raise FileExistsError(output_path)
    original = np.load(root / "request_original_indices.npy", mmap_mode="r")
    roles = np.load(root / "request_roles.npy", mmap_mode="r")
    wanted = {int(original[index]): index for index in np.flatnonzero(np.asarray(roles) == 0)}
    labels_by_local: dict[int, np.ndarray] = {}
    with (ROOT / paths["records_train"]).open("r", encoding="utf-8") as handle:
        for original_index, line in enumerate(handle):
            local = wanted.get(original_index)
            if local is None:
                continue
            row = json.loads(line)
            values = np.asarray(
                [float(value.get("clicked", 0) or 0) for value in row["candidates"]],
                dtype=np.float32,
            )
            if int((values > 0).sum()) != 1:
                raise ValueError("token HSO fit row does not have one positive")
            labels_by_local[local] = values
    fit_indices = np.asarray(sorted(labels_by_local), dtype=np.int64)
    offsets = [0]
    rows = []
    for local in fit_indices:
        row = labels_by_local[int(local)]
        rows.append(row)
        offsets.append(offsets[-1] + len(row))
    with output_path.open("wb") as handle:
        np.savez(
            handle,
            request_indices=fit_indices,
            offsets=np.asarray(offsets, dtype=np.int64),
            labels=np.concatenate(rows).astype(np.float32, copy=False),
        )
    report = {
        "analysis_id": config["analysis_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "post_lock_compact_fit_labels",
        "fit_requests": len(fit_indices),
        "fit_candidate_rows": offsets[-1],
        "fit_positive_rows": int(sum((row > 0).sum() for row in rows)),
        "reserve_labels_written": False,
        "dev_test_qrels_opened": False,
        "path": str(output_path.relative_to(ROOT)),
        "sha256": sha256_file(output_path),
    }
    atomic_json(report_path, report)
    print(json.dumps(report, sort_keys=True))


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    if args.stage == "prepare":
        prepare(config)
    elif args.stage == "freeze":
        freeze(config, config_path)
    else:
        stage_labels(config, config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
