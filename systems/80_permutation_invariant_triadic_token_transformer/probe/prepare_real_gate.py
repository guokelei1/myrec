#!/usr/bin/env python
"""Materialize the label-free fresh Amazon role for the terminal C80 gate."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from transformers import AutoTokenizer
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from myrec.analysis.history_signal_observability import atomic_json, sha256_file  # noqa: E402
from myrec.analysis.token_history_observability import TokenHistoryData  # noqa: E402
from prepare_amazon_token_history_observability import (  # noqa: E402
    candidate_hash,
    choose_wrong_donor,
    has_repeat,
    item_text,
    length_bin,
    load_blind,
    save_array,
    tokenize_texts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict) or config.get("candidate_id") != "c80":
        raise ValueError("C80 config differs")
    for key in ("dev", "test", "qrels"):
        if bool(config["authorization"][key]):
            raise PermissionError(f"C80 unauthorized split: {key}")
    return config


def fit_users(path: Path, data: TokenHistoryData) -> set[str]:
    wanted = set(int(value) for value in data.fit_indices)
    output: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["position"]) in wanted:
                output.add(str(row["user_id"]))
    if not output:
        raise ValueError("C80 fit-user surface is empty")
    return output


def prepare(config: dict[str, Any]) -> None:
    paths = config["paths"]
    root = ROOT / paths["fresh_root"]
    manifest_path = root / "token_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(manifest_path)
    root.mkdir(parents=True, exist_ok=True)
    records = load_blind(ROOT / paths["records_train_blind"])
    upstream = TokenHistoryData(ROOT / paths["fit_token_root"])
    used = set(int(value) for value in np.asarray(upstream.original_indices))
    users_fit = fit_users(ROOT / paths["fit_token_root"] / "requests.jsonl", upstream)
    selection = json.loads((ROOT / paths["c38_selection"]).read_text(encoding="utf-8"))
    selected = [
        int(index)
        for index in selection["unused_indices"]
        if int(index) not in used
        and not has_repeat(records[int(index)])
        and str(records[int(index)]["user_id"]) not in users_fit
    ]
    expected = int(config["selection"]["expected_requests"])
    if len(selected) != expected:
        raise ValueError(f"C80 fresh cardinality {len(selected)} != {expected}")
    fresh_users = {str(records[index]["user_id"]) for index in selected}
    if len(fresh_users) != len(selected):
        raise ValueError("C80 fresh role is not one request per user")

    boundaries = [int(value) for value in config["selection"]["history_length_bins"]]
    bins: defaultdict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        bins[length_bin(len(record["history"]), boundaries)].append(index)
    wrong = {
        index: choose_wrong_donor(
            records,
            bins,
            index,
            boundaries,
            int(config["selection"]["wrong_seed"]),
        )
        for index in selected
    }

    text_by_id: dict[str, str] = {}
    conflicts = 0
    for original in selected:
        items = [
            *records[original]["candidates"],
            *records[original]["history"],
            *records[wrong[original]]["history"],
        ]
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
    candidate_positions: list[int] = []
    history_offsets = [0]
    history_positions: list[int] = []
    wrong_offsets = [0]
    wrong_positions: list[int] = []
    query_texts: list[str] = []
    request_rows = []
    for position, original in enumerate(selected):
        record = records[original]
        candidates = [item_position[str(value["item_id"])] for value in record["candidates"]]
        history = [item_position[str(value["item_id"])] for value in record["history"]]
        wrong_history = [
            item_position[str(value["item_id"])]
            for value in records[wrong[original]]["history"]
        ]
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
                "role": "fresh",
                "wrong_donor_original_index": wrong[original],
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
        "request_original_indices.npy": np.asarray(selected, dtype=np.int64),
        "request_roles.npy": np.ones(len(selected), dtype=np.int8),
        "candidate_offsets.npy": np.asarray(candidate_offsets, dtype=np.int64),
        "candidate_item_positions.npy": np.asarray(candidate_positions, dtype=np.int32),
        "history_offsets.npy": np.asarray(history_offsets, dtype=np.int64),
        "history_item_positions.npy": np.asarray(history_positions, dtype=np.int32),
        "wrong_history_offsets.npy": np.asarray(wrong_offsets, dtype=np.int64),
        "wrong_history_item_positions.npy": np.asarray(wrong_positions, dtype=np.int32),
    }
    files = {name: save_array(root, name, value) for name, value in arrays.items()}
    tokenizer = AutoTokenizer.from_pretrained(
        ROOT / paths["bge_snapshot"], local_files_only=True
    )
    token = config["tokens"]
    query_ids, query_masks = tokenize_texts(
        tokenizer,
        query_texts,
        max_length=int(token["query_tokens"]),
        batch_size=int(token["tokenizer_batch_size"]),
    )
    item_limit = max(int(token["candidate_tokens"]), int(token["history_item_tokens"]))
    item_tokens, item_masks = tokenize_texts(
        tokenizer,
        [text_by_id[identifier] for identifier in item_ids],
        max_length=item_limit,
        batch_size=int(token["tokenizer_batch_size"]),
    )
    for name, value in (
        ("query_token_ids.npy", query_ids),
        ("query_attention_mask.npy", query_masks),
        ("item_token_ids.npy", item_tokens),
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
    report = {
        "candidate_id": "c80",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "label_free_fresh_selection_and_tokenization",
        "fresh_requests": len(selected),
        "fresh_users": len(fresh_users),
        "fit_users": len(users_fit),
        "fit_fresh_user_overlap": len(users_fit & fresh_users),
        "upstream_requests_excluded": len(used),
        "original_indices": selected,
        "items": len(item_ids),
        "candidate_rows": len(candidate_positions),
        "history_rows": len(history_positions),
        "wrong_history_rows": len(wrong_positions),
        "wrong_same_user": sum(
            str(records[index]["user_id"]) == str(records[donor]["user_id"])
            for index, donor in wrong.items()
        ),
        "item_text_conflicts_resolved": conflicts,
        "candidate_hash_reserve": candidate_hash(records, selected),
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
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "fresh_requests",
                    "fresh_users",
                    "items",
                    "candidate_rows",
                    "wrong_same_user",
                    "candidate_hash_reserve",
                )
            },
            sort_keys=True,
        )
    )


def main() -> int:
    args = parse_args()
    prepare(load_config(Path(args.config).resolve()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
