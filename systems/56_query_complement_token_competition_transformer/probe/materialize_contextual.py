"""Materialize frozen contextual BGE tokens for the locked C56 surface."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transformers import AutoModel


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from probe.locking import (  # noqa: E402
    load_config,
    read_json,
    sha256_file,
    verify_execution,
    write_once,
)


class PackedData:
    def __init__(self, root: Path) -> None:
        self.candidate_offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_indices = np.load(root / "candidate_embedding_indices.npy", mmap_mode="r")
        self.history_offsets = np.load(root / "history_offsets.npy", mmap_mode="r")
        self.history_indices = np.load(root / "history_embedding_indices.npy", mmap_mode="r")

    def candidates(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_indices[start:stop], dtype=np.int64)

    def history(self, index: int) -> np.ndarray:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return np.asarray(self.history_indices[start:stop], dtype=np.int64)


def required_indices(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    selection = read_json(REPO_ROOT / config["paths"]["selection"])
    data = PackedData(REPO_ROOT / config["paths"]["packed_train_root"])
    request_indices = sorted(
        set(
            int(value)
            for role in ("train", "holdout", "structural_nohistory", "structural_repeat")
            for value in selection["roles"][role]
        )
    )
    donor_indices = sorted(
        set(
            int(value)
            for role in ("train", "holdout")
            for value in selection["wrong_history_donors"][role]
        )
    )
    items: set[int] = set()
    for index in request_indices:
        items.update(map(int, data.candidates(index)))
        items.update(map(int, data.history(index)))
    for index in donor_indices:
        items.update(map(int, data.history(index)))
    return np.asarray(request_indices, dtype=np.int64), np.asarray(sorted(items), dtype=np.int64)


def assert_cuda(config: Mapping[str, Any], shard: int, device: str) -> None:
    physical = int(config["resources"]["physical_gpus"][shard])
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C56 contextual GPU binding differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C56 contextual materialization requires one visible GPU")


def seed_all(seed: int = 20263700) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def encode_to_memmap(
    *,
    model: Any,
    ids: np.ndarray,
    attention: np.ndarray,
    output_path: Path,
    batch_size: int,
    hidden_dim: int,
    device: torch.device,
) -> None:
    states = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(ids), ids.shape[1], hidden_dim),
    )
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            stop = min(len(ids), start + batch_size)
            input_ids = torch.from_numpy(np.asarray(ids[start:stop], dtype=np.int64)).to(device)
            mask = torch.from_numpy(np.asarray(attention[start:stop], dtype=np.int64)).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                value = model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            states[start:stop] = value.detach().cpu().numpy().astype(np.float16, copy=False)
    states.flush()


def materialize_shard(
    config_path: str | Path, shard: int, device_name: str
) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    assert_cuda(config, shard, device_name)
    seed_all()
    shard_count = int(config["resources"]["materialization_shards"])
    if not 0 <= shard < shard_count:
        raise ValueError("C56 contextual shard differs")
    requests, items = required_indices(config)
    request_subset = requests[np.arange(len(requests)) % shard_count == shard]
    item_subset = items[np.arange(len(items)) % shard_count == shard]
    source_root = REPO_ROOT / config["paths"]["c26_artifact_root"]
    source_items = np.load(source_root / "item_embedding_indices.npy", mmap_mode="r")
    item_positions = np.searchsorted(source_items, item_subset)
    if bool((item_positions >= len(source_items)).any()) or not np.array_equal(
        source_items[item_positions], item_subset
    ):
        raise RuntimeError("C56 contextual item token source missing")
    feature_indices = np.load(source_root / "feature_request_indices.npy", mmap_mode="r")
    # C26 feature order is role-concatenated rather than globally sorted.
    position = {int(value): row for row, value in enumerate(feature_indices)}
    if any(int(value) not in position for value in request_subset):
        raise RuntimeError("C56 contextual query token source missing")
    request_positions = np.asarray(
        [position[int(value)] for value in request_subset], dtype=np.int64
    )
    item_ids = np.asarray(
        np.load(source_root / "item_token_ids.npy", mmap_mode="r")[item_positions], dtype=np.int32
    )
    item_attention = np.asarray(
        np.load(source_root / "item_attention_mask.npy", mmap_mode="r")[item_positions], dtype=bool
    )
    item_content = np.asarray(
        np.load(source_root / "item_content_mask.npy", mmap_mode="r")[item_positions], dtype=bool
    )
    query_ids = np.asarray(
        np.load(source_root / "query_token_ids.npy", mmap_mode="r")[request_positions], dtype=np.int32
    )
    query_attention = np.asarray(
        np.load(source_root / "query_attention_mask.npy", mmap_mode="r")[request_positions], dtype=bool
    )
    query_content = np.asarray(
        np.load(source_root / "query_content_mask.npy", mmap_mode="r")[request_positions], dtype=bool
    )
    root = REPO_ROOT / config["paths"]["contextual_root"]
    root.mkdir(parents=True, exist_ok=True)
    files = {
        "item_indices": root / f"item_indices_shard_{shard}.npy",
        "item_states": root / f"item_states_shard_{shard}.npy",
        "item_content": root / f"item_content_shard_{shard}.npy",
        "query_indices": root / f"query_indices_shard_{shard}.npy",
        "query_states": root / f"query_states_shard_{shard}.npy",
        "query_content": root / f"query_content_shard_{shard}.npy",
    }
    report_path = root / f"shard_{shard}_report.json"
    if report_path.exists() or any(path.exists() for path in files.values()):
        raise FileExistsError(f"immutable C56 contextual shard exists: {shard}")
    np.save(files["item_indices"], item_subset)
    np.save(files["item_content"], item_content)
    np.save(files["query_indices"], request_subset)
    np.save(files["query_content"], query_content)
    model = AutoModel.from_pretrained(
        REPO_ROOT / config["paths"]["bge_snapshot"], local_files_only=True
    ).to(torch.device(device_name))
    hidden = int(config["encoding"]["input_dim"])
    if int(model.config.hidden_size) != hidden:
        raise RuntimeError("C56 contextual LM width differs")
    encode_to_memmap(
        model=model,
        ids=item_ids,
        attention=item_attention,
        output_path=files["item_states"],
        batch_size=int(config["encoding"]["batch_size"]),
        hidden_dim=hidden,
        device=torch.device(device_name),
    )
    encode_to_memmap(
        model=model,
        ids=query_ids,
        attention=query_attention,
        output_path=files["query_states"],
        batch_size=int(config["encoding"]["batch_size"]),
        hidden_dim=hidden,
        device=torch.device(device_name),
    )
    outputs = {
        name: {
            "path": str(path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(path),
            "shape": list(np.load(path, mmap_mode="r").shape),
            "dtype": str(np.load(path, mmap_mode="r").dtype),
        }
        for name, path in files.items()
    }
    report = {
        "candidate_id": "c56",
        "stage": "contextual_materialization",
        "shard": shard,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "execution_lock_sha256": execution_hash,
        "requests": len(request_subset),
        "items": len(item_subset),
        "outputs": outputs,
        "fit_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(report_path, report)
    return report


def aggregate(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, execution_hash = verify_execution(config)
    if sha256_file(REPO_ROOT / config["paths"]["selection"]) != config["integrity"]["c56_v2_selection_sha256"]:
        raise RuntimeError("C56 v2 selection changed before v3 contextual manifest")
    requests, items = required_indices(config)
    root = REPO_ROOT / config["paths"]["contextual_root"]
    shard_count = int(config["resources"]["materialization_shards"])
    reports = [read_json(root / f"shard_{shard}_report.json") for shard in range(shard_count)]
    for report in reports:
        for row in report["outputs"].values():
            if sha256_file(REPO_ROOT / row["path"]) != row["sha256"]:
                raise RuntimeError("C56 contextual shard output changed")
    actual_items = np.concatenate(
        [np.load(REPO_ROOT / report["outputs"]["item_indices"]["path"]) for report in reports]
    )
    actual_requests = np.concatenate(
        [np.load(REPO_ROOT / report["outputs"]["query_indices"]["path"]) for report in reports]
    )
    checks = {
        "item_coverage_exact": np.array_equal(np.sort(actual_items), items),
        "query_coverage_exact": np.array_equal(np.sort(actual_requests), requests),
        "item_unique": len(np.unique(actual_items)) == len(actual_items),
        "query_unique": len(np.unique(actual_requests)) == len(actual_requests),
        "all_outputs_hash_verified": True,
        "fit_labels_closed": True,
        "C26_A_B_escrow_dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c56",
        "stage": "contextual_manifest",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "execution_lock_sha256": execution_hash,
        "required_items": len(items),
        "required_queries": len(requests),
        "checks": checks,
        "shards": reports,
        "fit_labels_read": False,
        "C26_A_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(REPO_ROOT / config["paths"]["contextual_manifest"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("shard", "aggregate"), required=True)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.stage == "shard":
        if args.shard is None:
            raise ValueError("C56 shard stage requires --shard")
        value = materialize_shard(args.config, args.shard, args.device)
    else:
        value = aggregate(args.config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
