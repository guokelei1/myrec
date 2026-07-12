"""Materialize C28 D2p base scores, token inputs, and compact fit labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoModel, AutoTokenizer
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from myrec.analysis.finetuned_query_tower import _zscore, build_model, load_tokens  # noqa: E402
from train.locking import verify_proposal_lock  # noqa: E402
from train.real_data import copy_compact_labels  # noqa: E402
from train.structure import (  # noqa: E402
    FEATURE_ROLES,
    PackedStructure,
    atomic_json,
    candidate_key_sha256,
    load_config,
    read_json,
    sha256_file,
)


def yaml_mapping(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    np.save(path, value)
    loaded = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
    }


def assert_cuda(config: Mapping[str, Any], device: str) -> None:
    physical = int(config["resources"]["physical_gpu"])
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C28 GPU registration mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C28 requires one visible CUDA GPU")


def selected_items(
    data: PackedStructure, indices: Sequence[int], selection: Mapping[str, Any]
) -> np.ndarray:
    rows: list[np.ndarray] = [np.asarray([0], dtype=np.int64)]
    for raw_index in indices:
        index = int(raw_index)
        rows.extend((data.candidate_indices(index), data.history_indices(index)))
    for row in selection["wrong_history_donors"].values():
        for donor in row["indices"]:
            rows.append(data.history_indices(int(donor)))
    return np.unique(np.concatenate(rows)).astype(np.int64, copy=False)


def adapt_items(
    model: torch.nn.Module, indices: np.ndarray, *, device: str, batch_size: int
) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = torch.from_numpy(indices[start : start + batch_size]).to(device)
            states = F.normalize(
                model.item_adapter(model.item_embeddings[selected].float()), dim=-1, eps=1e-6
            )
            rows.append(states.cpu().numpy())
    return np.concatenate(rows).astype(np.float32, copy=False)


def tokenize_items(
    *,
    corpus_path: str | Path,
    item_mapping_path: str | Path,
    item_indices: np.ndarray,
    tokenizer: Any,
    max_length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mapping = read_json(item_mapping_path)
    token_ids = np.zeros((len(item_indices), max_length), dtype=np.int32)
    attention = np.zeros((len(item_indices), max_length), dtype=bool)
    content = np.zeros((len(item_indices), max_length), dtype=bool)
    special = np.asarray(sorted(set(int(value) for value in tokenizer.all_special_ids)))
    pointer = 0
    positions: list[int] = []
    titles: list[str] = []

    def flush() -> None:
        if not titles:
            return
        encoded = tokenizer(
            titles,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            add_special_tokens=True,
            return_tensors="np",
        )
        values = np.asarray(encoded["input_ids"], dtype=np.int32)
        masks = np.asarray(encoded["attention_mask"], dtype=bool)
        token_ids[positions] = values
        attention[positions] = masks
        content[positions] = masks & ~np.isin(values, special)
        positions.clear()
        titles.clear()

    with Path(corpus_path).open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if pointer >= len(item_indices):
                break
            target = int(item_indices[pointer])
            if row < target:
                continue
            if row != target:
                raise ValueError(f"C28 corpus row skipped: {row}/{target}")
            value = json.loads(line)
            item_id = str(value["item_id"])
            if int(mapping[item_id]) != row:
                raise ValueError("C28 corpus/item mapping differs")
            positions.append(pointer)
            titles.append(str(value.get("item_title", "")))
            pointer += 1
            if len(titles) >= 4096:
                flush()
    flush()
    if pointer != len(item_indices):
        raise ValueError("C28 corpus coverage differs")
    return token_ids, attention, content, {
        "items": len(item_indices),
        "nonempty_content": int(content.any(axis=1).sum()),
        "content_tokens": int(content.sum()),
        "special_token_ids": special.tolist(),
        "mapping_entries": len(mapping),
    }


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path, require_selection=True)
    assert_cuda(config, device)
    _, proposal_hash = verify_proposal_lock(config)
    paths = config["paths"]
    root = Path(paths["artifact_root"])
    report_path = root / "g0_report.json"
    if report_path.exists():
        raise FileExistsError("immutable C28 G0 exists")
    for name, expected_name in (
        ("candidate_manifest", "candidate_manifest_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
        ("c27_g0_report", "c27_g0_report_sha256"),
        ("c27_train_report", "c27_train_report_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C28 registered source changed: {name}")
    c27_g0 = read_json(paths["c27_g0_report"])
    c27_outcome = read_json(paths["c27_train_report"])
    if c27_outcome.get("internal_A_labels_opened") is not False or c27_outcome.get(
        "delayed_B_labels_opened"
    ) is not False or c27_g0.get("escrow_features_or_labels_opened") is not False:
        raise PermissionError("C28 source outcome isolation differs")
    selection = read_json(paths["selection"])
    data = PackedStructure(paths["packed_train_root"])
    feature_indices = np.asarray(
        [int(value) for role in FEATURE_ROLES for value in selection["roles"][role]["indices"]],
        dtype=np.int64,
    )
    if len(feature_indices) != len(set(int(value) for value in feature_indices)):
        raise AssertionError("C28 feature roles overlap")

    d2 = yaml_mapping(paths["d2_config"])
    if Path(d2["packed_data_dir"]) != Path(paths["packed_train_parent"]):
        raise ValueError("C28 D2 packed root differs")
    if Path(d2["tokenized_queries"]["output_dir"]) != Path(paths["query_tokens"]):
        raise ValueError("C28 D2 token root differs")
    if Path(d2["encoder"]["frozen_item_embeddings"]) != Path(paths["raw_item_embeddings"]):
        raise ValueError("C28 D2 item embedding path differs")
    if sha256_file(paths["raw_item_embeddings"]) != d2["encoder"]["item_embedding_sha256"]:
        raise ValueError("C28 D2 item embeddings differ")
    if sha256_file(paths["calibration_checkpoint"]) != config["integrity"][
        "calibration_checkpoint_sha256"
    ]:
        raise ValueError("C28 D2 checkpoint differs")
    checkpoint = torch.load(paths["calibration_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("analysis_id") != "finetuned_nonpersonalized_control_v1" or int(
        checkpoint.get("seed", -1)
    ) != 20260708:
        raise ValueError("C28 D2 checkpoint identity differs")
    model = build_model(d2, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    input_ids, attention_mask = load_tokens(d2, "train")
    if sha256_file(paths["internal_train_popularity"]) != config["integrity"][
        "internal_train_popularity_sha256"
    ]:
        raise ValueError("C28 popularity differs")
    popularity = np.load(paths["internal_train_popularity"], mmap_mode="r")
    item_indices = selected_items(data, feature_indices, selection)
    item_states = adapt_items(
        model,
        item_indices,
        device=device,
        batch_size=int(config["base"]["item_state_batch_size"]),
    )
    lower, upper = model.logit_scale_bounds
    scale = model.logit_scale.exp().clamp(min=lower, max=upper)
    score_rows: list[np.ndarray] = []
    offsets = [0]
    batch_size = int(config["base"]["max_requests_per_batch"])
    alpha = float(config["base"]["d2p_alpha"])
    with torch.inference_mode():
        for start in range(0, len(feature_indices), batch_size):
            requests = feature_indices[start : start + batch_size]
            token_ids = torch.from_numpy(np.asarray(input_ids[requests], dtype=np.int64)).to(device)
            token_mask = torch.from_numpy(
                np.asarray(attention_mask[requests], dtype=np.int64)
            ).to(device)
            encoded = model.encoder(input_ids=token_ids, attention_mask=token_mask)
            query = F.normalize(encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6)
            for row, raw_index in enumerate(requests):
                index = int(raw_index)
                candidates = data.candidate_indices(index).astype(np.int64, copy=False)
                positions = np.searchsorted(item_indices, candidates)
                candidate_states = torch.from_numpy(item_states[positions]).to(device)
                text_scores = (scale * torch.mv(candidate_states, query[row])).float().cpu().numpy()
                mixed = np.asarray(
                    alpha * _zscore(text_scores)
                    + (1.0 - alpha)
                    * _zscore(np.asarray(popularity[candidates], dtype=np.float32)),
                    dtype=np.float32,
                )
                if not np.isfinite(mixed).all():
                    raise ValueError("nonfinite C28 D2p")
                score_rows.append(mixed)
                offsets.append(offsets[-1] + len(mixed))
    del model, item_states
    torch.cuda.empty_cache()
    base_scores = np.concatenate(score_rows).astype(np.float32, copy=False)
    score_offsets = np.asarray(offsets, dtype=np.int64)

    tokenizer = AutoTokenizer.from_pretrained(paths["bge_snapshot"], local_files_only=True)
    item_token_ids, item_attention, item_content, item_audit = tokenize_items(
        corpus_path=paths["corpus"],
        item_mapping_path=paths["item_id2idx"],
        item_indices=item_indices,
        tokenizer=tokenizer,
        max_length=int(config["tokenization"]["max_item_tokens"]),
    )
    query_length = int(config["tokenization"]["max_query_tokens"])
    query_ids = np.asarray(input_ids[feature_indices, :query_length], dtype=np.int32)
    query_attention = np.asarray(attention_mask[feature_indices, :query_length], dtype=bool)
    special = np.asarray(sorted(set(int(value) for value in tokenizer.all_special_ids)))
    query_content = query_attention & ~np.isin(query_ids, special)
    encoder = AutoModel.from_pretrained(paths["bge_snapshot"], local_files_only=True)
    embeddings = encoder.embeddings.word_embeddings.weight.detach().cpu().float().numpy()
    padding_idx = int(tokenizer.pad_token_id)
    del encoder

    c27_root = Path(paths["c27_artifact_root"])
    for name in ("fit_request_indices.npy", "fit_label_offsets.npy", "fit_labels.npy"):
        if sha256_file(c27_root / name) != c27_g0["outputs"][name]["sha256"]:
            raise RuntimeError(f"C28 frozen C27 fit-label input changed: {name}")
    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = copy_compact_labels(c27_root, fit_indices)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": save_array(root, "feature_request_indices.npy", feature_indices),
        "feature_candidate_offsets.npy": save_array(
            root, "feature_candidate_offsets.npy", score_offsets
        ),
        "base_scores.npy": save_array(root, "base_scores.npy", base_scores),
        "query_token_ids.npy": save_array(root, "query_token_ids.npy", query_ids),
        "query_attention_mask.npy": save_array(
            root, "query_attention_mask.npy", query_attention
        ),
        "query_content_mask.npy": save_array(root, "query_content_mask.npy", query_content),
        "item_embedding_indices.npy": save_array(root, "item_embedding_indices.npy", item_indices),
        "item_token_ids.npy": save_array(root, "item_token_ids.npy", item_token_ids),
        "item_attention_mask.npy": save_array(
            root, "item_attention_mask.npy", item_attention
        ),
        "item_content_mask.npy": save_array(root, "item_content_mask.npy", item_content),
        "word_embeddings.npy": save_array(root, "word_embeddings.npy", embeddings),
        "fit_request_indices.npy": save_array(
            root, "fit_request_indices.npy", fit_labels.request_indices
        ),
        "fit_label_offsets.npy": save_array(root, "fit_label_offsets.npy", fit_labels.offsets),
        "fit_labels.npy": save_array(root, "fit_labels.npy", fit_labels.values),
    }
    report = {
        "candidate_id": "c28",
        "gate": "G0",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "proposal_lock_sha256": proposal_hash,
        "selection_sha256": paths["selection_sha256"],
        "feature_roles": list(FEATURE_ROLES),
        "fit_labels_reused_from_c27_compact": True,
        "original_train_label_array_opened": False,
        "internal_A_labels_opened": False,
        "delayed_B_labels_opened": False,
        "escrow_features_or_labels_opened": False,
        "dev_test_qrels_metrics_read": False,
        "candidate_rows": len(base_scores),
        "candidate_key_sha256": candidate_key_sha256(data, feature_indices),
        "query_nonempty_content": int(query_content.any(axis=1).sum()),
        "tokenization": {
            "padding_idx": padding_idx,
            "vocab_size": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            **item_audit,
        },
        "outputs": outputs,
        "physical_gpu": int(config["resources"]["physical_gpu"]),
        "primary_dev_evaluator_calls": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    materialize(args.config, args.device)


if __name__ == "__main__":
    main()
