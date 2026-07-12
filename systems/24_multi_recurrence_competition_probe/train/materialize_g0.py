"""Materialize C24 features without opening any new label values."""

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
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from myrec.analysis.finetuned_query_tower import _zscore, build_model, load_tokens  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, sort_candidates  # noqa: E402
from train.locking import verify_proposal_lock  # noqa: E402
from train.real_data import FEATURE_ROLES, slice_compact_labels  # noqa: E402
from train.structure import (  # noqa: E402
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
        raise RuntimeError("C24 GPU registration mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C24 requires one visible CUDA GPU")


def selected_items(data: PackedStructure, indices: Sequence[int]) -> np.ndarray:
    rows: list[np.ndarray] = [np.asarray([0], dtype=np.int64)]
    for raw_index in indices:
        index = int(raw_index)
        rows.append(np.asarray(data.candidate_indices(index), dtype=np.int64))
        rows.append(np.asarray(data.history_indices(index), dtype=np.int64))
    return np.unique(np.concatenate(rows)).astype(np.int64, copy=False)


def adapt_items(
    model: torch.nn.Module,
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = torch.from_numpy(indices[start : start + batch_size]).to(device)
            states = F.normalize(
                model.item_adapter(model.item_embeddings[selected].float()),
                dim=-1,
                eps=1e-6,
            )
            rows.append(states.cpu().numpy())
    return np.concatenate(rows).astype(np.float32, copy=False)


def ranked(request_id: str, items: np.ndarray, scores: np.ndarray) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [ScoredCandidate(str(item), float(score)) for item, score in zip(items, scores)],
        )
    ]


def alignment_audit(
    data: PackedStructure,
    indices: Sequence[int],
    offsets: np.ndarray,
    scores: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    keys: list[tuple[str, int]] = []
    item_rows: list[np.ndarray] = []
    for raw_index in indices:
        index = int(raw_index)
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        items = np.asarray(data.candidate_item_ids[start:stop], dtype=np.int64)
        item_rows.append(items)
        keys.extend((data.request_ids[index], int(item)) for item in items)
    if len(keys) != len(scores) or len(set(keys)) != len(keys):
        raise ValueError("C24 alignment keys differ")
    shuffled: dict[tuple[str, int], np.float32] = {}
    for raw in np.random.default_rng(seed).permutation(len(keys)):
        position = int(raw)
        shuffled[keys[position]] = np.float32(scores[position])
    recovered = np.asarray([shuffled[key] for key in keys], dtype=np.float32)
    if not np.array_equal(recovered, scores):
        raise AssertionError("C24 key recovery differs")
    mismatches = 0
    for row, raw_index in enumerate(indices):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        request_id = data.request_ids[int(raw_index)]
        mismatches += int(
            ranked(request_id, item_rows[row], scores[start:stop])
            != ranked(request_id, item_rows[row], recovered[start:stop])
        )
    if mismatches:
        raise AssertionError("C24 alignment changed ranks")
    return {"keys": len(keys), "bitwise_recovered": True, "rank_mismatches": 0}


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path, require_selection=True)
    if config["authorization"].get("cohort_materialization") is not True:
        raise PermissionError("C24 G0 not authorized")
    assert_cuda(config, device)
    _, proposal_hash = verify_proposal_lock(config)
    paths = config["paths"]
    root = Path(paths["artifact_root"])
    report_path = root / "g0_report.json"
    if report_path.exists():
        raise FileExistsError("immutable C24 G0 exists")
    for name, expected_name in (
        ("candidate_manifest", "candidate_manifest_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
        ("c23_selection", "c23_selection_sha256"),
        ("c23_g0_report", "c23_g0_report_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C24 registered source changed: {name}")
    selection = read_json(paths["selection"])
    data = PackedStructure(paths["packed_train_root"])
    feature_indices = np.asarray(
        [
            int(value)
            for role in FEATURE_ROLES
            for value in selection["roles"][role]["indices"]
        ],
        dtype=np.int64,
    )
    if len(feature_indices) != len(set(int(value) for value in feature_indices)):
        raise AssertionError("C24 feature roles overlap")

    d2 = yaml_mapping(paths["d2_config"])
    if Path(d2["packed_data_dir"]) != Path(paths["packed_train_parent"]):
        raise ValueError("C24 D2 packed root differs")
    if Path(d2["tokenized_queries"]["output_dir"]) != Path(paths["query_tokens"]):
        raise ValueError("C24 D2 token root differs")
    if Path(d2["encoder"]["frozen_item_embeddings"]) != Path(
        paths["raw_item_embeddings"]
    ):
        raise ValueError("C24 D2 item embedding path differs")
    if sha256_file(paths["raw_item_embeddings"]) != d2["encoder"][
        "item_embedding_sha256"
    ]:
        raise ValueError("C24 D2 item embeddings differ")
    checkpoint_hash = sha256_file(paths["calibration_checkpoint"])
    if checkpoint_hash != config["integrity"]["calibration_checkpoint_sha256"]:
        raise ValueError("C24 D2 checkpoint differs")
    checkpoint = torch.load(
        paths["calibration_checkpoint"], map_location="cpu", weights_only=False
    )
    if checkpoint.get("analysis_id") != "finetuned_nonpersonalized_control_v1" or int(
        checkpoint.get("seed", -1)
    ) != 20260708:
        raise ValueError("C24 D2 checkpoint identity differs")
    model = build_model(d2, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    input_ids, attention_mask = load_tokens(d2, "train")
    popularity_hash = sha256_file(paths["internal_train_popularity"])
    if popularity_hash != config["integrity"]["internal_train_popularity_sha256"]:
        raise ValueError("C24 popularity differs")
    popularity = np.load(paths["internal_train_popularity"], mmap_mode="r")
    alpha = float(config["base"]["d2p_alpha"])
    item_indices = selected_items(data, feature_indices)
    item_states = adapt_items(
        model,
        item_indices,
        device=device,
        batch_size=int(config["base"]["item_state_batch_size"]),
    )
    lower, upper = model.logit_scale_bounds
    scale = model.logit_scale.exp().clamp(min=lower, max=upper)
    query_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    offsets = [0]
    batch_size = int(config["base"]["max_requests_per_batch"])
    with torch.inference_mode():
        for start in range(0, len(feature_indices), batch_size):
            selected_requests = feature_indices[start : start + batch_size]
            token_ids = torch.from_numpy(
                np.asarray(input_ids[selected_requests], dtype=np.int64)
            ).to(device)
            token_mask = torch.from_numpy(
                np.asarray(attention_mask[selected_requests], dtype=np.int64)
            ).to(device)
            encoded = model.encoder(input_ids=token_ids, attention_mask=token_mask)
            query = F.normalize(encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6)
            query_rows.append(query.cpu().numpy())
            for row, raw_index in enumerate(selected_requests):
                index = int(raw_index)
                candidates = data.candidate_indices(index).astype(np.int64, copy=False)
                positions = np.searchsorted(item_indices, candidates)
                candidate_states = torch.from_numpy(item_states[positions]).to(device)
                text = (scale * torch.mv(candidate_states, query[row])).float().cpu().numpy()
                mixed = np.asarray(
                    alpha * _zscore(text)
                    + (1.0 - alpha)
                    * _zscore(np.asarray(popularity[candidates], dtype=np.float32)),
                    dtype=np.float32,
                )
                if not np.isfinite(mixed).all():
                    raise ValueError("nonfinite C24 D2p")
                score_rows.append(mixed)
                offsets.append(offsets[-1] + len(mixed))
    query_states = np.concatenate(query_rows).astype(np.float32, copy=False)
    base_scores = np.concatenate(score_rows).astype(np.float32, copy=False)
    score_offsets = np.asarray(offsets, dtype=np.int64)
    alignment = alignment_audit(
        data, feature_indices, score_offsets, base_scores, int(config["selection_seed"])
    )

    # Reuse only C23's already-opened compact fit labels; no original label file is opened.
    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = slice_compact_labels(paths["c23_artifact_root"], fit_indices)
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": save_array(
            root, "feature_request_indices.npy", feature_indices
        ),
        "query_embeddings.npy": save_array(root, "query_embeddings.npy", query_states),
        "item_embedding_indices.npy": save_array(
            root, "item_embedding_indices.npy", item_indices
        ),
        "item_embeddings.npy": save_array(root, "item_embeddings.npy", item_states),
        "feature_candidate_offsets.npy": save_array(
            root, "feature_candidate_offsets.npy", score_offsets
        ),
        "base_scores.npy": save_array(root, "base_scores.npy", base_scores),
        "fit_request_indices.npy": save_array(
            root, "fit_request_indices.npy", fit_labels.request_indices
        ),
        "fit_label_offsets.npy": save_array(root, "fit_label_offsets.npy", fit_labels.offsets),
        "fit_labels.npy": save_array(root, "fit_labels.npy", fit_labels.values),
    }
    report = {
        "candidate_id": "c24",
        "gate": "G0",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "proposal_lock_sha256": proposal_hash,
        "selection_sha256": paths["selection_sha256"],
        "feature_roles": list(FEATURE_ROLES),
        "fit_labels_reused_from_c23_compact": True,
        "original_train_label_array_opened": False,
        "internal_A_labels_opened": False,
        "escrow_features_or_labels_opened": False,
        "dev_test_qrels_metrics_read": False,
        "candidate_rows": len(base_scores),
        "candidate_key_sha256": candidate_key_sha256(data, feature_indices),
        "alignment": alignment,
        "base": {
            "variant": "calibration D2p",
            "alpha": alpha,
            "checkpoint_sha256": checkpoint_hash,
            "popularity_sha256": popularity_hash,
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
