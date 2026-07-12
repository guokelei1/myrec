"""Materialize correct frozen D2p states and fit-only labels for C23."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F
import transformers
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from myrec.analysis.finetuned_query_tower import _zscore, build_model, load_tokens  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, sort_candidates  # noqa: E402
from train.locking import verify_proposal_lock  # noqa: E402
from train.real_data import FEATURE_ROLES, open_original_selected_labels  # noqa: E402
from train.structure import (  # noqa: E402
    PackedStructure,
    atomic_json,
    candidate_key_sha256,
    load_config,
    read_json,
    sha256_file,
)


def _yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    np.save(path, value)
    loaded = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
    }


def _assert_cuda(config: Mapping[str, Any], device: str) -> None:
    if device != "cuda:0":
        raise ValueError("C23 process must address its visible GPU as cuda:0")
    physical = int(config["resources"]["physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("CUDA_VISIBLE_DEVICES differs from C23 registration")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C23 materialization requires exactly one visible CUDA GPU")


def _selected_item_indices(data: PackedStructure, indices: Sequence[int]) -> np.ndarray:
    rows: list[np.ndarray] = [np.asarray([0], dtype=np.int64)]
    for raw_index in indices:
        index = int(raw_index)
        rows.append(np.asarray(data.candidate_indices(index), dtype=np.int64))
        rows.append(np.asarray(data.history_indices(index), dtype=np.int64))
    return np.unique(np.concatenate(rows)).astype(np.int64, copy=False)


def _adapt_items(
    model: torch.nn.Module,
    indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            selected = torch.from_numpy(indices[start : start + batch_size]).to(device)
            states = F.normalize(
                model.item_adapter(model.item_embeddings[selected].float()),
                dim=-1,
                eps=1e-6,
            )
            output.append(states.cpu().numpy())
    return np.concatenate(output).astype(np.float32, copy=False)


def _ranked(request_id: str, item_ids: np.ndarray, scores: np.ndarray) -> list[str]:
    return [
        row.item_id
        for row in sort_candidates(
            request_id,
            [
                ScoredCandidate(str(item_id), float(score))
                for item_id, score in zip(item_ids, scores)
            ],
        )
    ]


def _key_alignment_audit(
    *,
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
        cs, ce = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        items = np.asarray(data.candidate_item_ids[cs:ce], dtype=np.int64)
        item_rows.append(items)
        keys.extend((data.request_ids[index], int(item)) for item in items)
    if len(keys) != len(scores) or len(keys) != len(set(keys)):
        raise ValueError("C23 D2p alignment keys are not one-to-one")
    shuffled: dict[tuple[str, int], np.float32] = {}
    for raw in np.random.default_rng(seed).permutation(len(keys)):
        position = int(raw)
        shuffled[keys[position]] = np.float32(scores[position])
    recovered = np.asarray([shuffled[key] for key in keys], dtype=np.float32)
    if not np.array_equal(recovered, scores):
        raise AssertionError("C23 D2p key alignment is not bitwise exact")
    rank_mismatches = 0
    for row, raw_index in enumerate(indices):
        start, stop = int(offsets[row]), int(offsets[row + 1])
        request_id = data.request_ids[int(raw_index)]
        if _ranked(request_id, item_rows[row], scores[start:stop]) != _ranked(
            request_id, item_rows[row], recovered[start:stop]
        ):
            rank_mismatches += 1
    if rank_mismatches:
        raise AssertionError("C23 D2p alignment changed ranking")
    return {
        "key_fields": ["request_id", "candidate_item_id"],
        "keys": len(keys),
        "source_rows_shuffled": True,
        "bitwise_array_equal": True,
        "rank_mismatches": 0,
    }


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path, require_frozen_selection=True)
    if config["authorization"].get("cohort_materialization") is not True:
        raise PermissionError("C23 cohort materialization is not authorized")
    _assert_cuda(config, device)
    lock, lock_hash = verify_proposal_lock(config)
    paths = config["paths"]
    artifact_root = Path(paths["artifact_root"])
    report_path = artifact_root / "g0_report.json"
    if report_path.exists():
        raise FileExistsError("immutable C23 G0 report already exists")
    if sha256_file(paths["packed_manifest"]) != paths["packed_manifest_sha256"]:
        raise ValueError("packed manifest changed")
    if sha256_file(paths["query_token_manifest"]) != paths["query_token_manifest_sha256"]:
        raise ValueError("query token manifest changed")
    if sha256_file(paths["candidate_manifest"]) != paths["candidate_manifest_sha256"]:
        raise ValueError("candidate manifest changed")
    selection = read_json(paths["selection"])
    data = PackedStructure.load(paths["packed_train_root"])
    feature_indices = np.asarray(
        [
            int(value)
            for role in FEATURE_ROLES
            for value in selection["roles"][role]["indices"]
        ],
        dtype=np.int64,
    )
    if len(feature_indices) != len(set(int(value) for value in feature_indices)):
        raise AssertionError("C23 feature roles overlap")

    d2_config = _yaml(paths["d2_config"])
    if Path(d2_config["packed_data_dir"]) != Path(paths["packed_train_parent"]):
        raise ValueError("C23 packed root differs from registered D2")
    if Path(d2_config["tokenized_queries"]["output_dir"]) != Path(paths["query_tokens"]):
        raise ValueError("C23 query tokens differ from registered D2")
    if Path(d2_config["encoder"]["frozen_item_embeddings"]) != Path(
        paths["raw_item_embeddings"]
    ):
        raise ValueError("C23 item embeddings differ from registered D2")
    checkpoint_hash = sha256_file(paths["calibration_checkpoint"])
    if checkpoint_hash != config["integrity"]["calibration_checkpoint_sha256"]:
        raise ValueError("C23 calibration checkpoint changed")
    checkpoint = torch.load(
        paths["calibration_checkpoint"], map_location="cpu", weights_only=False
    )
    if checkpoint.get("analysis_id") != "finetuned_nonpersonalized_control_v1":
        raise ValueError("C23 calibration checkpoint identity differs")
    if int(checkpoint.get("seed", -1)) != 20260708:
        raise ValueError("C23 calibration seed differs")
    model = build_model(d2_config, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    input_ids, attention_mask = load_tokens(d2_config, "train")
    if len(input_ids) != len(data) or len(attention_mask) != len(data):
        raise ValueError("C23 query tokens differ from packed requests")
    popularity_hash = sha256_file(paths["internal_train_popularity"])
    if popularity_hash != config["integrity"]["internal_train_popularity_sha256"]:
        raise ValueError("C23 popularity input changed")
    popularity = np.load(paths["internal_train_popularity"], mmap_mode="r")
    alpha = float(config["base"]["d2p_alpha"])
    if alpha != 0.6:
        raise ValueError("C23 requires registered D2p alpha 0.6")

    item_indices = _selected_item_indices(data, feature_indices)
    item_states = _adapt_items(
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
            batch_indices = feature_indices[start : start + batch_size]
            token_ids = torch.from_numpy(
                np.asarray(input_ids[batch_indices], dtype=np.int64)
            ).to(device)
            token_mask = torch.from_numpy(
                np.asarray(attention_mask[batch_indices], dtype=np.int64)
            ).to(device)
            encoded = model.encoder(input_ids=token_ids, attention_mask=token_mask)
            query = F.normalize(encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6)
            query_rows.append(query.cpu().numpy())
            for row, raw_index in enumerate(batch_indices):
                index = int(raw_index)
                cs, ce = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
                candidate_indices = np.asarray(
                    data.candidate_embedding_indices[cs:ce], dtype=np.int64
                )
                positions = np.searchsorted(item_indices, candidate_indices)
                candidate = torch.from_numpy(item_states[positions]).to(device)
                text = (scale * torch.mv(candidate, query[row])).float().cpu().numpy()
                mixed = np.asarray(
                    alpha * _zscore(text)
                    + (1.0 - alpha)
                    * _zscore(np.asarray(popularity[candidate_indices], dtype=np.float32)),
                    dtype=np.float32,
                )
                if not np.isfinite(mixed).all():
                    raise ValueError(f"nonfinite C23 D2p score at request {index}")
                score_rows.append(mixed)
                offsets.append(offsets[-1] + len(mixed))

    query_embeddings = np.concatenate(query_rows).astype(np.float32, copy=False)
    base_scores = np.concatenate(score_rows).astype(np.float32, copy=False)
    feature_offsets = np.asarray(offsets, dtype=np.int64)
    alignment = _key_alignment_audit(
        data=data,
        indices=feature_indices,
        offsets=feature_offsets,
        scores=base_scores,
        seed=int(config["selection_seed"]),
    )

    # First C23 label-value access: only the already-frozen fit role is copied.
    manifest = read_json(paths["packed_manifest"])
    label_metadata = manifest["files"]["train/candidate_labels.npy"]
    if Path(label_metadata["path"]) != Path(paths["train_candidate_labels"]):
        raise ValueError("C23 label path differs from packed manifest")
    label_hash = sha256_file(paths["train_candidate_labels"])
    if label_hash != label_metadata["sha256"]:
        raise ValueError("C23 train label source changed")
    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = open_original_selected_labels(
        data=data,
        indices=fit_indices,
        label_path=paths["train_candidate_labels"],
        selection_sha256=paths["selection_sha256"],
        selection_path=paths["selection"],
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": _save_array(
            artifact_root, "feature_request_indices.npy", feature_indices
        ),
        "query_embeddings.npy": _save_array(
            artifact_root, "query_embeddings.npy", query_embeddings
        ),
        "item_embedding_indices.npy": _save_array(
            artifact_root, "item_embedding_indices.npy", item_indices
        ),
        "item_embeddings.npy": _save_array(
            artifact_root, "item_embeddings.npy", item_states
        ),
        "feature_candidate_offsets.npy": _save_array(
            artifact_root, "feature_candidate_offsets.npy", feature_offsets
        ),
        "base_scores.npy": _save_array(artifact_root, "base_scores.npy", base_scores),
        "fit_request_indices.npy": _save_array(
            artifact_root, "fit_request_indices.npy", fit_labels.request_indices
        ),
        "fit_label_offsets.npy": _save_array(
            artifact_root, "fit_label_offsets.npy", fit_labels.offsets
        ),
        "fit_labels.npy": _save_array(artifact_root, "fit_labels.npy", fit_labels.values),
    }
    report = {
        "candidate_id": "c23",
        "gate": "G0",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "proposal_lock_sha256": lock_hash,
        "selection_sha256_frozen_before_labels": paths["selection_sha256"],
        "labels_opened_before_selection": False,
        "materialized_feature_roles": list(FEATURE_ROLES),
        "materialized_label_roles": ["fit"],
        "internal_A_labels_opened": False,
        "delayed_B_or_escrow_features_or_labels_opened": False,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
        "requests": {
            "features": len(feature_indices),
            "fit": len(fit_indices),
            "candidate_rows": len(base_scores),
            "feature_candidate_key_sha256": candidate_key_sha256(data, feature_indices),
            "role_candidate_key_sha256": {
                role: candidate_key_sha256(
                    data, [int(value) for value in selection["roles"][role]["indices"]]
                )
                for role in selection["roles"]
            },
        },
        "base": {
            "variant": "calibration D2p",
            "alpha": alpha,
            "checkpoint_sha256": checkpoint_hash,
            "popularity_sha256": popularity_hash,
            "full_candidate_sets": True,
            "candidate_sampling": False,
        },
        "alignment": alignment,
        "train_label_source_verified_after_selection": {
            "path": str(paths["train_candidate_labels"]),
            "sha256": label_hash,
            "values_opened_only_for": "fit",
        },
        "outputs": outputs,
        "elapsed_seconds": time.time() - started,
        "primary_dev_evaluator_calls": 0,
        "execution": {
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "visible_gpu_name": torch.cuda.get_device_name(0),
            "physical_gpu": int(config["resources"]["physical_gpu"]),
        },
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
