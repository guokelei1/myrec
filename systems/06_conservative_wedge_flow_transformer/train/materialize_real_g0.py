"""Materialize the label-isolated D2p coordinate for the C06 real gate.

Only fit, internal-A, and no-history requests receive frozen query/item/base
features.  Only fit labels are copied.  Internal-B and escrow are represented
solely by their already-frozen structural IDs; no feature or label slice for
either role is opened here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import transformers
import yaml
from torch.nn import functional as F


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from myrec.analysis.finetuned_query_tower import _zscore, build_model, load_tokens  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, sort_candidates  # noqa: E402
from train.real_data import (  # noqa: E402
    FEATURE_ROLES,
    StructuralTrainData,
    assert_candidate_manifest,
    assert_real_gate_lock,
    collate_structural,
    freeze_selection,
    iter_request_batches,
    load_config,
    open_selected_labels,
    read_json,
    seed_everything,
    selected_candidate_key_sha256,
    sha256_file,
    validate_execution_authority,
    write_json,
)


STRUCTURAL_PACKED_FILES = (
    "train/request_ids.jsonl",
    "train/query_indices.npy",
    "train/timestamps.npy",
    "train/candidate_offsets.npy",
    "train/candidate_embedding_indices.npy",
    "train/candidate_item_ids.npy",
    "train/history_offsets.npy",
    "train/history_embedding_indices.npy",
    "train/history_event_weights.npy",
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _git_metadata() -> dict[str, Any]:
    try:
        return {
            "commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "status_short": subprocess.check_output(
                ["git", "status", "--short"], text=True
            ).splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def verify_registered_structural_inputs(
    config: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Verify manifests and structural/token arrays without opening labels."""

    packed_manifest_path = Path(config["paths"]["packed_manifest"])
    actual_manifest_hash = sha256_file(packed_manifest_path)
    if actual_manifest_hash != str(config["paths"]["packed_manifest_sha256"]):
        raise ValueError("packed-data manifest hash mismatch")
    manifest = read_json(packed_manifest_path)
    if manifest.get("status") != "passed" or manifest.get("qrels_read"):
        raise ValueError("packed-data manifest failed label hygiene")
    if manifest["candidate_manifest"]["sha256"] != config["paths"][
        "candidate_manifest_sha256"
    ]:
        raise ValueError("packed-data candidate manifest differs from C06")
    verified = {
        "packed_manifest.json": {
            "path": str(packed_manifest_path),
            "sha256": actual_manifest_hash,
        }
    }
    for relative in STRUCTURAL_PACKED_FILES:
        metadata = manifest["files"][relative]
        actual = sha256_file(metadata["path"])
        if actual != metadata["sha256"]:
            raise ValueError(f"registered structural source changed: {relative}")
        verified[f"packed/{relative}"] = {
            "path": str(metadata["path"]),
            "sha256": actual,
        }

    token_manifest_path = Path(config["paths"]["query_token_manifest"])
    actual_token_hash = sha256_file(token_manifest_path)
    if actual_token_hash != str(config["paths"]["query_token_manifest_sha256"]):
        raise ValueError("query-token manifest hash mismatch")
    token_manifest = read_json(token_manifest_path)
    if token_manifest.get("qrels_read") or token_manifest.get("test_read"):
        raise ValueError("query-token manifest failed evidence hygiene")
    if int(token_manifest["request_counts"]["train"]) != int(
        config["integrity"]["packed_train_requests"]
    ):
        raise ValueError("query-token train count mismatch")
    verified["query_token_manifest.json"] = {
        "path": str(token_manifest_path),
        "sha256": actual_token_hash,
    }
    for name in ("input_ids", "attention_mask"):
        metadata = token_manifest["output_files"]["train"][name]
        actual = sha256_file(metadata["path"])
        if actual != metadata["sha256"]:
            raise ValueError(f"registered query-token array changed: {name}")
        verified[f"tokens/train_{name}.npy"] = {
            "path": str(metadata["path"]),
            "sha256": actual,
        }
    return verified


def verify_label_source_after_selection(
    config: Mapping[str, Any], *, selection_path: str | Path, selection_hash: str
) -> dict[str, Any]:
    """Hash the train click-label source only after selection is immutable."""

    if sha256_file(selection_path) != selection_hash:
        raise ValueError("selection changed before post-selection label verification")
    manifest = read_json(config["paths"]["packed_manifest"])
    metadata = manifest["files"]["train/candidate_labels.npy"]
    configured_path = Path(config["paths"]["train_candidate_labels"])
    if configured_path != Path(metadata["path"]):
        raise ValueError("configured train-label source differs from packed manifest")
    actual = sha256_file(configured_path)
    if actual != metadata["sha256"]:
        raise ValueError("train-label source changed after packed manifest")
    return {
        "path": str(configured_path),
        "sha256": actual,
        "verified_after_selection_sha256": selection_hash,
        "opened_for_hash_before_selection": False,
        "full_file_bytes_hashed_after_selection": True,
        "role_label_values_decoded_by_hash": False,
    }


def _load_d2p_model(
    config: Mapping[str, Any], device: str
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    d2_config_path = Path(config["paths"]["d2_config"])
    d2_config = _load_yaml(d2_config_path)
    if Path(d2_config["packed_data_dir"]) != Path(
        config["paths"]["packed_train_root"]
    ):
        raise ValueError("C06 packed data differs from calibration D2")
    if Path(d2_config["tokenized_queries"]["output_dir"]) != Path(
        config["paths"]["query_tokens"]
    ):
        raise ValueError("C06 query tokens differ from calibration D2")
    if Path(d2_config["encoder"]["frozen_item_embeddings"]) != Path(
        config["paths"]["raw_item_embeddings"]
    ):
        raise ValueError("C06 raw items differ from calibration D2")
    item_hash = sha256_file(config["paths"]["raw_item_embeddings"])
    if item_hash != str(d2_config["encoder"]["item_embedding_sha256"]):
        raise ValueError("raw item embedding hash mismatch")
    checkpoint_path = Path(config["paths"]["calibration_checkpoint"])
    checkpoint_hash = sha256_file(checkpoint_path)
    if checkpoint_hash != str(config["integrity"]["calibration_checkpoint_sha256"]):
        raise ValueError("calibration checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("analysis_id") != "finetuned_nonpersonalized_control_v1":
        raise ValueError("unexpected calibration checkpoint")
    if int(checkpoint.get("seed", -1)) != int(config["seed"]):
        raise ValueError("calibration checkpoint seed mismatch")
    model = build_model(d2_config, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    provenance = {
        "d2_config_path": str(d2_config_path),
        "d2_config_sha256": sha256_file(d2_config_path),
        "item_embeddings_sha256": item_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": checkpoint_hash,
    }
    return model, d2_config, provenance


def _ranked_ids(
    request_id: str, item_ids: np.ndarray, scores: np.ndarray
) -> list[str]:
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


def key_align_scores(
    *,
    request_ids: Sequence[str],
    candidate_item_ids: Sequence[np.ndarray],
    canonical_scores: np.ndarray,
    offsets: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Exercise shuffled key alignment and demand bitwise score recovery."""

    keys: list[tuple[str, int]] = []
    for request_id, item_ids in zip(request_ids, candidate_item_ids):
        keys.extend((str(request_id), int(item_id)) for item_id in item_ids)
    if len(keys) != len(canonical_scores):
        raise ValueError("alignment key/score count mismatch")
    shuffled: dict[tuple[str, int], np.float32] = {}
    for raw_position in np.random.default_rng(seed).permutation(len(keys)):
        position = int(raw_position)
        key = keys[position]
        if key in shuffled:
            raise ValueError(f"duplicate alignment key: {key}")
        value = np.float32(canonical_scores[position])
        if not np.isfinite(value):
            raise ValueError(f"non-finite aligned score: {key}")
        shuffled[key] = value
    if set(shuffled) != set(keys) or len(shuffled) != len(keys):
        raise ValueError("alignment has missing or unknown keys")
    aligned = np.asarray([shuffled[key] for key in keys], dtype=np.float32)
    if not np.array_equal(aligned, canonical_scores):
        raise AssertionError("key-aligned D2p scores are not bitwise exact")
    mismatches = 0
    for row, request_id in enumerate(request_ids):
        start = int(offsets[row])
        stop = int(offsets[row + 1])
        if _ranked_ids(
            str(request_id), candidate_item_ids[row], canonical_scores[start:stop]
        ) != _ranked_ids(str(request_id), candidate_item_ids[row], aligned[start:stop]):
            mismatches += 1
    if mismatches:
        raise AssertionError("D2p key alignment changed candidate ranking")
    return aligned, {
        "key_fields": ["request_id", "candidate_item_id"],
        "source_rows_shuffled": True,
        "duplicate_keys": 0,
        "missing_keys": 0,
        "unknown_keys": 0,
        "nonfinite_scores": 0,
        "bitwise_array_equal": True,
        "rank_mismatches": 0,
    }


def _selected_item_indices(
    data: StructuralTrainData,
    request_indices: Sequence[int],
    *,
    history_limit: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for raw_index in request_indices:
        index = int(raw_index)
        cs = int(data.candidate_offsets[index])
        ce = int(data.candidate_offsets[index + 1])
        rows.append(np.asarray(data.candidate_embedding_indices[cs:ce], dtype=np.int64))
        hs = int(data.history_offsets[index])
        he = int(data.history_offsets[index + 1])
        start = max(hs, he - history_limit)
        if start < he:
            rows.append(np.asarray(data.history_embedding_indices[start:he], dtype=np.int64))
    if not rows:
        raise ValueError("no selected C06 item state exists")
    # Collators use structural index zero only as a masked padding sentinel.
    # Materializing it keeps lookup total without granting it evidence weight.
    rows.append(np.asarray([0], dtype=np.int64))
    return np.unique(np.concatenate(rows)).astype(np.int64, copy=False)


def _adapt_selected_items(
    model: torch.nn.Module,
    item_indices: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(item_indices), batch_size):
            stop = min(start + batch_size, len(item_indices))
            selected = torch.from_numpy(item_indices[start:stop]).to(device)
            raw = model.item_embeddings[selected].float()
            adapted = F.normalize(model.item_adapter(raw), dim=-1, eps=1e-6)
            rows.append(adapted.cpu().numpy())
    return np.concatenate(rows, axis=0).astype(np.float32, copy=False)


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


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    validate_execution_authority(config, stage="cohort_materialization", device=device)
    lock_hash = assert_real_gate_lock(config)
    candidate_manifest_hash = assert_candidate_manifest(config)
    seed_everything(int(config["seed"]))
    artifact_root = Path(config["paths"]["artifact_root"])
    if (artifact_root / "g0_report.json").exists():
        raise FileExistsError("immutable C06 G0 report already exists")

    registered_inputs = verify_registered_structural_inputs(config)
    # No label-bearing object exists before this immutable structural selection.
    selection = freeze_selection(config)
    selection_hash_before_labels = str(selection["sha256"])
    label_source = verify_label_source_after_selection(
        config,
        selection_path=selection["path"],
        selection_hash=selection_hash_before_labels,
    )
    data = StructuralTrainData.load(config["paths"]["packed_train_root"])
    feature_indices = np.asarray(
        [
            index
            for role in FEATURE_ROLES
            for index in selection["roles"][role]["indices"]
        ],
        dtype=np.int64,
    )
    if len(set(int(value) for value in feature_indices)) != len(feature_indices):
        raise AssertionError("G0 feature roles overlap")

    model, d2_config, model_provenance = _load_d2p_model(config, device)
    input_ids, attention_mask = load_tokens(d2_config, "train")
    if len(input_ids) != len(data) or len(attention_mask) != len(data):
        raise ValueError("query-token/request count mismatch")
    popularity_path = Path(config["paths"]["internal_train_popularity"])
    popularity_hash = sha256_file(popularity_path)
    if popularity_hash != str(config["integrity"]["internal_train_popularity_sha256"]):
        raise ValueError("internal-train popularity hash mismatch")
    popularity = np.load(popularity_path, mmap_mode="r")
    alpha = float(config["base"]["d2p_alpha"])
    if alpha != 0.6:
        raise ValueError("C06 G0 requires the registered D2p alpha 0.6")
    lower, upper = model.logit_scale_bounds
    scale = model.logit_scale.exp().clamp(min=lower, max=upper)

    query_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    item_id_rows: list[np.ndarray] = []
    request_id_rows: list[str] = []
    candidate_offsets = [0]
    batch_count = 0
    with torch.inference_mode():
        for batch_indices in iter_request_batches(
            data,
            feature_indices,
            history_limit=int(config["model"]["max_history"]),
            max_requests=int(config["base"]["max_requests_per_batch"]),
            max_padded_candidates=int(
                config["base"]["max_padded_candidates_per_batch"]
            ),
            max_padded_history=int(config["base"]["max_padded_history_per_batch"]),
            seed=int(config["seed"]),
            shuffle=False,
        ):
            batch = collate_structural(
                data, batch_indices, history_limit=int(config["model"]["max_history"])
            )
            token_ids = torch.from_numpy(
                np.asarray(input_ids[batch_indices], dtype=np.int64)
            ).to(device)
            token_mask = torch.from_numpy(
                np.asarray(attention_mask[batch_indices], dtype=np.int64)
            ).to(device)
            encoded = model.encoder(input_ids=token_ids, attention_mask=token_mask)
            query = F.normalize(encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6)
            candidate_indices = torch.from_numpy(batch["candidate_indices"]).to(device)
            candidate = F.normalize(
                model.item_adapter(model.item_embeddings[candidate_indices].float()),
                dim=-1,
                eps=1e-6,
            )
            text_scores = (
                scale * torch.einsum("bd,bcd->bc", query, candidate)
            ).float().cpu().numpy()
            query_rows.append(query.cpu().numpy())
            for row, raw_index in enumerate(batch_indices):
                index = int(raw_index)
                count = int(batch["candidate_mask"][row].sum())
                embedding_indices = np.asarray(
                    batch["candidate_indices"][row, :count], dtype=np.int64
                )
                item_ids = np.asarray(
                    batch["candidate_item_ids"][row, :count], dtype=np.int64
                ).copy()
                mixed = np.asarray(
                    alpha * _zscore(text_scores[row, :count])
                    + (1.0 - alpha)
                    * _zscore(np.asarray(popularity[embedding_indices], dtype=np.float32)),
                    dtype=np.float32,
                )
                if not np.isfinite(mixed).all():
                    raise ValueError(f"non-finite D2p scores for request {index}")
                score_rows.append(mixed)
                item_id_rows.append(item_ids)
                request_id_rows.append(data.request_ids[index])
                candidate_offsets.append(candidate_offsets[-1] + count)
            batch_count += 1

    query_embeddings = np.concatenate(query_rows).astype(np.float32, copy=False)
    canonical_scores = np.concatenate(score_rows).astype(np.float32, copy=False)
    offsets = np.asarray(candidate_offsets, dtype=np.int64)
    aligned_scores, alignment = key_align_scores(
        request_ids=request_id_rows,
        candidate_item_ids=item_id_rows,
        canonical_scores=canonical_scores,
        offsets=offsets,
        seed=int(config["seed"]),
    )
    expected_request_ids = [data.request_ids[int(index)] for index in feature_indices]
    if request_id_rows != expected_request_ids:
        raise AssertionError("G0 feature request order changed")
    item_embedding_indices = _selected_item_indices(
        data,
        feature_indices,
        history_limit=int(config["model"]["max_history"]),
    )
    item_embeddings = _adapt_selected_items(
        model,
        item_embedding_indices,
        device=device,
        batch_size=int(config["base"]["item_state_batch_size"]),
    )

    # This is the first label access in G0.  Only the frozen fit slices are read.
    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = open_selected_labels(
        data,
        fit_indices,
        label_path=config["paths"]["train_candidate_labels"],
        allowed_indices=set(fit_indices),
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": _save_array(
            artifact_root, "feature_request_indices.npy", feature_indices
        ),
        "query_embeddings.npy": _save_array(
            artifact_root, "query_embeddings.npy", query_embeddings
        ),
        "item_embeddings.npy": _save_array(
            artifact_root, "item_embeddings.npy", item_embeddings
        ),
        "item_embedding_indices.npy": _save_array(
            artifact_root, "item_embedding_indices.npy", item_embedding_indices
        ),
        "feature_candidate_offsets.npy": _save_array(
            artifact_root, "feature_candidate_offsets.npy", offsets
        ),
        "base_scores.npy": _save_array(
            artifact_root, "base_scores.npy", aligned_scores
        ),
        "fit_request_indices.npy": _save_array(
            artifact_root, "fit_request_indices.npy", fit_labels.request_indices
        ),
        "fit_label_offsets.npy": _save_array(
            artifact_root, "fit_label_offsets.npy", fit_labels.offsets
        ),
        "fit_labels.npy": _save_array(
            artifact_root, "fit_labels.npy", fit_labels.values
        ),
    }
    elapsed = time.time() - started
    report = {
        "candidate_id": "c06",
        "gate": "G0",
        "gate_id": config["gate_id"],
        "run_id": config["g0_run_id"],
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "real_gate_lock_sha256": lock_hash,
        "candidate_manifest_sha256": candidate_manifest_hash,
        "selection_path": selection["path"],
        "selection_sha256_frozen_before_labels": selection_hash_before_labels,
        "labels_opened_before_selection": False,
        "materialized_feature_roles": list(FEATURE_ROLES),
        "materialized_label_roles": ["fit"],
        "internal_A_labels_opened": False,
        "internal_B_features_or_labels_opened": False,
        "escrow_features_or_labels_opened": False,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
        "base": {
            "variant": "calibration D2p",
            "alpha": alpha,
            "dtype": "float32",
            "autocast": False,
            "full_candidate_sets": True,
            "candidate_sampling": False,
            "popularity_path": str(popularity_path),
            "popularity_sha256": popularity_hash,
            **model_provenance,
        },
        "requests": {
            **{
                role: len(selection["roles"][role]["indices"])
                for role in selection["roles"]
            },
            "feature_total": len(feature_indices),
            "feature_candidate_rows": len(aligned_scores),
            "batches": batch_count,
            "pool_counts": selection["pool_counts"],
            "feature_candidate_key_sha256": selected_candidate_key_sha256(
                data, feature_indices
            ),
            "role_candidate_key_sha256": {
                role: selected_candidate_key_sha256(
                    data, selection["roles"][role]["indices"]
                )
                for role in selection["roles"]
            },
        },
        "registered_structural_input_files": registered_inputs,
        "train_label_source_verified_after_selection": label_source,
        "alignment": alignment,
        "outputs": outputs,
        "elapsed_seconds": elapsed,
        "a40_gpu_hours_reserved": elapsed / 3600.0,
        "primary_dev_evaluator_calls": 0,
        "execution": {
            "command": " ".join(shlex.quote(value) for value in sys.argv),
            "environment": config["environment"],
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "visible_gpu_name": torch.cuda.get_device_name(0),
            "program_device": device,
            "physical_gpu": config["resources"]["g0_physical_gpu"],
            "git": _git_metadata(),
        },
    }
    report_path = artifact_root / "g0_report.json"
    write_json(report_path, report)
    run_metadata = Path("runs") / str(config["g0_run_id"]) / "metadata.json"
    write_json(
        run_metadata,
        {
            "candidate_id": "c06",
            "gate": "G0",
            "run_id": config["g0_run_id"],
            "status": "passed",
            "created_at": report["created_at"],
            "config_path": str(config_path),
            "config_sha256": report["config_sha256"],
            "real_gate_lock_sha256": lock_hash,
            "g0_report_path": str(report_path),
            "g0_report_sha256": sha256_file(report_path),
            "qrels_read": False,
            "dev_records_read": False,
            "test_read": False,
            **report["execution"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    arguments = parser.parse_args()
    materialize(arguments.config, arguments.device)


if __name__ == "__main__":
    main()
