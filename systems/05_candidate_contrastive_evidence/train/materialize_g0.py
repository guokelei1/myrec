"""Materialize the label-safe, exact D2p coordinate for C05 G0.

The request cohort is frozen from structural arrays before PackedRequestData
opens any label-shaped array.  The script then reconstructs the calibration
D2p scores in FP32 over every candidate and aligns them through the canonical
(request_id, candidate_item_id) key.
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
from typing import Any

import numpy as np
import torch
import transformers
import yaml
from torch.nn import functional as F


CANDIDATE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CANDIDATE_ROOT.parents[1]
sys.path.insert(0, str(CANDIDATE_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from myrec.analysis.finetuned_query_tower import (  # noqa: E402
    _zscore,
    build_model,
    iter_query_batches,
    load_tokens,
)
from myrec.analysis.supervised_diagnostics import PackedRequestData  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, sort_candidates  # noqa: E402
from train.data import (  # noqa: E402
    assert_candidate_manifest,
    assert_proposal_lock,
    freeze_selection,
    load_config,
    read_json,
    seed_everything,
    selected_candidate_key_sha256,
    sha256_file,
    validate_gpu,
    write_json,
)


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _verify_registered_inputs(
    config: dict[str, Any], *, include_label_shaped: bool
) -> dict[str, dict[str, str]]:
    """Verify source manifests without touching labels before selection."""

    packed_manifest_path = Path(config["paths"]["packed_manifest"])
    packed_manifest_hash = sha256_file(packed_manifest_path)
    if packed_manifest_hash != config["paths"]["packed_manifest_sha256"]:
        raise ValueError("packed-data manifest hash mismatch")
    packed_manifest = read_json(packed_manifest_path)
    if packed_manifest.get("status") != "passed" or packed_manifest.get(
        "qrels_read"
    ):
        raise ValueError("packed-data manifest did not pass label hygiene")
    if packed_manifest["candidate_manifest"]["sha256"] != config["paths"][
        "candidate_manifest_sha256"
    ]:
        raise ValueError("packed-data candidate manifest differs from C05")

    structural = (
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
    label_shaped = (
        "train/candidate_labels.npy",
        "train/candidate_purchase_labels.npy",
        "train/candidate_b0b_scores.npy",
    )
    packed_files = structural + label_shaped if include_label_shaped else structural
    verified: dict[str, dict[str, str]] = {
        "packed_manifest.json": {
            "path": str(packed_manifest_path),
            "sha256": packed_manifest_hash,
        }
    }
    for relative in packed_files:
        metadata = packed_manifest["files"][relative]
        actual = sha256_file(metadata["path"])
        if actual != metadata["sha256"]:
            raise ValueError(f"packed source differs from registered manifest: {relative}")
        verified[f"packed/{relative}"] = {
            "path": str(metadata["path"]),
            "sha256": actual,
        }

    token_manifest_path = Path(config["paths"]["query_token_manifest"])
    token_manifest_hash = sha256_file(token_manifest_path)
    if token_manifest_hash != config["paths"]["query_token_manifest_sha256"]:
        raise ValueError("query-token manifest hash mismatch")
    token_manifest = read_json(token_manifest_path)
    if token_manifest.get("qrels_read") or token_manifest.get("test_read"):
        raise ValueError("query-token manifest did not pass evidence hygiene")
    if int(token_manifest["request_counts"]["train"]) != int(
        config["integrity"]["packed_train_requests"]
    ):
        raise ValueError("query-token train count mismatch")
    verified["query_token_manifest.json"] = {
        "path": str(token_manifest_path),
        "sha256": token_manifest_hash,
    }
    for name in ("input_ids", "attention_mask"):
        metadata = token_manifest["output_files"]["train"][name]
        actual = sha256_file(metadata["path"])
        if actual != metadata["sha256"]:
            raise ValueError(f"train query-token source changed: {name}")
        verified[f"tokens/train_{name}.npy"] = {
            "path": str(metadata["path"]),
            "sha256": actual,
        }
    return verified


def _load_locked_model(
    config: dict[str, Any], device: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    d2_config_path = Path(config["paths"]["d2_config"])
    d2_config = _load_yaml(d2_config_path)
    if Path(d2_config["packed_data_dir"]) != Path(
        config["paths"]["packed_train_root"]
    ):
        raise ValueError("C05 packed data differs from the calibration D2 config")
    if Path(d2_config["tokenized_queries"]["output_dir"]) != Path(
        config["paths"]["query_tokens"]
    ):
        raise ValueError("C05 query tokens differ from the calibration D2 config")
    if Path(d2_config["encoder"]["frozen_item_embeddings"]) != Path(
        config["paths"]["raw_item_embeddings"]
    ):
        raise ValueError("C05 item embeddings differ from the calibration D2 config")
    actual_item_hash = sha256_file(config["paths"]["raw_item_embeddings"])
    expected_item_hash = str(d2_config["encoder"]["item_embedding_sha256"])
    if actual_item_hash != expected_item_hash:
        raise ValueError("raw item-embedding hash mismatch")

    checkpoint_path = Path(config["paths"]["calibration_checkpoint"])
    actual_checkpoint_hash = sha256_file(checkpoint_path)
    expected_checkpoint_hash = str(
        config["integrity"]["calibration_checkpoint_sha256"]
    )
    if actual_checkpoint_hash != expected_checkpoint_hash:
        raise ValueError("calibration checkpoint hash mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("analysis_id") != "finetuned_nonpersonalized_control_v1":
        raise ValueError("unexpected calibration checkpoint analysis_id")
    if int(checkpoint.get("seed", -1)) != int(config["seed"]):
        raise ValueError("unexpected calibration checkpoint seed")

    model = build_model(d2_config, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    return model, {
        "d2_config_path": str(d2_config_path),
        "d2_config_sha256": sha256_file(d2_config_path),
        "item_embeddings_sha256": actual_item_hash,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": actual_checkpoint_hash,
    }


def _ranked_ids(request_id: str, item_ids: np.ndarray, scores: np.ndarray) -> list[str]:
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


def _git_metadata() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).splitlines()
        return {"commit": commit, "dirty": bool(status), "status_short": status}
    except (OSError, subprocess.CalledProcessError) as error:
        return {"error": str(error)}


def _key_align(
    *,
    request_ids: list[str],
    candidate_item_ids: list[np.ndarray],
    canonical_scores: np.ndarray,
    offsets: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    keys: list[tuple[str, int]] = []
    for request_id, item_ids in zip(request_ids, candidate_item_ids):
        keys.extend((request_id, int(item_id)) for item_id in item_ids)
    if len(keys) != len(canonical_scores):
        raise ValueError("key/score row mismatch")
    permutation = np.random.default_rng(seed).permutation(len(keys))
    shuffled: dict[tuple[str, int], np.float32] = {}
    for position in permutation:
        key = keys[int(position)]
        if key in shuffled:
            raise ValueError(f"duplicate alignment key: {key}")
        value = np.float32(canonical_scores[int(position)])
        if not np.isfinite(value):
            raise ValueError(f"non-finite base score: {key}")
        shuffled[key] = value
    expected = set(keys)
    if set(shuffled) != expected or len(shuffled) != len(keys):
        raise ValueError("missing or unknown alignment keys")
    aligned = np.asarray([shuffled[key] for key in keys], dtype=np.float32)
    if not np.array_equal(aligned, canonical_scores):
        raise AssertionError("key-aligned base scores are not bitwise exact")

    rank_mismatches = 0
    for row, request_id in enumerate(request_ids):
        start, end = int(offsets[row]), int(offsets[row + 1])
        item_ids = candidate_item_ids[row]
        if _ranked_ids(request_id, item_ids, canonical_scores[start:end]) != _ranked_ids(
            request_id, item_ids, aligned[start:end]
        ):
            rank_mismatches += 1
    if rank_mismatches:
        raise AssertionError(f"base alignment changed {rank_mismatches} request ranks")
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


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config_path = Path(config_path)
    config = load_config(config_path)
    validate_gpu(device)
    seed_everything(int(config["seed"]))
    proposal_lock_hash = assert_proposal_lock(config)
    candidate_manifest_hash = assert_candidate_manifest(config)
    artifact_root = Path(config["paths"]["artifact_root"])
    if (artifact_root / "g0_report.json").exists():
        raise FileExistsError("immutable C05 G0 report already exists")

    preselection_inputs = _verify_registered_inputs(
        config, include_label_shaped=False
    )

    # This call opens IDs, offsets, candidate indices, and history indices only.
    # In particular it runs before PackedRequestData.load opens train labels.
    selection = freeze_selection(config)
    selected_indices = np.asarray(
        selection["fit"]["indices"] + selection["internal"]["indices"],
        dtype=np.int64,
    )
    selection_hash_before_labels = str(selection["sha256"])

    # Label-shaped source files are first opened only after the request cohort
    # is immutable.  Merge them into the same input manifest for G2a replay.
    registered_inputs = _verify_registered_inputs(
        config, include_label_shaped=True
    )
    if not all(
        registered_inputs[key] == value
        for key, value in preselection_inputs.items()
    ):
        raise ValueError("structural input changed while freezing selection")

    data = PackedRequestData.load(config["paths"]["packed_train_root"], "train")
    if len(data) != int(config["integrity"]["packed_train_requests"]):
        raise ValueError("packed train request count changed")
    input_ids, attention_mask = load_tokens(
        _load_yaml(config["paths"]["d2_config"]), "train"
    )
    if len(input_ids) != len(data) or len(attention_mask) != len(data):
        raise ValueError("train token/request count mismatch")

    popularity_path = Path(config["paths"]["internal_train_popularity"])
    popularity_hash = sha256_file(popularity_path)
    if popularity_hash != str(
        config["integrity"]["internal_train_popularity_sha256"]
    ):
        raise ValueError("internal-train popularity hash mismatch")
    popularity = np.load(popularity_path, mmap_mode="r")

    model, model_provenance = _load_locked_model(config, device)
    alpha = float(config["base"]["d2p_alpha"])
    if alpha != 0.6:
        raise ValueError("G0 preregistration requires D2p alpha 0.6")
    lower, upper = model.logit_scale_bounds
    scale = model.logit_scale.exp().clamp(min=lower, max=upper)

    query_rows: list[np.ndarray] = []
    score_rows: list[np.ndarray] = []
    item_id_rows: list[np.ndarray] = []
    selected_request_ids: list[str] = []
    candidate_offsets = [0]
    batch_count = 0
    with torch.inference_mode():
        for batch in iter_query_batches(
            data,
            selected_indices,
            int(config["base"]["max_requests_per_batch"]),
            int(config["base"]["max_padded_candidates_per_batch"]),
            int(config["seed"]),
            False,
        ):
            indices = np.asarray(batch["request_indices"], dtype=np.int64)
            token_ids = torch.from_numpy(
                np.asarray(input_ids[indices], dtype=np.int64)
            ).to(device)
            token_mask = torch.from_numpy(
                np.asarray(attention_mask[indices], dtype=np.int64)
            ).to(device)
            candidate_indices = torch.from_numpy(batch["candidate_indices"]).to(device)
            encoded = model.encoder(input_ids=token_ids, attention_mask=token_mask)
            query = F.normalize(
                encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6
            )
            candidate = model.item_embeddings[candidate_indices].float()
            candidate = F.normalize(model.item_adapter(candidate), dim=-1, eps=1e-6)
            text_scores = (
                scale * torch.einsum("bd,bcd->bc", query, candidate)
            ).float().cpu().numpy()
            query_rows.append(query.float().cpu().numpy())

            for row, raw_index in enumerate(indices):
                index = int(raw_index)
                count = int(batch["candidate_mask"][row].sum())
                embedding_indices = np.asarray(
                    batch["candidate_indices"][row, :count], dtype=np.int64
                )
                item_ids = np.asarray(
                    batch["candidate_item_ids"][row, :count], dtype=np.int64
                ).copy()
                text_z = _zscore(text_scores[row, :count])
                popularity_z = _zscore(
                    np.asarray(popularity[embedding_indices], dtype=np.float32)
                )
                mixed = np.asarray(
                    alpha * text_z + (1.0 - alpha) * popularity_z,
                    dtype=np.float32,
                )
                if not np.isfinite(mixed).all():
                    raise ValueError(f"non-finite D2p scores for request {index}")
                score_rows.append(mixed)
                item_id_rows.append(item_ids)
                selected_request_ids.append(data.request_ids[index])
                candidate_offsets.append(candidate_offsets[-1] + count)
            batch_count += 1

    query_embeddings = np.concatenate(query_rows, axis=0).astype(np.float32, copy=False)
    canonical_scores = np.concatenate(score_rows).astype(np.float32, copy=False)
    offsets = np.asarray(candidate_offsets, dtype=np.int64)
    aligned_scores, alignment = _key_align(
        request_ids=selected_request_ids,
        candidate_item_ids=item_id_rows,
        canonical_scores=canonical_scores,
        offsets=offsets,
        seed=int(config["seed"]),
    )
    if selected_request_ids != [data.request_ids[int(i)] for i in selected_indices]:
        raise AssertionError("materialized request order changed")

    artifact_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "selected_request_indices.npy": selected_indices,
        "query_embeddings.npy": query_embeddings,
        "selected_candidate_offsets.npy": offsets,
        "base_scores.npy": aligned_scores,
        "item_adapter_weight.npy": model.item_adapter.weight.detach()
        .float()
        .cpu()
        .numpy(),
    }
    for filename, values in outputs.items():
        np.save(artifact_root / filename, values)

    output_manifest = {
        filename: {
            "path": str(artifact_root / filename),
            "sha256": sha256_file(artifact_root / filename),
            "shape": list(np.load(artifact_root / filename, mmap_mode="r").shape),
            "dtype": str(np.load(artifact_root / filename, mmap_mode="r").dtype),
        }
        for filename in outputs
    }
    elapsed = time.time() - started
    packed_train = Path(config["paths"]["packed_train_root"]) / "train"
    packed_candidate_files = {
        filename: {
            "path": str(packed_train / filename),
            "sha256": sha256_file(packed_train / filename),
        }
        for filename in (
            "candidate_offsets.npy",
            "candidate_embedding_indices.npy",
            "candidate_item_ids.npy",
        )
    }
    report = {
        "candidate_id": "c05",
        "gate": "G0",
        "run_id": config["g0_run_id"],
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "proposal_lock_path": str(config["paths"]["proposal_lock"]),
        "proposal_lock_sha256": proposal_lock_hash,
        "candidate_manifest_sha256": candidate_manifest_hash,
        "selection_path": str(selection["path"]),
        "selection_sha256_frozen_before_labels": selection_hash_before_labels,
        "labels_opened_before_selection": False,
        "train_labels_opened_after_selection": True,
        "qrels_read": False,
        "dev_records_read": False,
        "test_read": False,
        "base": {
            "variant": "calibration D2p",
            "alpha": alpha,
            "dtype": "float32",
            "autocast": False,
            "candidate_sampling": False,
            "full_candidate_sets": True,
            "popularity_path": str(popularity_path),
            "popularity_sha256": popularity_hash,
            **model_provenance,
        },
        "requests": {
            "fit": len(selection["fit"]["indices"]),
            "internal": len(selection["internal"]["indices"]),
            "total": len(selected_indices),
            "candidate_rows": len(aligned_scores),
            "batches": batch_count,
            "pool_counts": selection["pool_counts"],
            "selected_candidate_key_sha256": selected_candidate_key_sha256(
                data, selected_indices
            ),
        },
        "packed_candidate_files": packed_candidate_files,
        "registered_input_files": registered_inputs,
        "alignment": alignment,
        "outputs": output_manifest,
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
            "program_device": config["program_device"],
            "registered_physical_gpu": int(config["physical_gpu"]),
            "git": _git_metadata(),
        },
    }
    report_path = artifact_root / "g0_report.json"
    run_metadata_path = Path("runs") / config["g0_run_id"] / "metadata.json"
    report["run_metadata_path"] = str(run_metadata_path)
    write_json(report_path, report)
    write_json(
        run_metadata_path,
        {
            "run_id": config["g0_run_id"],
            "candidate_id": "c05",
            "gate": "G0",
            "status": "passed",
            "created_at": report["created_at"],
            "config_path": str(config_path),
            "config_sha256": report["config_sha256"],
            "proposal_lock_sha256": proposal_lock_hash,
            "g0_report_path": str(report_path),
            "g0_report_sha256": sha256_file(report_path),
            **report["execution"],
            "qrels_read": False,
            "dev_records_read": False,
            "test_read": False,
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
