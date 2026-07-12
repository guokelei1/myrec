"""Materialize C43 fit/A features and fit labels after proposal lock."""

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
from transformers import AutoModel
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from myrec.analysis.finetuned_query_tower import _zscore, build_model, load_tokens  # noqa: E402
from train.locking import verify_proposal_lock  # noqa: E402
from train.real_data import open_original_labels  # noqa: E402
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
    if path.exists():
        raise FileExistsError(path)
    np.save(path, value)
    loaded = np.load(path, mmap_mode="r")
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
    }


def assert_cuda(config: Mapping[str, Any], device: str) -> None:
    physical = int(config["resources"]["g0_physical_gpu"])
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C43 G0 GPU registration mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C43 G0 requires exactly one visible CUDA GPU")


def selected_candidate_items(data: PackedStructure, indices: Sequence[int]) -> np.ndarray:
    rows = [data.candidate_indices(int(index)).astype(np.int64, copy=False) for index in indices]
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
                model.item_adapter(model.item_embeddings[selected].float()), dim=-1, eps=1e-6
            )
            rows.append(states.cpu().numpy())
    return np.concatenate(rows).astype(np.float32, copy=False)


def packed_rows(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    values: list[np.ndarray] = []
    for row in rows:
        array = np.asarray(row, dtype=np.int64)
        values.append(array)
        offsets.append(offsets[-1] + len(array))
    flat = np.concatenate(values).astype(np.int64, copy=False) if offsets[-1] else np.empty(0, np.int64)
    return np.asarray(offsets, dtype=np.int64), flat


def validate_registered_sources(config: Mapping[str, Any]) -> None:
    paths = config["paths"]
    integrity = config["integrity"]
    names = (
        ("c40_model_source", "c40_model_source_sha256"),
        ("c40_report", "c40_report_sha256"),
        ("c41_report", "c41_report_sha256"),
        ("c42_report", "c42_report_sha256"),
        ("c37_config", "c37_config_sha256"),
        ("c37_selection", "c37_selection_sha256"),
        ("c37_g0_report", "c37_g0_report_sha256"),
        ("c37_train_report", "c37_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("label_free_request_manifest", "label_free_request_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
        ("raw_item_embeddings", "raw_item_embeddings_sha256"),
        ("calibration_checkpoint", "calibration_checkpoint_sha256"),
        ("internal_train_popularity", "internal_train_popularity_sha256"),
        ("train_candidate_labels", "train_candidate_labels_sha256"),
        ("candidate_manifest", "candidate_manifest_sha256"),
    )
    for name, expected in names:
        if sha256_file(paths[name]) != integrity[expected]:
            raise RuntimeError(f"C43 registered source changed: {name}")
    if sha256_file(SYSTEM_ROOT / "model/metric_coupled.py") != integrity[
        "c40_model_source_sha256"
    ]:
        raise RuntimeError("C43 operator source differs from C40")


def materialize(config_path: str | Path, device: str) -> dict[str, Any]:
    started = time.time()
    config = load_config(config_path, require_selection=True)
    assert_cuda(config, device)
    _, proposal_hash = verify_proposal_lock(config)
    validate_registered_sources(config)
    paths = config["paths"]
    root = Path(paths["artifact_root"])
    report_path = root / "g0_report.json"
    if report_path.exists():
        raise FileExistsError("immutable C43 G0 exists")

    selection = read_json(paths["selection"])
    data = PackedStructure(paths["packed_train_root"])
    feature_indices = np.asarray(
        [int(value) for role in FEATURE_ROLES for value in selection["roles"][role]["indices"]],
        dtype=np.int64,
    )
    if len(feature_indices) != len(set(int(value) for value in feature_indices)):
        raise AssertionError("C43 feature roles overlap")
    donor_by_recipient: dict[int, int] = {}
    for role, row in selection["wrong_history_donors"].items():
        donor_by_recipient.update(
            {
                int(recipient): int(donor)
                for recipient, donor in zip(selection["roles"][role]["indices"], row["indices"])
            }
        )
    true_rows = [data.history_indices(int(index)).astype(np.int64, copy=False) for index in feature_indices]
    wrong_rows = [
        data.history_indices(donor_by_recipient[int(index)]).astype(np.int64, copy=False)
        if int(index) in donor_by_recipient
        else np.empty(0, dtype=np.int64)
        for index in feature_indices
    ]
    true_offsets, true_items = packed_rows(true_rows)
    wrong_offsets, wrong_items = packed_rows(wrong_rows)

    d2 = yaml_mapping(paths["d2_config"])
    if Path(d2["packed_data_dir"]) != Path(paths["packed_train_parent"]):
        raise ValueError("C43 D2 packed root differs")
    if Path(d2["tokenized_queries"]["output_dir"]) != Path(paths["query_tokens"]):
        raise ValueError("C43 D2 token root differs")
    if Path(d2["encoder"]["frozen_item_embeddings"]) != Path(paths["raw_item_embeddings"]):
        raise ValueError("C43 D2 item embedding path differs")
    if d2["encoder"]["item_embedding_sha256"] != config["integrity"][
        "raw_item_embeddings_sha256"
    ]:
        raise ValueError("C43 D2 item embedding registration differs")
    checkpoint = torch.load(paths["calibration_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("analysis_id") != "finetuned_nonpersonalized_control_v1":
        raise ValueError("C43 D2 checkpoint identity differs")
    if int(checkpoint.get("seed", -1)) != 20260708:
        raise ValueError("C43 D2 checkpoint seed differs")
    model = build_model(d2, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    transport_encoder = AutoModel.from_pretrained(paths["bge_snapshot"], local_files_only=True).to(device)
    transport_encoder.eval()
    input_ids, attention_mask = load_tokens(d2, "train")
    popularity = np.load(paths["internal_train_popularity"], mmap_mode="r")
    item_indices = selected_candidate_items(data, feature_indices)
    item_states = adapt_items(
        model,
        item_indices,
        device=device,
        batch_size=int(config["base"]["item_state_batch_size"]),
    )
    lower, upper = model.logit_scale_bounds
    scale = model.logit_scale.exp().clamp(min=lower, max=upper)
    alpha = float(config["base"]["d2p_alpha"])
    batch_size = int(config["base"]["max_requests_per_batch"])
    score_rows: list[np.ndarray] = []
    query_rows: list[np.ndarray] = []
    score_offsets = [0]
    with torch.inference_mode():
        for start in range(0, len(feature_indices), batch_size):
            requests = feature_indices[start : start + batch_size]
            token_ids = torch.from_numpy(np.asarray(input_ids[requests], dtype=np.int64)).to(device)
            token_mask = torch.from_numpy(np.asarray(attention_mask[requests], dtype=np.int64)).to(device)
            encoded = model.encoder(input_ids=token_ids, attention_mask=token_mask)
            d2_query = F.normalize(encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6)
            raw_encoded = transport_encoder(input_ids=token_ids, attention_mask=token_mask)
            raw_query = F.normalize(raw_encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6)
            query_rows.append(raw_query.cpu().numpy())
            for row, raw_index in enumerate(requests):
                index = int(raw_index)
                candidates = data.candidate_indices(index).astype(np.int64, copy=False)
                positions = np.searchsorted(item_indices, candidates)
                if not np.array_equal(item_indices[positions], candidates):
                    raise RuntimeError("C43 selected-item lookup differs")
                candidate_states = torch.from_numpy(item_states[positions]).to(device)
                text_scores = (scale * torch.mv(candidate_states, d2_query[row])).float().cpu().numpy()
                mixed = np.asarray(
                    alpha * _zscore(text_scores)
                    + (1.0 - alpha) * _zscore(np.asarray(popularity[candidates], dtype=np.float32)),
                    dtype=np.float32,
                )
                if not np.isfinite(mixed).all():
                    raise ValueError("nonfinite C43 D2p")
                score_rows.append(mixed)
                score_offsets.append(score_offsets[-1] + len(mixed))
    del model, transport_encoder, item_states
    torch.cuda.empty_cache()
    base_scores = np.concatenate(score_rows).astype(np.float32, copy=False)
    query_embeddings = np.concatenate(query_rows).astype(np.float32, copy=False)

    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = open_original_labels(
        data=data,
        indices=fit_indices,
        path=paths["train_candidate_labels"],
        expected_sha256=config["integrity"]["train_candidate_labels_sha256"],
        selection_path=paths["selection"],
        selection_sha256=paths["selection_sha256"],
    )
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": save_array(root, "feature_request_indices.npy", feature_indices),
        "feature_candidate_offsets.npy": save_array(root, "feature_candidate_offsets.npy", np.asarray(score_offsets, dtype=np.int64)),
        "base_scores.npy": save_array(root, "base_scores.npy", base_scores),
        "query_embeddings.npy": save_array(root, "query_embeddings.npy", query_embeddings),
        "history_request_indices.npy": save_array(root, "history_request_indices.npy", feature_indices),
        "true_history_offsets.npy": save_array(root, "true_history_offsets.npy", true_offsets),
        "true_history_items.npy": save_array(root, "true_history_items.npy", true_items),
        "wrong_history_offsets.npy": save_array(root, "wrong_history_offsets.npy", wrong_offsets),
        "wrong_history_items.npy": save_array(root, "wrong_history_items.npy", wrong_items),
        "fit_request_indices.npy": save_array(root, "fit_request_indices.npy", fit_labels.request_indices),
        "fit_label_offsets.npy": save_array(root, "fit_label_offsets.npy", fit_labels.offsets),
        "fit_labels.npy": save_array(root, "fit_labels.npy", fit_labels.values),
    }
    A_indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    old_features = set(
        int(value)
        for value in np.load(
            Path(paths["c37_selection"]).parent / "feature_request_indices.npy", mmap_mode="r"
        )
    )
    fit_mixed = [
        fit_labels.row(index, int(data.candidate_offsets[index + 1] - data.candidate_offsets[index]))
        for index in fit_indices
    ]
    checks = {
        "proposal_verified": True,
        "operator_source_exact": sha256_file(SYSTEM_ROOT / "model/metric_coupled.py")
        == config["integrity"]["c40_model_source_sha256"],
        "selection_bound": sha256_file(paths["selection"]) == paths["selection_sha256"],
        "feature_roles_exact": len(feature_indices) == 7712,
        "A_exact_unopened_source_union": set(A_indices)
        == set(read_json(paths["c37_selection"])["roles"]["delayed_B"]["indices"])
        | set(read_json(paths["c37_selection"])["roles"]["escrow"]["indices"]),
        "A_prior_feature_overlap_zero": not old_features.intersection(A_indices),
        "A_history_nonempty": all(data.history_count(index) > 0 for index in A_indices),
        "A_strict_nonrepeat": all(data.repeat_candidate_count(index) == 0 for index in A_indices),
        "wrong_donor_coverage": all(index in donor_by_recipient for index in fit_indices + A_indices),
        "query_shape": query_embeddings.shape == (len(feature_indices), 512),
        "base_finite": bool(np.isfinite(base_scores).all()),
        "query_finite": bool(np.isfinite(query_embeddings).all()),
        "fit_has_mixed_labels": all(bool((row > 0).any()) and bool((row <= 0).any()) for row in fit_mixed),
        "fit_labels_only": True,
        "internal_A_labels_closed": True,
        "dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c43",
        "gate": "G0",
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "proposal_lock_sha256": proposal_hash,
        "selection_sha256": paths["selection_sha256"],
        "feature_roles": list(FEATURE_ROLES),
        "checks": checks,
        "fit_labels_opened_after_proposal_lock": True,
        "internal_A_label_free_features_opened": True,
        "internal_A_labels_opened": False,
        "dev_test_qrels_read": False,
        "optimizer_steps": 0,
        "candidate_rows": len(base_scores),
        "candidate_key_sha256": candidate_key_sha256(data, feature_indices),
        "A_candidate_key_sha256": candidate_key_sha256(data, A_indices),
        "query_embedding_shape": list(query_embeddings.shape),
        "true_history_rows": len(true_items),
        "wrong_history_rows": len(wrong_items),
        "outputs": outputs,
        "physical_gpu": int(config["resources"]["g0_physical_gpu"]),
        "primary_dev_evaluator_calls": 0,
    }
    atomic_json(report_path, report)
    print(json.dumps({"candidate_id": "c43", "stage": "g0", "status": report["status"], "checks": checks}, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    materialize(args.config, args.device)


if __name__ == "__main__":
    main()
