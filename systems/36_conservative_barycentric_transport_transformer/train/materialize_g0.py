"""Materialize label-free C36-A features and compact fit labels after proposal lock."""

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
from train.authentication import build_authentication, load_user_ids  # noqa: E402
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
        raise RuntimeError("C36 G0 GPU registration mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C36 G0 requires exactly one visible CUDA GPU")


def bootstrap_mean(values: np.ndarray, *, samples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 256):
        stop = min(samples, start + 256)
        draws = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[draws].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return {
        "requests": len(values),
        "mean": float(values.mean()),
        "percentile_95_ci": [float(low), float(high)],
        "samples": samples,
        "seed": seed,
    }


def selected_candidate_items(data: PackedStructure, indices: Sequence[int]) -> np.ndarray:
    rows = [data.candidate_indices(int(index)).astype(np.int64, copy=False) for index in indices]
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


def validate_registered_sources(config: Mapping[str, Any]) -> None:
    paths = config["paths"]
    for name, expected_name in (
        ("candidate_manifest", "candidate_manifest_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
        ("label_free_request_metadata", "label_free_request_metadata_sha256"),
        ("label_free_request_manifest", "label_free_request_manifest_sha256"),
        ("schema_incident_report", "schema_incident_report_sha256"),
        ("c35_selection", "c35_selection_sha256"),
        ("c35_g0_report", "c35_g0_report_sha256"),
        ("c35_train_report", "c35_train_report_sha256"),
        ("c34_selection", "c34_selection_sha256"),
        ("c33_selection", "c33_selection_sha256"),
        ("c32_selection", "c32_selection_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C36 registered source changed: {name}")
    c35_g0 = read_json(paths["c35_g0_report"])
    c35_outcome = read_json(paths["c35_train_report"])
    if c35_g0.get("delayed_B_features_labels_scores_opened") is not False:
        raise PermissionError("C36 source reserved role was materialized")
    if c35_outcome.get("delayed_B_features_labels_scores_opened") is not False:
        raise PermissionError("C36 source reserved boundary differs")
    if c35_outcome.get("status") != "failed_A1_terminal":
        raise PermissionError("C36 C35 terminal state differs")
    if c35_outcome.get("internal_A_labels_opened") is not True:
        raise PermissionError("C36 C35 A-label boundary differs")
    if c35_outcome.get("escrow_dev_test_opened") is not False:
        raise PermissionError("C36 C35 escrow/dev/test boundary differs")


def authentication_gate(
    authentication: Any,
    A_indices: Sequence[int],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    positions = {int(index): row for row, index in enumerate(authentication.request_indices)}
    selected = np.asarray([positions[int(index)] for index in A_indices], dtype=np.int64)
    true_counts = np.diff(authentication.true_offsets)[selected]
    wrong_counts = np.diff(authentication.wrong_offsets)[selected]
    true_source = authentication.true_source_counts[selected]
    wrong_source = authentication.wrong_source_counts[selected]
    true_fraction = true_counts / np.maximum(true_source, 1)
    wrong_fraction = wrong_counts / np.maximum(wrong_source, 1)
    difference = true_fraction - wrong_fraction
    samples = int(config["evaluation"]["bootstrap_samples"])
    seed = int(config["selection"]["seed"])
    difference_bootstrap = bootstrap_mean(difference, samples=samples, seed=seed)
    gate = config["gate"]
    checks = {
        "true_nonempty_coverage": float((true_counts > 0).mean())
        >= float(gate["g0_internal_A_true_auth_nonempty_fraction_min"]),
        "true_minus_wrong_authenticity_ci": difference_bootstrap["percentile_95_ci"][0]
        >= float(gate["g0_internal_A_true_minus_wrong_authenticity_min"]),
        "true_greater_wrong_fraction": float((true_fraction > wrong_fraction).mean())
        >= float(gate["g0_internal_A_true_greater_wrong_fraction_min"]),
        "wrong_authenticity_mean": float(wrong_fraction.mean())
        <= float(gate["g0_internal_A_wrong_authenticity_mean_max"]),
        "same_timestamp_score_before_update": authentication.audit[
            "same_timestamp_score_before_update"
        ]
        is True,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "internal_A_requests": len(A_indices),
        "true_nonempty_fraction": float((true_counts > 0).mean()),
        "wrong_nonempty_fraction": float((wrong_counts > 0).mean()),
        "true_authenticity": bootstrap_mean(true_fraction, samples=samples, seed=seed + 1),
        "wrong_authenticity": bootstrap_mean(wrong_fraction, samples=samples, seed=seed + 2),
        "true_minus_wrong_authenticity": difference_bootstrap,
        "true_greater_wrong_fraction": float((true_fraction > wrong_fraction).mean()),
        "wrong_greater_true_fraction": float((wrong_fraction > true_fraction).mean()),
    }


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
        raise FileExistsError("immutable C36 G0 exists")

    selection = read_json(paths["selection"])
    data = PackedStructure(paths["packed_train_root"])
    feature_indices = np.asarray(
        [int(value) for role in FEATURE_ROLES for value in selection["roles"][role]["indices"]],
        dtype=np.int64,
    )
    if len(feature_indices) != len(set(int(value) for value in feature_indices)):
        raise AssertionError("C36 feature roles overlap")
    feature_set = set(int(value) for value in feature_indices)
    donor_by_recipient: dict[int, int] = {}
    for role, row in selection["wrong_history_donors"].items():
        for recipient, donor in zip(selection["roles"][role]["indices"], row["indices"]):
            if int(recipient) in feature_set:
                donor_by_recipient[int(recipient)] = int(donor)
    users = load_user_ids(paths["label_free_request_metadata"], data)
    authentication = build_authentication(
        data=data,
        user_ids=users,
        target_indices=feature_indices,
        donor_by_recipient=donor_by_recipient,
    )
    A_indices = [int(value) for value in selection["roles"]["internal_A"]["indices"]]
    auth_gate = authentication_gate(authentication, A_indices, config)
    if auth_gate["status"] != "passed":
        report = {
            "candidate_id": "c36",
            "gate": "G0",
            "status": "failed_authentication_terminal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.time() - started,
            "proposal_lock_sha256": proposal_hash,
            "selection_sha256": paths["selection_sha256"],
            "authentication_gate": auth_gate,
            "authentication_audit": authentication.audit,
            "outputs": {},
            "fit_labels_opened": False,
            "internal_A_label_free_features_opened": True,
            "internal_A_labels_opened": False,
            "delayed_B_features_labels_scores_opened": False,
            "escrow_dev_test_opened": False,
            "c36_code_dev_test_qrels_metrics_read": False,
        }
        atomic_json(report_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return report

    d2 = yaml_mapping(paths["d2_config"])
    if Path(d2["packed_data_dir"]) != Path(paths["packed_train_parent"]):
        raise ValueError("C36 D2 packed root differs")
    if Path(d2["tokenized_queries"]["output_dir"]) != Path(paths["query_tokens"]):
        raise ValueError("C36 D2 token root differs")
    if Path(d2["encoder"]["frozen_item_embeddings"]) != Path(paths["raw_item_embeddings"]):
        raise ValueError("C36 D2 item embedding path differs")
    if sha256_file(paths["raw_item_embeddings"]) != d2["encoder"]["item_embedding_sha256"]:
        raise ValueError("C36 D2 item embeddings differ")
    if sha256_file(paths["calibration_checkpoint"]) != config["integrity"][
        "calibration_checkpoint_sha256"
    ]:
        raise ValueError("C36 D2 checkpoint differs")
    checkpoint = torch.load(paths["calibration_checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("analysis_id") != "finetuned_nonpersonalized_control_v1":
        raise ValueError("C36 D2 checkpoint identity differs")
    if int(checkpoint.get("seed", -1)) != 20260708:
        raise ValueError("C36 D2 checkpoint seed differs")
    model = build_model(d2, device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    # D2p keeps its registered fine-tuned query tower.  C36 transport must use
    # the untouched BGE coordinate system shared by raw query/item embeddings,
    # exactly as in the pre-outcome diagnostic.
    transport_encoder = AutoModel.from_pretrained(
        paths["bge_snapshot"], local_files_only=True
    ).to(device)
    transport_encoder.eval()
    input_ids, attention_mask = load_tokens(d2, "train")
    if sha256_file(paths["internal_train_popularity"]) != config["integrity"][
        "internal_train_popularity_sha256"
    ]:
        raise ValueError("C36 popularity differs")
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
            token_mask = torch.from_numpy(np.asarray(attention_mask[requests], dtype=np.int64)).to(
                device
            )
            encoded = model.encoder(input_ids=token_ids, attention_mask=token_mask)
            d2_query = F.normalize(
                encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6
            )
            transport_encoded = transport_encoder(
                input_ids=token_ids, attention_mask=token_mask
            )
            transport_query = F.normalize(
                transport_encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6
            )
            query_rows.append(transport_query.cpu().numpy())
            for row, raw_index in enumerate(requests):
                index = int(raw_index)
                candidates = data.candidate_indices(index).astype(np.int64, copy=False)
                positions = np.searchsorted(item_indices, candidates)
                if not np.array_equal(item_indices[positions], candidates):
                    raise RuntimeError("C36 selected item lookup differs")
                candidate_states = torch.from_numpy(item_states[positions]).to(device)
                text_scores = (
                    scale * torch.mv(candidate_states, d2_query[row])
                ).float().cpu().numpy()
                mixed = np.asarray(
                    alpha * _zscore(text_scores)
                    + (1.0 - alpha)
                    * _zscore(np.asarray(popularity[candidates], dtype=np.float32)),
                    dtype=np.float32,
                )
                if not np.isfinite(mixed).all():
                    raise ValueError("nonfinite C36 D2p")
                score_rows.append(mixed)
                score_offsets.append(score_offsets[-1] + len(mixed))
    del model, transport_encoder, item_states
    torch.cuda.empty_cache()
    base_scores = np.concatenate(score_rows).astype(np.float32, copy=False)
    query_embeddings = np.concatenate(query_rows).astype(np.float32, copy=False)
    score_offsets_array = np.asarray(score_offsets, dtype=np.int64)

    fit_indices = [int(value) for value in selection["roles"]["fit"]["indices"]]
    fit_labels = open_original_labels(
        data=data,
        indices=fit_indices,
        path=paths["train_candidate_labels"],
        selection_path=paths["selection"],
        selection_sha256=paths["selection_sha256"],
    )
    root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "feature_request_indices.npy": save_array(root, "feature_request_indices.npy", feature_indices),
        "feature_candidate_offsets.npy": save_array(
            root, "feature_candidate_offsets.npy", score_offsets_array
        ),
        "base_scores.npy": save_array(root, "base_scores.npy", base_scores),
        "query_embeddings.npy": save_array(root, "query_embeddings.npy", query_embeddings),
        "authentication_request_indices.npy": save_array(
            root, "authentication_request_indices.npy", authentication.request_indices
        ),
        "auth_true_offsets.npy": save_array(root, "auth_true_offsets.npy", authentication.true_offsets),
        "auth_true_items.npy": save_array(root, "auth_true_items.npy", authentication.true_items),
        "auth_wrong_offsets.npy": save_array(
            root, "auth_wrong_offsets.npy", authentication.wrong_offsets
        ),
        "auth_wrong_items.npy": save_array(root, "auth_wrong_items.npy", authentication.wrong_items),
        "auth_profile_sizes.npy": save_array(
            root, "auth_profile_sizes.npy", authentication.profile_sizes
        ),
        "auth_true_source_counts.npy": save_array(
            root, "auth_true_source_counts.npy", authentication.true_source_counts
        ),
        "auth_wrong_source_counts.npy": save_array(
            root, "auth_wrong_source_counts.npy", authentication.wrong_source_counts
        ),
        "fit_request_indices.npy": save_array(
            root, "fit_request_indices.npy", fit_labels.request_indices
        ),
        "fit_label_offsets.npy": save_array(root, "fit_label_offsets.npy", fit_labels.offsets),
        "fit_labels.npy": save_array(root, "fit_labels.npy", fit_labels.values),
    }
    report = {
        "candidate_id": "c36",
        "gate": "G0",
        "status": "passed",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "proposal_lock_sha256": proposal_hash,
        "selection_sha256": paths["selection_sha256"],
        "feature_roles": list(FEATURE_ROLES),
        "authentication_gate": auth_gate,
        "authentication_audit": authentication.audit,
        "fit_labels_opened_after_proposal_lock": True,
        "original_train_label_array_opened_for_fit_only": True,
        "internal_A_label_free_features_opened": True,
        "internal_A_labels_opened": False,
        "delayed_B_features_labels_scores_opened": False,
        "escrow_features_or_labels_opened": False,
        "escrow_dev_test_opened": False,
        "c36_code_dev_test_qrels_metrics_read": False,
        "global_schema_inspection_incident_registered": True,
        "candidate_rows": len(base_scores),
        "candidate_key_sha256": candidate_key_sha256(data, feature_indices),
        "fit_candidate_key_sha256": candidate_key_sha256(data, fit_indices),
        "query_embedding_shape": list(query_embeddings.shape),
        "outputs": outputs,
        "physical_gpu": int(config["resources"]["g0_physical_gpu"]),
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
