"""Run C74's label-free pretrained-LM mechanics gate."""

from __future__ import annotations

import json
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import torch
from transformers import AutoModel


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from execution.lm_locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_g0_lock,
)
from model.adaptive_semantic_relay import (  # noqa: E402
    MODES,
    AdaptiveSemanticRelayLMRanker,
    listwise_loss,
)
from train.data_bridge import C74Store, to_device  # noqa: E402


FORWARD_NAMES = (
    "query_input_ids",
    "query_attention_mask",
    "query_content_mask",
    "candidate_input_ids",
    "candidate_attention_mask",
    "candidate_content_mask",
    "history_input_ids",
    "history_attention_mask",
    "history_content_mask",
    "history_event_mask",
    "candidate_mask",
    "base_scores",
    "item_only_scores",
    "repeat_request",
    "query_present",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def make_model(config: dict[str, Any], mode: str, device: torch.device) -> AdaptiveSemanticRelayLMRanker:
    backbone = AutoModel.from_pretrained(
        REPO_ROOT / config["paths"]["bge_snapshot"], local_files_only=True
    )
    row = config["model"]
    return AdaptiveSemanticRelayLMRanker(
        backbone=backbone,
        mode=mode,
        trainable_last_lm_layers=int(row["trainable_last_lm_layers"]),
        input_dim=int(row["input_dim"]),
        route_rank=int(row["route_rank"]),
        max_history=int(config["selection"]["max_history"]),
        temperature=float(row["temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        route_init_std=float(row["route_init_std"]),
    ).to(device)


def forward_kwargs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: batch[name] for name in FORWARD_NAMES}


def active_groups(model: AdaptiveSemanticRelayLMRanker, names: set[str]) -> dict[str, bool]:
    layers = len(model._backbone_layers())
    first = layers - model.trainable_last_lm_layers
    return {
        "first_adaptive_lm_layer": any(name.startswith(f"backbone.encoder.layer.{first}.") for name in names),
        "last_lm_layer": any(name.startswith(f"backbone.encoder.layer.{layers - 1}.") for name in names),
        "history_route_down": "history_route.down.weight" in names,
        "history_route_up": "history_route.up.weight" in names,
        "candidate_route_down": "candidate_route.down.weight" in names,
        "candidate_route_up": "candidate_route.up.weight" in names,
        "chronology_bias": "chronology_bias" in names,
        "frozen_early_lm": not any(
            name.startswith("backbone.encoder.layer.")
            and int(name.split(".")[3]) < first
            for name in names
        ),
    }


def main() -> None:
    config = load_config()
    _, g0_lock_hash = verify_g0_lock(config)
    physical = int(config["resources"]["g0_physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C74 G0 physical GPU differs")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C74 G0 requires one visible CUDA GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C74 G0 deterministic CUBLAS setting absent")
    device = torch.device("cuda:0")
    seed_all(20265200)
    store = C74Store(config, REPO_ROOT)
    if store._labels is not None:
        raise RuntimeError("C74 labels opened before G0")
    root = REPO_ROOT / config["paths"]["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    manifest = store.split_manifest()
    split_path = root / "split_manifest.json"
    atomic_json(split_path, manifest)
    indices = store.validation_indices[:2]
    raw_batch = store.collate(indices, label_access=False, history_source="true")
    batch = to_device(raw_batch, device)
    pseudo = torch.zeros_like(batch["base_scores"])
    pseudo[:, 0] = batch["candidate_mask"][:, 0].float()
    counts: dict[str, dict[str, int]] = {}
    gradients: dict[str, dict[str, bool]] = {}
    primary_model: AdaptiveSemanticRelayLMRanker | None = None
    for mode in MODES:
        seed_all(20265200)
        model = make_model(config, mode, device)
        counts[mode] = {
            "total": model.parameter_count(),
            "trainable": model.trainable_parameter_count(),
        }
        optimizer = torch.optim.AdamW(
            [value for value in model.parameters() if value.requires_grad], lr=1e-3
        )
        names: set[str] = set()
        for _ in range(3):
            output = model(**forward_kwargs(batch))
            loss = listwise_loss(output, pseudo, batch["candidate_mask"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            names |= {
                name
                for name, parameter in model.named_parameters()
                if parameter.grad is not None and bool(parameter.grad.ne(0).any())
            }
            optimizer.step()
        gradients[mode] = active_groups(model, names)
        if mode == "semantic_conservative_relay":
            primary_model = model
        else:
            del model
        torch.cuda.empty_cache()
    assert primary_model is not None
    primary_model.eval()
    with torch.inference_mode():
        output = primary_model(**forward_kwargs(batch))
        repeated = primary_model(**forward_kwargs(batch))
        reverse = torch.arange(batch["candidate_mask"].shape[1] - 1, -1, -1, device=device)
        reversed_batch = dict(batch)
        for name in (
            "candidate_input_ids",
            "candidate_attention_mask",
            "candidate_content_mask",
            "candidate_mask",
            "base_scores",
            "item_only_scores",
            "labels",
        ):
            reversed_batch[name] = batch[name][:, reverse]
        reversed_output = primary_model(**forward_kwargs(reversed_batch))
        nohistory = dict(batch)
        nohistory["history_event_mask"] = torch.zeros_like(batch["history_event_mask"])
        nohistory_output = primary_model(**forward_kwargs(nohistory))
        query_masked = dict(batch)
        query_masked["query_present"] = torch.zeros_like(batch["query_present"])
        query_output = primary_model(**forward_kwargs(query_masked))
        repeat = dict(batch)
        repeat["repeat_request"] = torch.ones_like(batch["repeat_request"])
        repeat_output = primary_model(**forward_kwargs(repeat))
    mask = batch["candidate_mask"]
    expected_base = batch["base_scores"].float().masked_fill(~mask, 0.0)
    expected_item = batch["item_only_scores"].float().masked_fill(~mask, 0.0)
    permutation_error = float((output.scores - reversed_output.scores[:, reverse]).abs().max().cpu())
    deterministic_error = float((output.scores - repeated.scores).abs().max().cpu())
    nohistory_error = float((nohistory_output.scores - expected_base).abs().max().cpu())
    query_error = float((query_output.scores - expected_base).abs().max().cpu())
    repeat_error = float((repeat_output.scores - expected_item).abs().max().cpu())
    parameter_names = [name for name, _ in primary_model.named_parameters() if not name.startswith("backbone.")]
    checks = {
        "design_gate_passed": True,
        "split_disjoint": manifest["overlap"] == 0,
        "candidate_hashes_present": len(manifest["train_candidate_hash"]) == 64 and len(manifest["validation_candidate_hash"]) == 64,
        "labels_closed": store._labels is None,
        "equal_total_parameters": len({row["total"] for row in counts.values()}) == 1,
        "equal_trainable_parameters": len({row["trainable"] for row in counts.values()}) == 1,
        "all_gradient_groups": all(all(row.values()) for row in gradients.values()),
        "primary_rank_active": bool(output.correction[mask].ne(0).any()),
        "candidate_permutation": permutation_error <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "deterministic": deterministic_error <= float(config["evaluation"]["deterministic_tolerance"]),
        "nohistory_exact": nohistory_error <= float(config["evaluation"]["exact_fallback_tolerance"]),
        "query_mask_exact": query_error <= float(config["evaluation"]["exact_fallback_tolerance"]),
        "repeat_exact": repeat_error <= float(config["evaluation"]["exact_fallback_tolerance"]),
        "no_separate_value_output_head": not any(
            fragment in name for name in parameter_names for fragment in ("value", "output_head", "relay_ffn")
        ),
        "fresh_dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c74",
        "created_at": timestamp(),
        "stage": "pretrained_lm_label_free_G0",
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "decision": "authorize_execution_lock" if all(checks.values()) else "close_before_training",
        "g0_lock_sha256": g0_lock_hash,
        "split_manifest": {
            "path": str(split_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(split_path),
            **manifest,
        },
        "model_parameters": counts,
        "gradient_groups": gradients,
        "numeric": {
            "candidate_permutation_max_abs": permutation_error,
            "deterministic_max_abs": deterministic_error,
            "nohistory_max_abs": nohistory_error,
            "query_mask_max_abs": query_error,
            "repeat_max_abs": repeat_error,
            "primary_correction_rms": float(output.correction[mask].square().mean().sqrt().cpu()),
        },
        "checks": checks,
        "fit_labels_opened": False,
        "validation_labels_opened": False,
        "fresh_dev_test_qrels_opened": False,
    }
    atomic_json(REPO_ROOT / config["paths"]["lm_probe_g0"], value)
    print(json.dumps({"status": value["status"], "checks": checks}))


if __name__ == "__main__":
    main()
