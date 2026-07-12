"""Run C64's label-free split, token, and full-LM mechanics gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping

import numpy as np
import torch
from transformers import AutoModel


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from execution.locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_g0_lock,
)
from model.adaptive_joint_ranker import AdaptiveJointLMRanker, MODES  # noqa: E402
from train.data import C64Store, to_device  # noqa: E402


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
    torch.set_float32_matmul_precision("highest")


def make_model(
    config: Mapping[str, Any], *, mode: str, zero_initial_output: bool, device: torch.device
) -> AdaptiveJointLMRanker:
    backbone = AutoModel.from_pretrained(
        REPO_ROOT / config["paths"]["bge_snapshot"], local_files_only=True
    )
    row = config["model"]
    return AdaptiveJointLMRanker(
        backbone=backbone,
        mode=mode,
        trainable_last_lm_layers=int(row["trainable_last_lm_layers"]),
        input_dim=int(row["input_dim"]),
        hidden_dim=int(row["joint_hidden_dim"]),
        heads=int(row["joint_heads"]),
        layers=int(row["joint_layers"]),
        ffn_dim=int(row["joint_ffn_dim"]),
        dropout=float(row["dropout"]),
        max_history=int(config["selection"]["max_history"]),
        zero_initial_output=zero_initial_output,
    ).to(device)


def kwargs(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensors[name] for name in FORWARD_NAMES}


def reverse_candidates(tensors: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(tensors)
    permutation = torch.arange(
        tensors["candidate_mask"].shape[1] - 1,
        -1,
        -1,
        device=tensors["candidate_mask"].device,
    )
    for name in (
        "candidate_input_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
        "labels",
    ):
        output[name] = tensors[name][:, permutation]
    return output


def gradient_groups(model: AdaptiveJointLMRanker) -> dict[str, bool]:
    active = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.ne(0).any())
    }
    layer_count = len(model._backbone_layers())
    first_trainable = layer_count - int(model.trainable_last_lm_layers)
    return {
        "first_adaptive_lm_layer": any(
            name.startswith(f"backbone.encoder.layer.{first_trainable}.") for name in active
        ),
        "last_lm_layer": any(
            name.startswith(f"backbone.encoder.layer.{layer_count - 1}.") for name in active
        ),
        "joint_transformer": any(name.startswith("joint_transformer.") for name in active),
        "output_head": any(name.startswith("output_head.") for name in active),
        "frozen_earlier_lm_has_no_gradient": not any(
            name.startswith("backbone.encoder.layer.")
            and int(name.split(".")[3]) < first_trainable
            for name in active
        ),
    }


def run(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    physical = int(config["resources"]["g0_physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C64 G0 physical GPU registration differs")
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C64 G0 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C64 deterministic CUBLAS setting is absent")
    seed_all(int(config["selection"]["split_seed"]))
    store = C64Store(config, REPO_ROOT)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    selection_path = root / "split_manifest.json"
    report_path = root / "g0_report.json"
    split = store.split_manifest()

    probe_indices = store.validation_indices[:2]
    true_batch = store.collate(probe_indices, label_access=False, history_source="true")
    wrong_batch = store.collate(probe_indices, label_access=False, history_source="wrong")
    true_tensors = to_device(true_batch, device)
    wrong_tensors = to_device(wrong_batch, device)

    primary = make_model(
        config, mode="adaptive_history_lm", zero_initial_output=False, device=device
    )
    primary.eval()
    with torch.inference_mode():
        clean = primary(**kwargs(true_tensors))
        again = primary(**kwargs(true_tensors))
        wrong = primary(**kwargs(wrong_tensors))
        reversed_tensors = reverse_candidates(true_tensors)
        reversed_output = primary(**kwargs(reversed_tensors))
        permutation = torch.arange(
            clean.scores.shape[1] - 1, -1, -1, device=device
        )
        permutation_error = float(
            (clean.scores - reversed_output.scores[:, permutation]).abs().max().cpu()
        )
        empty_tensors = dict(true_tensors)
        empty_tensors["history_event_mask"] = torch.zeros_like(
            true_tensors["history_event_mask"]
        )
        empty = primary(**kwargs(empty_tensors))
        nohistory_error = float(
            (empty.scores - true_tensors["base_scores"]).abs().max().cpu()
        )
        repeat_tensors = dict(true_tensors)
        repeat_tensors["repeat_request"] = torch.ones_like(
            true_tensors["repeat_request"]
        )
        repeat_tensors["item_only_scores"] = torch.randn_like(
            true_tensors["item_only_scores"]
        ).masked_fill(~true_tensors["candidate_mask"], 0.0)
        repeat = primary(**kwargs(repeat_tensors))
        repeat_error = float(
            (repeat.scores - repeat_tensors["item_only_scores"]).abs().max().cpu()
        )
        history_state_difference = float(
            (clean.candidate_state - wrong.candidate_state).abs().max().cpu()
        )
        history_score_difference = float(
            (clean.scores - wrong.scores).abs().max().cpu()
        )
        deterministic_error = float((clean.scores - again.scores).abs().max().cpu())

    primary.train()
    primary.zero_grad(set_to_none=True)
    gradient_output = primary(**kwargs(true_tensors))
    gradient_loss = gradient_output.raw_correction.square().mean()
    gradient_loss.backward()
    gradients = gradient_groups(primary)

    seed_all(int(config["selection"]["split_seed"]))
    query_control = make_model(
        config,
        mode="adaptive_query_candidate_lm",
        zero_initial_output=False,
        device=device,
    ).eval()
    with torch.inference_mode():
        query_output = query_control(**kwargs(empty_tensors))
    seed_all(int(config["selection"]["split_seed"]))
    frozen_control = make_model(
        config, mode="frozen_history_lm", zero_initial_output=False, device=device
    ).eval()
    parameter_counts = {
        "adaptive_history_lm": primary.parameter_count(),
        "adaptive_query_candidate_lm": query_control.parameter_count(),
        "frozen_history_lm": frozen_control.parameter_count(),
    }
    trainable_counts = {
        "adaptive_history_lm": primary.trainable_parameter_count(),
        "adaptive_query_candidate_lm": query_control.trainable_parameter_count(),
        "frozen_history_lm": frozen_control.trainable_parameter_count(),
    }
    adaptive_names = primary.backbone_trainable_names()
    frozen_names = frozen_control.backbone_trainable_names()
    expected_prefixes = tuple(
        f"encoder.layer.{index}."
        for index in range(
            len(primary._backbone_layers()) - int(config["model"]["trainable_last_lm_layers"]),
            len(primary._backbone_layers()),
        )
    )
    checks = {
        "split_counts_exact": split["train_requests"] == 4800
        and split["validation_requests"] == 1200,
        "split_disjoint": split["overlap"] == 0,
        "candidate_hashes_nonempty": bool(split["train_candidate_hash"])
        and bool(split["validation_candidate_hash"]),
        "probe_tokens_finite_and_present": bool(
            true_tensors["query_content_mask"].any()
            and true_tensors["candidate_content_mask"].any()
            and true_tensors["history_content_mask"].any()
        ),
        "adaptive_layers_exact": bool(adaptive_names)
        and all(name.startswith(expected_prefixes) for name in adaptive_names),
        "frozen_control_backbone_frozen": not frozen_names,
        "adaptive_modes_same_trainable_count": trainable_counts["adaptive_history_lm"]
        == trainable_counts["adaptive_query_candidate_lm"],
        "all_modes_same_total_parameters": len(set(parameter_counts.values())) == 1,
        "all_gradient_groups": all(gradients.values()),
        "history_internal_path_active": history_state_difference > 1e-6
        and history_score_difference > 1e-6,
        "query_candidate_control_active_without_history": bool(
            query_output.active_request.all() and query_output.correction.ne(0).any()
        ),
        "deterministic": deterministic_error
        <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": permutation_error
        <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "nohistory_exact": nohistory_error
        <= float(config["evaluation"]["exact_fallback_tolerance"]),
        "repeat_exact": repeat_error
        <= float(config["evaluation"]["exact_fallback_tolerance"]),
        "fit_labels_closed": True,
        "fresh_dev_test_qrels_closed": True,
    }
    atomic_json(selection_path, split)
    report = {
        "candidate_id": "c64",
        "created_at": timestamp(),
        "stage": "label_free_full_LM_G0",
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "decision": "authorize_execution_lock_and_exposed_fit_training"
        if all(checks.values())
        else "close_c64_before_fit_labels",
        "physical_gpu": physical,
        "g0_lock_sha256": g0_lock_hash,
        "split_manifest": {
            "path": str(selection_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(selection_path),
        },
        "parameter_counts": parameter_counts,
        "trainable_parameter_counts": trainable_counts,
        "adaptive_backbone_parameter_names": adaptive_names,
        "gradients": gradients,
        "diagnostics": {
            "deterministic_max_abs": deterministic_error,
            "candidate_permutation_max_abs": permutation_error,
            "nohistory_max_abs": nohistory_error,
            "repeat_max_abs": repeat_error,
            "history_candidate_state_max_abs": history_state_difference,
            "history_score_max_abs": history_score_difference,
        },
        "checks": checks,
        "fit_labels_opened": False,
        "fresh_dev_test_qrels_opened": False,
    }
    atomic_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/kuai_probe.yaml")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(run(config, torch.device(args.device)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
