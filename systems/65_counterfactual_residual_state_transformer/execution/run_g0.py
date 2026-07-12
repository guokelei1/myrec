"""Run C65's label-free real-token structural gate."""

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
    timestamp,
    verify_g0_lock,
)
from model.counterfactual_residual import (  # noqa: E402
    MODES,
    CounterfactualResidualStateTransformer,
)
from train.data_bridge import C64Store, to_device  # noqa: E402


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
    config: Mapping[str, Any], *, mode: str, zero: bool, device: torch.device
) -> CounterfactualResidualStateTransformer:
    backbone = AutoModel.from_pretrained(
        REPO_ROOT / config["paths"]["bge_snapshot"], local_files_only=True
    )
    row = config["model"]
    return CounterfactualResidualStateTransformer(
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
        zero_initial_output=zero,
        null_reference_stop_gradient=bool(row["null_reference_stop_gradient"]),
    ).to(device)


def merge_batches(
    true_tensors: Mapping[str, torch.Tensor], wrong_tensors: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    output = dict(true_tensors)
    output.update(
        {
            "wrong_history_input_ids": wrong_tensors["history_input_ids"],
            "wrong_history_attention_mask": wrong_tensors["history_attention_mask"],
            "wrong_history_content_mask": wrong_tensors["history_content_mask"],
            "wrong_history_event_mask": wrong_tensors["history_event_mask"],
        }
    )
    output.pop("labels", None)
    return output


def reverse_candidates(values: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = dict(values)
    permutation = torch.arange(
        values["candidate_mask"].shape[1] - 1,
        -1,
        -1,
        device=values["candidate_mask"].device,
    )
    for name in (
        "candidate_input_ids",
        "candidate_attention_mask",
        "candidate_content_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        output[name] = values[name][:, permutation]
    return output


def gradient_groups(
    model: CounterfactualResidualStateTransformer,
) -> dict[str, bool]:
    active = {
        name
        for name, parameter in model.named_parameters()
        if parameter.grad is not None and bool(parameter.grad.ne(0).any())
    }
    layers = len(model.backbone_layers())
    first = layers - model.trainable_last_lm_layers
    return {
        "first_adaptive_lm_layer": any(
            name.startswith(f"core.backbone.encoder.layer.{first}.") for name in active
        ),
        "last_lm_layer": any(
            name.startswith(f"core.backbone.encoder.layer.{layers - 1}.") for name in active
        ),
        "joint_transformer": any(
            name.startswith("core.joint_transformer.") for name in active
        ),
        "output_head": any(name.startswith("core.output_head.") for name in active),
        "residual_norm": any(name.startswith("residual_norm.") for name in active),
        "frozen_earlier_lm_has_no_gradient": not any(
            name.startswith("core.backbone.encoder.layer.")
            and int(name.split(".")[4]) < first
            for name in active
        ),
    }


def run(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    _, g0_lock_hash = verify_g0_lock(config)
    physical = int(config["resources"]["g0_physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C65 G0 physical GPU differs")
    if str(device) != "cuda:0" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C65 G0 requires one registered visible GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C65 deterministic CUBLAS setting is absent")
    seed_all(20264600)
    store = C64Store(config, REPO_ROOT)
    frozen_split = json.loads(
        (REPO_ROOT / config["paths"]["c64_split_manifest"]).read_text(encoding="utf-8")
    )
    current_split = store.split_manifest()
    probe_indices = store.validation_indices[:2]
    true_batch = store.collate(probe_indices, label_access=False, history_source="true")
    wrong_batch = store.collate(probe_indices, label_access=False, history_source="wrong")
    true_tensors = to_device(true_batch, device)
    wrong_tensors = to_device(wrong_batch, device)
    values = merge_batches(true_tensors, wrong_tensors)

    seed_all(20264600)
    primary = make_model(
        config,
        mode="hidden_residual_wrong_neutral",
        zero=False,
        device=device,
    ).eval()
    primary.backbone.eval()
    with torch.inference_mode():
        clean = primary(**values)
        again = primary(**values)
        reversed_values = reverse_candidates(values)
        reversed_output = primary(**reversed_values)
        permutation = torch.arange(clean.scores.shape[1] - 1, -1, -1, device=device)
        permutation_error = float(
            (clean.scores - reversed_output.scores[:, permutation]).abs().max().cpu()
        )
        deterministic_error = float((clean.scores - again.scores).abs().max().cpu())
        empty_values = dict(values)
        empty_values["history_event_mask"] = torch.zeros_like(values["history_event_mask"])
        empty = primary(**empty_values)
        nohistory_error = float(
            (empty.scores - values["base_scores"]).abs().max().cpu()
        )
        repeat_values = dict(values)
        repeat_values["repeat_request"] = torch.ones_like(values["repeat_request"])
        repeat_values["item_only_scores"] = torch.randn_like(
            values["item_only_scores"]
        ).masked_fill(~values["candidate_mask"], 0.0)
        repeat = primary(**repeat_values)
        repeat_error = float(
            (repeat.scores - repeat_values["item_only_scores"]).abs().max().cpu()
        )
        residual_rms = float(clean.state_residual.square().mean().sqrt().cpu())
        wrong_residual_rms = float(
            clean.wrong_state_residual.square().mean().sqrt().cpu()
        )
        true_wrong_score_difference = float(
            (clean.scores - clean.wrong_scores).abs().max().cpu()
        )

    primary.train()
    primary.backbone.eval()
    primary.zero_grad(set_to_none=True)
    gradient_output = primary(**values)
    gradient_loss = (
        gradient_output.raw_correction.square().mean()
        + gradient_output.raw_wrong_correction.square().mean()
    )
    gradient_loss.backward()
    gradients = gradient_groups(primary)

    parameter_counts = {"hidden_residual_wrong_neutral": primary.parameter_count()}
    mode_outputs = {"hidden_residual_wrong_neutral": clean.scores}
    for mode in MODES[1:]:
        seed_all(20264600)
        model = make_model(config, mode=mode, zero=False, device=device).eval()
        model.backbone.eval()
        parameter_counts[mode] = model.parameter_count()
        with torch.inference_mode():
            mode_outputs[mode] = model(**values).scores
        del model
    checks = {
        "split_counts_exact": len(store.train_indices) == 4800
        and len(store.validation_indices) == 1200,
        "split_matches_c64": current_split["train_candidate_hash"]
        == frozen_split["train_candidate_hash"]
        and current_split["validation_candidate_hash"]
        == frozen_split["validation_candidate_hash"],
        "candidate_hash_asserted": bool(current_split["validation_candidate_hash"]),
        "all_modes_same_parameters": len(set(parameter_counts.values())) == 1,
        "all_gradient_groups": all(gradients.values()),
        "hidden_residual_active": residual_rms > 1e-5,
        "wrong_residual_active": wrong_residual_rms > 1e-5,
        "true_wrong_score_active": true_wrong_score_difference > 1e-6,
        "hidden_differs_from_ordinary": not torch.equal(
            mode_outputs["hidden_residual_wrong_neutral"],
            mode_outputs["ordinary_factual_wrong_neutral"],
        ),
        "hidden_differs_from_logit": not torch.equal(
            mode_outputs["hidden_residual_wrong_neutral"],
            mode_outputs["logit_difference_wrong_neutral"],
        ),
        "null_reference_stop_gradient": primary.null_reference_stop_gradient,
        "deterministic": deterministic_error
        <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": permutation_error
        <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "nohistory_exact": nohistory_error
        <= float(config["evaluation"]["exact_fallback_tolerance"]),
        "repeat_exact": repeat_error
        <= float(config["evaluation"]["exact_fallback_tolerance"]),
        "train_validation_fresh_dev_test_qrels_labels_closed": True,
    }
    report = {
        "candidate_id": "c65",
        "created_at": timestamp(),
        "stage": "label_free_counterfactual_state_G0",
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "decision": "authorize_execution_lock_and_exposed_train_labels"
        if all(checks.values())
        else "close_c65_before_train_labels",
        "physical_gpu": physical,
        "g0_lock_sha256": g0_lock_hash,
        "parameter_counts": parameter_counts,
        "gradients": gradients,
        "diagnostics": {
            "candidate_permutation_max_abs": permutation_error,
            "deterministic_max_abs": deterministic_error,
            "nohistory_max_abs": nohistory_error,
            "repeat_max_abs": repeat_error,
            "state_residual_rms": residual_rms,
            "wrong_state_residual_rms": wrong_residual_rms,
            "true_wrong_score_max_abs": true_wrong_score_difference,
        },
        "checks": checks,
        "train_validation_fresh_dev_test_qrels_labels_opened": False,
    }
    target = REPO_ROOT / config["paths"]["artifact_root"] / "g0_report.json"
    atomic_json(target, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/train_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(run(config, torch.device("cuda:0")), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
