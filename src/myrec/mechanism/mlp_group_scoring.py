"""Frozen-sample D4 SwiGLU group localization for native Q2/Q3 scoring."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.baselines.motivation_v12_ranker import _answer_target_tokens
from myrec.mechanism.attention_edge_scoring import _aggregate_paths, _path_scores
from myrec.mechanism.mlp_group_interventions import (
    MLP_GROUPS,
    MLP_GROUP_SEED,
    QwenMLPGroupCapture,
    QwenMLPGroupPatch,
    exact_permutation_recomposition,
    frozen_mlp_groups,
)
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.representation_probe import instrument_pointwise_prompt
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    resolve_qwen_backbone,
)


BRANCH_NODE_IDS = (
    "block_input_residual",
    "attention_o_projection",
    "mlp_down_projection",
    "block_output_residual",
)


def score_mlp_group_sample_row(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidate: Mapping[str, Any],
    donor_record: ModelRecord,
    donor_candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Localize all 16 fixed groups for one qrels-blind candidate row."""

    full_paths = build_native_pointwise_paths(
        tokenizer, record, [candidate], record.history, config, device=device
    )
    null_paths = build_native_pointwise_paths(
        tokenizer, record, [candidate], [], config, device=device
    )
    cross_paths = build_native_pointwise_paths(
        tokenizer,
        donor_record,
        [donor_candidate],
        donor_record.history,
        config,
        device=device,
    )
    full = capture_mlp_paths(model, full_paths, block=block)
    null = capture_mlp_paths(model, null_paths, block=block)
    cross = capture_mlp_paths(model, cross_paths, block=block)
    baseline_full = float(_aggregate_paths(full_paths, full["path_scores"])[0])
    baseline_null = float(_aggregate_paths(null_paths, null["path_scores"])[0])
    group_rows = []
    maximum_same_identity_delta = 0.0
    for group_id in range(MLP_GROUPS):
        same_values = patch_mlp_paths(
            model,
            null_paths,
            full["products"],
            block=block,
            group_id=group_id,
        )
        cross_values = patch_mlp_paths(
            model,
            null_paths,
            cross["products"],
            block=block,
            group_id=group_id,
        )
        identity_values = patch_mlp_paths(
            model,
            full_paths,
            full["products"],
            block=block,
            group_id=group_id,
        )
        same_score = float(_aggregate_paths(null_paths, same_values)[0])
        cross_score = float(_aggregate_paths(null_paths, cross_values)[0])
        identity_score = float(_aggregate_paths(full_paths, identity_values)[0])
        maximum_same_identity_delta = max(
            maximum_same_identity_delta, abs(identity_score - baseline_full)
        )
        group_rows.append(
            {
                "group_id": group_id,
                "same_full_to_null_score": same_score,
                "cross_full_to_null_score": cross_score,
                "same_minus_null": same_score - baseline_null,
                "cross_minus_null": cross_score - baseline_null,
                "same_minus_cross": same_score - cross_score,
                **_group_activation_summary(
                    full["products"], null["products"], group_id
                ),
            }
        )
    permutation = _permutation_control(model, block, full["products"])
    result = {
        "baseline_full": baseline_full,
        "baseline_null": baseline_null,
        "groups": group_rows,
        "maximum_same_group_identity_delta": maximum_same_identity_delta,
        "permutation_recomposition_max_abs_error": permutation["maximum_abs_error"],
        "permutation_low_precision_max_ratio": permutation["maximum_bound_ratio"],
        "permutation_recomposition_dtype": permutation["recomposition_dtype"],
        "permutation_bound_reference_dtype": permutation["bound_reference_dtype"],
        "residual_geometry": {
            "full": _residual_geometry(full["nodes"]),
            "null": _residual_geometry(null["nodes"]),
            "cross": _residual_geometry(cross["nodes"]),
        },
    }
    if not _all_finite(result):
        raise FloatingPointError("D4 MLP sample result contains a non-finite value")
    return result


def build_native_pointwise_paths(
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
) -> list[dict[str, Any]]:
    method_id = str(config["method_id"])
    if method_id == "q3_tallrec_generalqwen":
        targets = (
            ("yes", _answer_target_tokens(tokenizer, "Yes"), 1.0),
            ("no", _answer_target_tokens(tokenizer, "No"), -1.0),
        )
        reserve = max(len(target) for _name, target, _weight in targets)
    elif method_id == "q2_recranker_generalqwen":
        targets = (("prompt", [], 1.0),)
        reserve = 0
    else:
        raise ValueError("D4 native paths support only Q2/Q3")
    training = config["training"]
    prompts = [
        instrument_pointwise_prompt(
            tokenizer,
            method_id,
            record,
            candidate,
            history=history,
            history_budget=int(training["history_budget"]),
            max_length=int(training["max_length"]) - reserve,
        )
        for candidate in candidates
    ]
    paths = []
    for name, target, weight in targets:
        ids, mask, padding = _left_pad_sequences(
            [list(prompt.token_ids) + list(target) for prompt in prompts],
            tokenizer.pad_token_id,
            device,
        )
        positions = _torch().tensor(
            [
                [
                    left + prompt.candidate_readout + offset
                    for offset in range(len(target) if target else 1)
                ]
                for left, prompt in zip(padding, prompts)
            ],
            dtype=_torch().long,
            device=device,
        )
        paths.append(
            {
                "name": name,
                "target": list(target),
                "weight": float(weight),
                "ids": ids,
                "mask": mask,
                "positions": positions,
            }
        )
    return paths


def capture_mlp_paths(
    model: Any, paths: Sequence[Mapping[str, Any]], *, block: int
) -> dict[str, Any]:
    specs = tuple(NodeSpec(node_id=node_id, block=block) for node_id in BRANCH_NODE_IDS)
    path_scores = []
    products = []
    nodes = []
    with QwenMLPGroupCapture(model, block) as product_capture, QwenNodeCapture(
        model, specs
    ) as node_capture:
        for path in paths:
            positions = path["positions"]
            product_capture.arm(positions)
            node_capture.arm(positions, sequence_length=path["ids"].shape[1])
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            products.append(product_capture.disarm())
            nodes.append(node_capture.disarm())
            path_scores.append(_path_scores(output, path))
    return {"path_scores": path_scores, "products": products, "nodes": nodes}


def patch_mlp_paths(
    model: Any,
    paths: Sequence[Mapping[str, Any]],
    donor_products: Sequence[Any],
    *,
    block: int,
    group_id: int,
) -> list[np.ndarray]:
    if len(paths) != len(donor_products):
        raise ValueError("D4 paths and donor products differ")
    values = []
    with QwenMLPGroupPatch(model, block, [group_id]) as patch:
        for path, donor in zip(paths, donor_products):
            patch.arm(path["positions"], donor)
            output = model(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                use_cache=False,
                logits_to_keep=(len(path["target"]) + 1 if path["target"] else 1),
            )
            patch.disarm()
            values.append(_path_scores(output, path))
    return values


def _group_activation_summary(
    full_products: Sequence[Any], null_products: Sequence[Any], group_id: int
) -> dict[str, float]:
    torch = _torch()
    groups = frozen_mlp_groups(full_products[0].shape[-1])
    indices = torch.tensor(groups[group_id], dtype=torch.long, device=full_products[0].device)
    full = torch.cat([value.index_select(-1, indices).float().reshape(-1, len(indices)) for value in full_products])
    null = torch.cat([value.index_select(-1, indices).float().reshape(-1, len(indices)) for value in null_products])
    return {
        "full_rms": float(full.square().mean().sqrt().item()),
        "null_rms": float(null.square().mean().sqrt().item()),
        "full_hoyer_sparsity": _hoyer(full),
        "null_hoyer_sparsity": _hoyer(null),
        "full_null_cosine": float(
            torch.nn.functional.cosine_similarity(full, null, dim=-1).mean().item()
        ),
    }


def _hoyer(value: Any) -> float:
    flat = value.reshape(-1, value.shape[-1])
    n = flat.shape[-1]
    ratio = flat.abs().sum(-1) / flat.square().sum(-1).sqrt().clamp_min(1.0e-12)
    return float(((math.sqrt(n) - ratio) / (math.sqrt(n) - 1.0)).mean().item())


def _residual_geometry(path_nodes: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    torch = _torch()
    prefix = next(iter(path_nodes[0])).split(".")[0]
    def collect(node_id: str) -> Any:
        return torch.cat(
            [nodes[f"{prefix}.{node_id}"].float().reshape(-1, nodes[f"{prefix}.{node_id}"].shape[-1]) for nodes in path_nodes]
        )
    residual = collect("block_input_residual")
    attention = collect("attention_o_projection")
    mlp = collect("mlp_down_projection")
    output = collect("block_output_residual")
    recomposed = residual + attention + mlp
    return {
        "residual_norm": float(residual.norm(dim=-1).mean().item()),
        "attention_increment_norm": float(attention.norm(dim=-1).mean().item()),
        "mlp_increment_norm": float(mlp.norm(dim=-1).mean().item()),
        "block_output_norm": float(output.norm(dim=-1).mean().item()),
        "attention_mlp_cosine": float(torch.nn.functional.cosine_similarity(attention, mlp, dim=-1).mean().item()),
        "residual_attention_cosine": float(torch.nn.functional.cosine_similarity(residual, attention, dim=-1).mean().item()),
        "residual_mlp_cosine": float(torch.nn.functional.cosine_similarity(residual, mlp, dim=-1).mean().item()),
        "recomposition_max_abs_error": float((recomposed - output).abs().max().item()),
    }


def _permutation_control(
    model: Any, block: int, products: Sequence[Any]
) -> dict[str, float]:
    layer = resolve_qwen_backbone(model).layers[int(block)]
    intermediate = int(layer.mlp.down_proj.in_features)
    permutation = sorted(
        range(intermediate),
        key=lambda index: (
            hashlib.sha256(f"{MLP_GROUP_SEED}\0{index}".encode()).hexdigest(),
            index,
        ),
    )
    permutation_tensor = _torch().tensor(
        permutation, dtype=_torch().long, device=products[0].device
    )
    maximum = 0.0
    maximum_ratio = 0.0
    source_dtypes = {str(product.dtype).removeprefix("torch.") for product in products}
    if len(source_dtypes) != 1:
        raise ValueError("D4 MLP product dtypes differ across native paths")
    for product in products:
        original, recomposed, _ = exact_permutation_recomposition(
            product.float(), layer.mlp.down_proj.weight.float(), permutation_tensor
        )
        error = float((original - recomposed).abs().max().item())
        reference = float(original.abs().max().item())
        # Recompose in FP32 to minimize audit error, but use the dtype of the
        # native SwiGLU product for the frozen 4*eps tensor bound.  Using FP32
        # epsilon here tests GEMM reduction-order noise against a precision the
        # actual BF16/FP16 activation path never promises.
        bound = 4.0 * float(_torch().finfo(product.dtype).eps) * max(1.0, reference)
        maximum = max(maximum, error)
        maximum_ratio = max(maximum_ratio, error / bound)
    return {
        "maximum_abs_error": maximum,
        "maximum_bound_ratio": maximum_ratio,
        "recomposition_dtype": "float32",
        "bound_reference_dtype": next(iter(source_dtypes)),
    }


def _all_finite(value: Any) -> bool:
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _torch() -> Any:
    import torch

    return torch
