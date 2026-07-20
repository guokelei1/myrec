#!/usr/bin/env python3
"""Describe Q2 full-parameter adaptation across Transformer components.

The audit compares the frozen Q2 checkpoint with its declared Qwen3-0.6B
base, tensor by tensor.  It reports exact Frobenius update energy, per-parameter
RMS, alignment, and concentration for every block and Q/K/V/O, norm, and
SwiGLU family.  It reads training provenance but no dev qrels or source test.

This is descriptive weight geometry, not evidence that a large update is
harmful or that a small update is capacity-limited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open


BASE_PATH = Path("models/huggingface/Qwen3-0.6B/model.safetensors")
FINAL_PATH = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q2_recranker_generalqwen_seed20260714/checkpoint_latest/model/model.safetensors"
)
TRAINING_METADATA_PATH = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q2_recranker_generalqwen_seed20260714/training_metadata.json"
)
CONFIG_PATH = Path("configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml")
BLOCK_FLOW_PATH = Path(
    "runs/20260718_kuaisearch_mech_d1_candidate_block_flow_v1/metrics.json"
)
BASE_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
FINAL_SHA256 = "83e3467dc26a02e65a0a49efabf08273ddb6dc7bcea7b06fe5bb0aaf2825f7c9"
BLOCK_FLOW_SHA256 = "78220e91afc060af149d6d6ef9ca31ee3bbc067905c964741f4906c30f2d801e"
LAYERS = tuple(range(28))
FAMILIES = (
    "input_rmsnorm",
    "post_attention_rmsnorm",
    "q_norm",
    "k_norm",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "mlp_gate_proj",
    "mlp_up_proj",
    "mlp_down_proj",
)
REGIONS = {
    "blocks_00_06": tuple(range(0, 7)),
    "blocks_07_13": tuple(range(7, 14)),
    "blocks_14_20": tuple(range(14, 21)),
    "blocks_21_27": tuple(range(21, 28)),
}
SUFFIX_TO_FAMILY = {
    "input_layernorm.weight": "input_rmsnorm",
    "post_attention_layernorm.weight": "post_attention_rmsnorm",
    "self_attn.q_norm.weight": "q_norm",
    "self_attn.k_norm.weight": "k_norm",
    "self_attn.q_proj.weight": "q_proj",
    "self_attn.k_proj.weight": "k_proj",
    "self_attn.v_proj.weight": "v_proj",
    "self_attn.o_proj.weight": "o_proj",
    "mlp.gate_proj.weight": "mlp_gate_proj",
    "mlp.up_proj.weight": "mlp_up_proj",
    "mlp.down_proj.weight": "mlp_down_proj",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d7_q2_parameter_update_geometry_v1",
    )
    parser.add_argument("--chunk-elements", type=int, default=4_000_000)
    args = parser.parse_args()
    if args.chunk_elements <= 0:
        raise ValueError("chunk-elements must be positive")
    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    base_path = root / BASE_PATH
    final_path = root / FINAL_PATH
    if _sha256_file(base_path) != BASE_SHA256:
        raise ValueError("Q2 declared base weights hash drift")
    if _sha256_file(final_path) != FINAL_SHA256:
        raise ValueError("Q2 frozen checkpoint weights hash drift")

    training_metadata_path = root / TRAINING_METADATA_PATH
    training_metadata = _read_json(training_metadata_path)
    expected_training = {
        "status": "completed",
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": "q2_recranker_generalqwen@e207d2213741c16f997a",
        "base_weights_sha256": BASE_SHA256,
        "qrels_read": True,
        "resume_state_complete": True,
    }
    for key, value in expected_training.items():
        if training_metadata.get(key) != value:
            raise ValueError(f"Q2 training metadata differs: {key}")

    block_flow_path = root / BLOCK_FLOW_PATH
    if _sha256_file(block_flow_path) != BLOCK_FLOW_SHA256:
        raise ValueError("Q2 block-flow binding hash drift")
    block_flow = _read_json(block_flow_path)
    if (
        block_flow.get("status") != "completed"
        or block_flow.get("qrels_read") is not False
        or block_flow.get("source_test_opened") is not False
    ):
        raise ValueError("Q2 block-flow safety boundary differs")

    tensor_rows: list[dict[str, Any]] = []
    layer_accumulators = {layer: _empty_accumulator() for layer in LAYERS}
    family_accumulators = {family: _empty_accumulator() for family in FAMILIES}
    region_accumulators = {region: _empty_accumulator() for region in REGIONS}
    region_family_accumulators = {
        (region, family): _empty_accumulator()
        for region in REGIONS
        for family in FAMILIES
    }
    non_layer_accumulators = {
        "tied_embedding_readout": _empty_accumulator(),
        "final_rmsnorm": _empty_accumulator(),
    }
    transformer_accumulator = _empty_accumulator()
    global_accumulator = _empty_accumulator()

    with safe_open(base_path, framework="pt", device="cpu") as base_file, safe_open(
        final_path, framework="pt", device="cpu"
    ) as final_file:
        base_keys = set(base_file.keys())
        final_keys = set(final_file.keys())
        if base_keys - final_keys != {"lm_head.weight"} or final_keys - base_keys:
            raise ValueError("Q2 base/final parameter key boundary differs")
        for key in sorted(final_keys):
            layer, family = _classify_key(key)
            base = base_file.get_tensor(key)
            final = final_file.get_tensor(key)
            if tuple(base.shape) != tuple(final.shape):
                raise ValueError(f"Q2 tensor shape differs: {key}")
            stats = _tensor_statistics(base, final, args.chunk_elements)
            _add_accumulator(global_accumulator, stats)
            if layer is None:
                _add_accumulator(non_layer_accumulators[family], stats)
            else:
                _add_accumulator(transformer_accumulator, stats)
                _add_accumulator(layer_accumulators[layer], stats)
                _add_accumulator(family_accumulators[family], stats)
                region = _region_for_layer(layer)
                _add_accumulator(region_accumulators[region], stats)
                _add_accumulator(region_family_accumulators[(region, family)], stats)
            tensor_rows.append(
                {
                    "parameter_name": key,
                    "layer_zero_based": layer,
                    "family": family,
                    "shape": list(final.shape),
                    **_finalize_accumulator(stats),
                }
            )

    global_summary = _finalize_accumulator(global_accumulator)
    transformer_summary = _finalize_accumulator(transformer_accumulator)
    layer_rows = [
        {
            "layer_zero_based": layer,
            "region": _region_for_layer(layer),
            **_finalize_accumulator(
                layer_accumulators[layer], transformer_accumulator["update_sq_sum"]
            ),
        }
        for layer in LAYERS
    ]
    family_rows = [
        {
            "family": family,
            **_finalize_accumulator(
                family_accumulators[family], transformer_accumulator["update_sq_sum"]
            ),
        }
        for family in FAMILIES
    ]
    region_rows = [
        {
            "region": region,
            "layer_zero_based_indices": list(layers),
            **_finalize_accumulator(
                region_accumulators[region], transformer_accumulator["update_sq_sum"]
            ),
        }
        for region, layers in REGIONS.items()
    ]
    region_family_rows = [
        {
            "region": region,
            "layer_zero_based_indices": list(REGIONS[region]),
            "family": family,
            **_finalize_accumulator(
                region_family_accumulators[(region, family)],
                transformer_accumulator["update_sq_sum"],
            ),
        }
        for region in REGIONS
        for family in FAMILIES
    ]
    non_layer_rows = [
        {
            "family": family,
            **_finalize_accumulator(accumulator, global_accumulator["update_sq_sum"]),
        }
        for family, accumulator in non_layer_accumulators.items()
    ]
    correlations = _block_flow_correlations(layer_rows, block_flow)
    concentration = _layer_concentration(layer_rows, region_rows)

    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d7_q2_parameter_update_geometry",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_component_selector": False,
        "interpretation_boundary": (
            "Parameter delta magnitude measures where full-parameter training moved "
            "the checkpoint, not whether that movement caused transfer benefit or harm. "
            "BF16 base weights also limit sub-quantization interpretation."
        ),
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": training_metadata["checkpoint_id"],
        "adaptation": "full_parameter",
        "tied_embedding_readout": True,
        "base_lm_head_omitted_from_final_due_tied_weights": True,
        "sources": {
            "base_weights_path": BASE_PATH.as_posix(),
            "base_weights_sha256": BASE_SHA256,
            "final_weights_path": FINAL_PATH.as_posix(),
            "final_weights_sha256": FINAL_SHA256,
            "training_metadata_path": TRAINING_METADATA_PATH.as_posix(),
            "training_metadata_sha256": _sha256_file(training_metadata_path),
            "config_path": CONFIG_PATH.as_posix(),
            "config_sha256": _sha256_file(root / CONFIG_PATH),
            "block_flow_path": BLOCK_FLOW_PATH.as_posix(),
            "block_flow_sha256": BLOCK_FLOW_SHA256,
        },
        "training_supervision_read_by_original_training": True,
        "dev_qrels_read": False,
        "confirmation_qrels_read": False,
        "test_qrels_read": False,
        "source_test_opened": False,
        "statistics_definition": (
            "All update norms are exact tensor-wise sums after promoting the BF16 base "
            "and FP32 final slices to FP32; group RMS is sqrt(sum(delta^2)/parameter_count)."
        ),
        "global_summary": global_summary,
        "transformer_summary": transformer_summary,
        "layer_concentration": concentration,
        "block_flow_correlations": correlations,
        "tensor_rows": tensor_rows,
        "layer_rows": layer_rows,
        "family_rows": family_rows,
        "region_rows": region_rows,
        "region_family_rows": region_family_rows,
        "non_layer_rows": non_layer_rows,
        "command": " ".join(os.sys.argv),
    }
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "parameter_update_geometry.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "tensors": len(tensor_rows),
                "parameters": global_summary["parameter_count"],
                "output": str(output_path),
                "sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _classify_key(key: str) -> tuple[int | None, str]:
    if key == "model.embed_tokens.weight":
        return None, "tied_embedding_readout"
    if key == "model.norm.weight":
        return None, "final_rmsnorm"
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
    if match is None:
        raise ValueError(f"unclassified Q2 parameter: {key}")
    layer = int(match.group(1))
    suffix = match.group(2)
    if layer not in LAYERS or suffix not in SUFFIX_TO_FAMILY:
        raise ValueError(f"unclassified Q2 layer parameter: {key}")
    return layer, SUFFIX_TO_FAMILY[suffix]


def _tensor_statistics(
    base: torch.Tensor, final: torch.Tensor, chunk_elements: int
) -> dict[str, Any]:
    if base.numel() != final.numel():
        raise ValueError("base/final tensor sizes differ")
    accumulator = _empty_accumulator()
    base_flat = base.reshape(-1)
    final_flat = final.reshape(-1)
    for start in range(0, base.numel(), chunk_elements):
        end = min(start + chunk_elements, base.numel())
        base_chunk = base_flat[start:end].float()
        final_chunk = final_flat[start:end].float()
        update = final_chunk - base_chunk
        accumulator["parameter_count"] += end - start
        accumulator["base_sq_sum"] += float(torch.sum(base_chunk.double() ** 2).item())
        accumulator["final_sq_sum"] += float(torch.sum(final_chunk.double() ** 2).item())
        accumulator["update_sq_sum"] += float(torch.sum(update.double() ** 2).item())
        accumulator["base_final_dot_sum"] += float(
            torch.sum(base_chunk.double() * final_chunk.double()).item()
        )
        accumulator["base_update_dot_sum"] += float(
            torch.sum(base_chunk.double() * update.double()).item()
        )
        accumulator["update_abs_sum"] += float(torch.sum(update.double().abs()).item())
        accumulator["exact_zero_count"] += int(torch.count_nonzero(update == 0).item())
        accumulator["maximum_absolute_update"] = max(
            accumulator["maximum_absolute_update"],
            float(torch.max(update.abs()).item()) if update.numel() else 0.0,
        )
    return accumulator


def _empty_accumulator() -> dict[str, Any]:
    return {
        "parameter_count": 0,
        "base_sq_sum": 0.0,
        "final_sq_sum": 0.0,
        "update_sq_sum": 0.0,
        "base_final_dot_sum": 0.0,
        "base_update_dot_sum": 0.0,
        "update_abs_sum": 0.0,
        "exact_zero_count": 0,
        "maximum_absolute_update": 0.0,
    }


def _add_accumulator(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key in (
        "parameter_count",
        "base_sq_sum",
        "final_sq_sum",
        "update_sq_sum",
        "base_final_dot_sum",
        "base_update_dot_sum",
        "update_abs_sum",
        "exact_zero_count",
    ):
        target[key] += source[key]
    target["maximum_absolute_update"] = max(
        target["maximum_absolute_update"], source["maximum_absolute_update"]
    )


def _finalize_accumulator(
    accumulator: Mapping[str, Any], update_energy_denominator: float | None = None
) -> dict[str, Any]:
    count = int(accumulator["parameter_count"])
    if count <= 0:
        raise ValueError("cannot finalize an empty parameter accumulator")
    base_sq = float(accumulator["base_sq_sum"])
    final_sq = float(accumulator["final_sq_sum"])
    update_sq = float(accumulator["update_sq_sum"])
    base_update_denominator = math.sqrt(base_sq * update_sq)
    base_final_denominator = math.sqrt(base_sq * final_sq)
    row = {
        "parameter_count": count,
        "base_frobenius": math.sqrt(base_sq),
        "final_frobenius": math.sqrt(final_sq),
        "update_frobenius": math.sqrt(update_sq),
        "base_rms": math.sqrt(base_sq / count),
        "final_rms": math.sqrt(final_sq / count),
        "update_rms": math.sqrt(update_sq / count),
        "relative_update_frobenius": (
            None if base_sq <= 0.0 else math.sqrt(update_sq / base_sq)
        ),
        "mean_absolute_update": float(accumulator["update_abs_sum"]) / count,
        "maximum_absolute_update": float(accumulator["maximum_absolute_update"]),
        "exact_zero_fraction": int(accumulator["exact_zero_count"]) / count,
        "base_update_cosine": (
            None
            if base_update_denominator <= 0.0
            else float(accumulator["base_update_dot_sum"]) / base_update_denominator
        ),
        "base_final_cosine": (
            None
            if base_final_denominator <= 0.0
            else float(accumulator["base_final_dot_sum"]) / base_final_denominator
        ),
    }
    if update_energy_denominator is not None:
        row["update_energy_share"] = update_sq / update_energy_denominator
    return row


def _region_for_layer(layer: int) -> str:
    for region, layers in REGIONS.items():
        if layer in layers:
            return region
    raise ValueError(f"layer outside registered regions: {layer}")


def _block_flow_correlations(
    layer_rows: Sequence[Mapping[str, Any]], block_flow: Mapping[str, Any]
) -> dict[str, Any]:
    parameter_rms = [float(row["update_rms"]) for row in layer_rows]
    flow_rows = sorted(
        (
            row
            for row in block_flow["block_rows"]
            if row["model_key"] == "q2" and row["normalized_query_fold"] == "all"
        ),
        key=lambda row: int(row["block_zero_based"]),
    )
    if len(flow_rows) != len(LAYERS):
        raise ValueError("Q2 block-flow layer coverage differs")
    metrics = {
        "mean_output_common_energy_fraction": [
            float(row["mean_output_common_energy_fraction"]) for row in flow_rows
        ],
        "mean_update_common_energy_fraction": [
            float(row["mean_update_common_energy_fraction"]) for row in flow_rows
        ],
        "mean_common_energy_change": [
            float(row["mean_common_energy_change"]) for row in flow_rows
        ],
        "mean_candidate_relative_energy_change": [
            float(row["mean_candidate_relative_energy_change"]) for row in flow_rows
        ],
    }
    return {
        "descriptive_only": True,
        "layers": len(LAYERS),
        "parameter_metric": "per-layer update_rms",
        "pearson": {
            name: _pearson(parameter_rms, values) for name, values in metrics.items()
        },
    }


def _layer_concentration(
    layer_rows: Sequence[Mapping[str, Any]], region_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    rms = [float(row["update_rms"]) for row in layer_rows]
    mean = math.fsum(rms) / len(rms)
    variance = math.fsum((value - mean) ** 2 for value in rms) / len(rms)
    late = next(row for row in region_rows if row["region"] == "blocks_21_27")
    return {
        "minimum_layer_update_rms": min(rms),
        "maximum_layer_update_rms": max(rms),
        "max_to_min_layer_update_rms_ratio": max(rms) / min(rms),
        "layer_update_rms_coefficient_of_variation": math.sqrt(variance) / mean,
        "late_region_transformer_update_energy_share": late["update_energy_share"],
    }


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs must have equal length >=2")
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator <= 0.0:
        return None
    return math.fsum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered)
    ) / denominator


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
