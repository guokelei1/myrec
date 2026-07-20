#!/usr/bin/env python3
"""Measure channel/head concentration of frozen Q2 parameter updates.

This extends the all-parameter magnitude audit with per-output, per-input,
attention-head, SwiGLU-intermediate, and norm-channel energy distributions for
all 28 Transformer blocks.  It is qrels-blind descriptive weight geometry and
does not select a component for intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from safetensors import safe_open


BASE_PATH = Path("models/huggingface/Qwen3-0.6B/model.safetensors")
FINAL_PATH = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q2_recranker_generalqwen_seed20260714/checkpoint_latest/model/model.safetensors"
)
PARAMETER_GEOMETRY_PATH = Path(
    "runs/20260718_kuaisearch_mech_d7_q2_parameter_update_geometry_v1/"
    "parameter_update_geometry.json"
)
BLOCK_FLOW_PATH = Path(
    "runs/20260718_kuaisearch_mech_d1_candidate_block_flow_v1/metrics.json"
)
BASE_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
FINAL_SHA256 = "83e3467dc26a02e65a0a49efabf08273ddb6dc7bcea7b06fe5bb0aaf2825f7c9"
PARAMETER_GEOMETRY_SHA256 = (
    "e9502da40e7473c34baccfba8fb4aa9876f3b739bfc338a8dada1a7bf3c578c3"
)
BLOCK_FLOW_SHA256 = "78220e91afc060af149d6d6ef9ca31ee3bbc067905c964741f4906c30f2d801e"
LAYERS = tuple(range(28))
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
SUMMARY_METRICS = (
    "normalized_participation_ratio",
    "normalized_entropy_effective_count",
    "top_1pct_energy_share",
    "top_5pct_energy_share",
    "top_10pct_energy_share",
    "maximum_to_mean_energy_ratio",
    "zero_energy_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d7_q2_update_anisotropy_v1",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    base_path = root / BASE_PATH
    final_path = root / FINAL_PATH
    parameter_geometry_path = root / PARAMETER_GEOMETRY_PATH
    block_flow_path = root / BLOCK_FLOW_PATH
    for path, expected, label in (
        (base_path, BASE_SHA256, "base weights"),
        (final_path, FINAL_SHA256, "Q2 weights"),
        (parameter_geometry_path, PARAMETER_GEOMETRY_SHA256, "parameter geometry"),
        (block_flow_path, BLOCK_FLOW_SHA256, "block flow"),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"Q2 {label} hash drift")
    parameter_geometry = _read_json(parameter_geometry_path)
    block_flow = _read_json(block_flow_path)
    if (
        parameter_geometry.get("status") != "completed"
        or parameter_geometry.get("dev_qrels_read") is not False
        or block_flow.get("status") != "completed"
        or block_flow.get("qrels_read") is not False
    ):
        raise ValueError("Q2 upstream analysis boundary differs")

    axis_rows: list[dict[str, Any]] = []
    semantic_rows: list[dict[str, Any]] = []
    semantic_energies: dict[tuple[int, str], np.ndarray] = {}
    with safe_open(base_path, framework="pt", device="cpu") as base_file, safe_open(
        final_path, framework="pt", device="cpu"
    ) as final_file:
        for key in sorted(final_file.keys()):
            classified = _classify_layer_key(key)
            if classified is None:
                continue
            layer, family = classified
            base = base_file.get_tensor(key).float()
            final = final_file.get_tensor(key).float()
            if tuple(base.shape) != tuple(final.shape):
                raise ValueError(f"Q2 tensor shape differs: {key}")
            update = final - base
            if update.ndim == 1:
                axis_rows.append(
                    {
                        "parameter_name": key,
                        "layer_zero_based": layer,
                        "region": _region_for_layer(layer),
                        "family": family,
                        "axis": "element",
                        **_distribution_metrics(update.double().square()),
                    }
                )
            elif update.ndim == 2:
                axis_rows.extend(
                    [
                        {
                            "parameter_name": key,
                            "layer_zero_based": layer,
                            "region": _region_for_layer(layer),
                            "family": family,
                            "axis": "output_row",
                            **_distribution_metrics(
                                update.double().square().sum(dim=1)
                            ),
                        },
                        {
                            "parameter_name": key,
                            "layer_zero_based": layer,
                            "region": _region_for_layer(layer),
                            "family": family,
                            "axis": "input_column",
                            **_distribution_metrics(
                                update.double().square().sum(dim=0)
                            ),
                        },
                    ]
                )
            else:
                raise ValueError(f"Q2 parameter rank is unsupported: {key}")
            group_kind, energy = _semantic_energy(family, update)
            energy_numpy = energy.detach().cpu().numpy().astype(np.float64, copy=False)
            semantic_energies[(layer, family)] = energy_numpy
            semantic_rows.append(
                {
                    "parameter_name": key,
                    "layer_zero_based": layer,
                    "region": _region_for_layer(layer),
                    "family": family,
                    "semantic_group": group_kind,
                    **_distribution_metrics(energy),
                }
            )

    if len(semantic_rows) != len(LAYERS) * len(SUFFIX_TO_FAMILY):
        raise ValueError("Q2 semantic family/layer coverage differs")
    region_family_rows = _aggregate_region_family(semantic_rows, semantic_energies)
    family_rows = _aggregate_family(semantic_rows, semantic_energies)
    layer_branch_rows = _aggregate_layer_branches(semantic_energies)
    correlations = _branch_flow_correlations(layer_branch_rows, block_flow)
    early_late = _early_late_contrasts(region_family_rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d7_q2_update_anisotropy",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_component_selector": False,
        "interpretation_boundary": (
            "Channel/head concentration describes the geometry of the frozen training "
            "update. It does not establish functional rank, utilization, or causal harm; "
            "BF16 base quantization also bounds small-update interpretation."
        ),
        "method_id": "q2_recranker_generalqwen",
        "checkpoint_id": parameter_geometry["checkpoint_id"],
        "sources": {
            "base_weights_path": BASE_PATH.as_posix(),
            "base_weights_sha256": BASE_SHA256,
            "final_weights_path": FINAL_PATH.as_posix(),
            "final_weights_sha256": FINAL_SHA256,
            "parameter_geometry_path": PARAMETER_GEOMETRY_PATH.as_posix(),
            "parameter_geometry_sha256": PARAMETER_GEOMETRY_SHA256,
            "block_flow_path": BLOCK_FLOW_PATH.as_posix(),
            "block_flow_sha256": BLOCK_FLOW_SHA256,
        },
        "semantic_group_definition": {
            "q_proj": "16 query heads x 128 output dimensions",
            "k_proj": "8 key/value heads x 128 output dimensions",
            "v_proj": "8 key/value heads x 128 output dimensions",
            "o_proj": "16 query-head input groups x 128 dimensions",
            "mlp_gate_proj": "3072 SwiGLU intermediate output channels",
            "mlp_up_proj": "3072 SwiGLU intermediate output channels",
            "mlp_down_proj": "3072 SwiGLU intermediate input channels",
            "norms": "individual norm channels",
        },
        "dev_qrels_read": False,
        "confirmation_qrels_read": False,
        "test_qrels_read": False,
        "source_test_opened": False,
        "axis_rows": axis_rows,
        "semantic_rows": semantic_rows,
        "region_family_rows": region_family_rows,
        "family_rows": family_rows,
        "layer_branch_rows": layer_branch_rows,
        "early_late_contrasts": early_late,
        "block_flow_correlations": correlations,
        "command": " ".join(os.sys.argv),
    }
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "update_anisotropy.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "axis_rows": len(axis_rows),
                "semantic_rows": len(semantic_rows),
                "output": str(output_path),
                "sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _classify_layer_key(key: str) -> tuple[int, str] | None:
    match = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", key)
    if match is None:
        if key in {"model.embed_tokens.weight", "model.norm.weight"}:
            return None
        raise ValueError(f"unclassified Q2 parameter: {key}")
    layer = int(match.group(1))
    suffix = match.group(2)
    if layer not in LAYERS or suffix not in SUFFIX_TO_FAMILY:
        raise ValueError(f"unclassified Q2 layer parameter: {key}")
    return layer, SUFFIX_TO_FAMILY[suffix]


def _semantic_energy(family: str, update: torch.Tensor) -> tuple[str, torch.Tensor]:
    squared = update.double().square()
    if family == "q_proj":
        if tuple(update.shape) != (2048, 1024):
            raise ValueError("Q2 q_proj shape differs")
        return "query_attention_head", squared.reshape(16, 128, 1024).sum(dim=(1, 2))
    if family in {"k_proj", "v_proj"}:
        if tuple(update.shape) != (1024, 1024):
            raise ValueError(f"Q2 {family} shape differs")
        return "key_value_attention_head", squared.reshape(8, 128, 1024).sum(dim=(1, 2))
    if family == "o_proj":
        if tuple(update.shape) != (1024, 2048):
            raise ValueError("Q2 o_proj shape differs")
        return "query_attention_head", squared.reshape(1024, 16, 128).sum(dim=(0, 2))
    if family in {"mlp_gate_proj", "mlp_up_proj"}:
        if tuple(update.shape) != (3072, 1024):
            raise ValueError(f"Q2 {family} shape differs")
        return "swiglu_intermediate_channel", squared.sum(dim=1)
    if family == "mlp_down_proj":
        if tuple(update.shape) != (1024, 3072):
            raise ValueError("Q2 mlp_down_proj shape differs")
        return "swiglu_intermediate_channel", squared.sum(dim=0)
    if family in {"input_rmsnorm", "post_attention_rmsnorm", "q_norm", "k_norm"}:
        if update.ndim != 1:
            raise ValueError(f"Q2 {family} rank differs")
        return "norm_channel", squared
    raise ValueError(f"Q2 semantic family is unsupported: {family}")


def _distribution_metrics(energy: torch.Tensor | np.ndarray) -> dict[str, Any]:
    values = np.asarray(
        energy.detach().cpu().numpy() if isinstance(energy, torch.Tensor) else energy,
        dtype=np.float64,
    ).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("update energy distribution is invalid")
    total = float(math.fsum(float(value) for value in values))
    if total <= 0.0:
        return {
            "groups": len(values),
            "total_update_energy": 0.0,
            "normalized_participation_ratio": None,
            "normalized_entropy_effective_count": None,
            "top_1pct_energy_share": None,
            "top_5pct_energy_share": None,
            "top_10pct_energy_share": None,
            "maximum_to_mean_energy_ratio": None,
            "zero_energy_fraction": 1.0,
        }
    probabilities = values / total
    positive = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return {
        "groups": len(values),
        "total_update_energy": total,
        "normalized_participation_ratio": float(
            1.0 / (len(values) * np.sum(probabilities**2))
        ),
        "normalized_entropy_effective_count": float(math.exp(entropy) / len(values)),
        "top_1pct_energy_share": _top_share(probabilities, 0.01),
        "top_5pct_energy_share": _top_share(probabilities, 0.05),
        "top_10pct_energy_share": _top_share(probabilities, 0.10),
        "maximum_to_mean_energy_ratio": float(len(values) * np.max(probabilities)),
        "zero_energy_fraction": float(np.mean(values == 0.0)),
    }


def _top_share(probabilities: np.ndarray, fraction: float) -> float:
    count = max(1, int(math.ceil(len(probabilities) * fraction)))
    return float(np.sum(np.partition(probabilities, len(probabilities) - count)[-count:]))


def _aggregate_region_family(
    semantic_rows: Sequence[Mapping[str, Any]],
    energies: Mapping[tuple[int, str], np.ndarray],
) -> list[dict[str, Any]]:
    result = []
    for region, layers in REGIONS.items():
        for family in SUFFIX_TO_FAMILY.values():
            selected = [
                row
                for row in semantic_rows
                if row["region"] == region and row["family"] == family
            ]
            if len(selected) != len(layers):
                raise ValueError("Q2 region/family layer coverage differs")
            pooled = np.concatenate([energies[(layer, family)] for layer in layers])
            row = {
                "region": region,
                "layer_zero_based_indices": list(layers),
                "family": family,
                "semantic_group": selected[0]["semantic_group"],
                **{f"pooled_{key}": value for key, value in _distribution_metrics(pooled).items()},
            }
            for metric in SUMMARY_METRICS:
                row[f"mean_layer_{metric}"] = _mean_optional(
                    [value[metric] for value in selected]
                )
            result.append(row)
    return result


def _aggregate_family(
    semantic_rows: Sequence[Mapping[str, Any]],
    energies: Mapping[tuple[int, str], np.ndarray],
) -> list[dict[str, Any]]:
    result = []
    for family in SUFFIX_TO_FAMILY.values():
        selected = [row for row in semantic_rows if row["family"] == family]
        pooled = np.concatenate([energies[(layer, family)] for layer in LAYERS])
        row = {
            "family": family,
            "semantic_group": selected[0]["semantic_group"],
            **{f"pooled_{key}": value for key, value in _distribution_metrics(pooled).items()},
        }
        for metric in SUMMARY_METRICS:
            row[f"mean_layer_{metric}"] = _mean_optional(
                [value[metric] for value in selected]
            )
        result.append(row)
    return result


def _aggregate_layer_branches(
    energies: Mapping[tuple[int, str], np.ndarray],
) -> list[dict[str, Any]]:
    branch_families = {
        "attention_heads": ("q_proj", "k_proj", "v_proj", "o_proj"),
        "mlp_intermediate_channels": (
            "mlp_gate_proj",
            "mlp_up_proj",
            "mlp_down_proj",
        ),
        "norm_channels": (
            "input_rmsnorm",
            "post_attention_rmsnorm",
            "q_norm",
            "k_norm",
        ),
    }
    rows = []
    for layer in LAYERS:
        for branch, families in branch_families.items():
            pooled = np.concatenate([energies[(layer, family)] for family in families])
            rows.append(
                {
                    "layer_zero_based": layer,
                    "region": _region_for_layer(layer),
                    "branch": branch,
                    "families": list(families),
                    **_distribution_metrics(pooled),
                }
            )
    return rows


def _early_late_contrasts(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for family in SUFFIX_TO_FAMILY.values():
        early = next(
            row for row in rows if row["family"] == family and row["region"] == "blocks_00_06"
        )
        late = next(
            row for row in rows if row["family"] == family and row["region"] == "blocks_21_27"
        )
        result.append(
            {
                "family": family,
                "semantic_group": early["semantic_group"],
                "early_region": "blocks_00_06",
                "late_region": "blocks_21_27",
                "early_mean_layer_normalized_participation_ratio": early[
                    "mean_layer_normalized_participation_ratio"
                ],
                "late_mean_layer_normalized_participation_ratio": late[
                    "mean_layer_normalized_participation_ratio"
                ],
                "late_minus_early_mean_layer_normalized_participation_ratio": (
                    late["mean_layer_normalized_participation_ratio"]
                    - early["mean_layer_normalized_participation_ratio"]
                ),
                "late_over_early_mean_layer_normalized_participation_ratio": (
                    late["mean_layer_normalized_participation_ratio"]
                    / early["mean_layer_normalized_participation_ratio"]
                ),
            }
        )
    return result


def _branch_flow_correlations(
    branch_rows: Sequence[Mapping[str, Any]], block_flow: Mapping[str, Any]
) -> dict[str, Any]:
    flow_rows = sorted(
        (
            row
            for row in block_flow["block_rows"]
            if row["model_key"] == "q2" and row["normalized_query_fold"] == "all"
        ),
        key=lambda row: int(row["block_zero_based"]),
    )
    flow = [float(row["mean_update_common_energy_fraction"]) for row in flow_rows]
    result = {}
    for branch in ("attention_heads", "mlp_intermediate_channels", "norm_channels"):
        selected = sorted(
            (row for row in branch_rows if row["branch"] == branch),
            key=lambda row: int(row["layer_zero_based"]),
        )
        values = [float(row["normalized_participation_ratio"]) for row in selected]
        result[branch] = _pearson(values, flow)
    return {
        "descriptive_only": True,
        "flow_metric": "mean_update_common_energy_fraction",
        "weight_metric": "per-layer pooled normalized_participation_ratio",
        "pearson": result,
    }


def _region_for_layer(layer: int) -> str:
    for region, layers in REGIONS.items():
        if layer in layers:
            return region
    raise ValueError(f"layer outside registered regions: {layer}")


def _mean_optional(values: Sequence[Any]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not finite else math.fsum(finite) / len(finite)


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
        x * y for x, y in zip(left_centered, right_centered)
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
