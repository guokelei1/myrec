#!/usr/bin/env python3
"""Compare Q3 LoRA function updates across attention heads and depth.

The analysis reconstructs delta-W=2*B@A for every q/v LoRA path at step 500,
the frozen final checkpoint, and their subsequent increment.  It measures
head and input-channel concentration without reading dev qrels and provides a
same-geometry contrast to the Q2 full-parameter head audit.
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

import numpy as np
import torch
from safetensors import safe_open


STEP500_PATH = Path(
    "artifacts/motivation_v1_2/resume_canary/q3_step500_seed20260714/"
    "checkpoint_latest/model/adapter_model.safetensors"
)
FINAL_PATH = Path(
    "artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714/"
    "checkpoint_latest/model/adapter_model.safetensors"
)
ADAPTER_CONFIG_PATH = Path(
    "artifacts/motivation_v1_2/checkpoints/q3_tallrec_generalqwen_seed20260714/"
    "checkpoint_latest/model/adapter_config.json"
)
LORA_PATH_ANALYSIS_PATH = Path(
    "runs/20260718_kuaisearch_mech_d7_q3_lora_path_v1/lora_path_analysis.json"
)
Q2_REFERENCE_PATH = Path(
    "runs/20260718_kuaisearch_mech_d7_q2_update_anisotropy_v1/update_anisotropy.json"
)
BLOCK_FLOW_PATH = Path(
    "runs/20260718_kuaisearch_mech_d1_candidate_block_flow_v1/metrics.json"
)
STEP500_SHA256 = "d68e9cd48e89100e512cb98023746fadb5701c91ec10e20e306d215550d1f86e"
FINAL_SHA256 = "fd51a9c6b9ee3a6651597c263a8120db52cb79d62e7c80e544666e46bc5e1cef"
LORA_PATH_ANALYSIS_SHA256 = (
    "5a016c841a85fc7bf08209053af4aad7569ad6a25a586b1aa01d150d6459b791"
)
Q2_REFERENCE_SHA256 = "0df0390970355476d8c8d680500524b96581e3210b519631c5bd95bba9110314"
BLOCK_FLOW_SHA256 = "78220e91afc060af149d6d6ef9ca31ee3bbc067905c964741f4906c30f2d801e"
LAYERS = tuple(range(28))
PROJECTIONS = ("q", "v")
STATES = ("step_500", "frozen_final_checkpoint", "final_minus_step500")
REGIONS = {
    "blocks_00_06": tuple(range(0, 7)),
    "blocks_07_13": tuple(range(7, 14)),
    "blocks_14_20": tuple(range(14, 21)),
    "blocks_21_27": tuple(range(21, 28)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d7_q3_lora_head_geometry_v1",
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    paths = {
        "step_500": root / STEP500_PATH,
        "frozen_final_checkpoint": root / FINAL_PATH,
    }
    for path, expected, label in (
        (paths["step_500"], STEP500_SHA256, "step-500 adapter"),
        (paths["frozen_final_checkpoint"], FINAL_SHA256, "final adapter"),
        (root / LORA_PATH_ANALYSIS_PATH, LORA_PATH_ANALYSIS_SHA256, "LoRA path analysis"),
        (root / Q2_REFERENCE_PATH, Q2_REFERENCE_SHA256, "Q2 anisotropy reference"),
        (root / BLOCK_FLOW_PATH, BLOCK_FLOW_SHA256, "block-flow reference"),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"Q3 {label} hash drift")
    adapter_config = _read_json(root / ADAPTER_CONFIG_PATH)
    if (
        adapter_config.get("r") != 8
        or adapter_config.get("lora_alpha") != 16
        or sorted(adapter_config.get("target_modules", [])) != ["q_proj", "v_proj"]
    ):
        raise ValueError("Q3 adapter scaling/target contract differs")
    lora_path = _read_json(root / LORA_PATH_ANALYSIS_PATH)
    q2_reference = _read_json(root / Q2_REFERENCE_PATH)
    block_flow = _read_json(root / BLOCK_FLOW_PATH)
    if (
        lora_path.get("status") != "completed"
        or lora_path.get("qrels_read") is not False
        or q2_reference.get("status") != "completed"
        or q2_reference.get("dev_qrels_read") is not False
        or block_flow.get("status") != "completed"
        or block_flow.get("qrels_read") is not False
    ):
        raise ValueError("Q3 upstream analysis safety boundary differs")

    functions = {
        state: _load_lora_functions(path, scaling=2.0)
        for state, path in paths.items()
    }
    functions["final_minus_step500"] = {
        key: functions["frozen_final_checkpoint"][key] - functions["step_500"][key]
        for key in functions["frozen_final_checkpoint"]
    }
    rows = []
    for state in STATES:
        for layer in LAYERS:
            for projection in PROJECTIONS:
                delta_w = functions[state][(layer, projection)]
                head_energy = _head_energy(delta_w, projection)
                input_energy = delta_w.double().square().sum(dim=0)
                rows.append(
                    {
                        "state": state,
                        "layer_zero_based": layer,
                        "region": _region_for_layer(layer),
                        "projection": projection,
                        "delta_w_frobenius": float(delta_w.double().norm().item()),
                        "output_head": _distribution_metrics(head_energy),
                        "input_channel": _distribution_metrics(input_energy),
                    }
                )
    region_rows = _build_region_rows(rows)
    early_late = _build_early_late(region_rows)
    q2_comparison = _build_q2_comparison(region_rows, q2_reference)
    correlations = _build_flow_correlations(rows, block_flow)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d7_q3_lora_head_geometry",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_head_selector": False,
        "interpretation_boundary": (
            "LoRA delta-W head concentration is gauge-invariant function geometry, "
            "but concentration alone does not prove head utilization, capacity collapse, "
            "or causal transfer harm."
        ),
        "method_id": "q3_tallrec_generalqwen",
        "checkpoint_id": "q3_tallrec_generalqwen@ea46a89671b63741ada8",
        "function_definition": "delta_w=(lora_alpha/r)*B@A=2*B@A",
        "sources": {
            "step500_adapter_path": STEP500_PATH.as_posix(),
            "step500_adapter_sha256": STEP500_SHA256,
            "final_adapter_path": FINAL_PATH.as_posix(),
            "final_adapter_sha256": FINAL_SHA256,
            "adapter_config_path": ADAPTER_CONFIG_PATH.as_posix(),
            "adapter_config_sha256": _sha256_file(root / ADAPTER_CONFIG_PATH),
            "lora_path_analysis_path": LORA_PATH_ANALYSIS_PATH.as_posix(),
            "lora_path_analysis_sha256": LORA_PATH_ANALYSIS_SHA256,
            "q2_reference_path": Q2_REFERENCE_PATH.as_posix(),
            "q2_reference_sha256": Q2_REFERENCE_SHA256,
            "block_flow_path": BLOCK_FLOW_PATH.as_posix(),
            "block_flow_sha256": BLOCK_FLOW_SHA256,
        },
        "dev_qrels_read": False,
        "confirmation_qrels_read": False,
        "test_qrels_read": False,
        "source_test_opened": False,
        "states": list(STATES),
        "projections": list(PROJECTIONS),
        "rows": rows,
        "region_rows": region_rows,
        "early_late_contrasts": early_late,
        "q2_same_geometry_comparison": q2_comparison,
        "block_flow_correlations": correlations,
        "command": " ".join(os.sys.argv),
    }
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lora_head_geometry.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "rows": len(rows),
                "output": str(output_path),
                "sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _load_lora_functions(
    path: Path, *, scaling: float
) -> dict[tuple[int, str], torch.Tensor]:
    factors: dict[tuple[int, str], dict[str, torch.Tensor]] = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        if len(keys) != 112:
            raise ValueError("Q3 adapter parameter count differs")
        for key in keys:
            layer, projection, factor = _parse_adapter_key(key)
            factors.setdefault((layer, projection), {})[factor] = handle.get_tensor(key).double()
    expected = {(layer, projection) for layer in LAYERS for projection in PROJECTIONS}
    if set(factors) != expected or any(set(pair) != {"A", "B"} for pair in factors.values()):
        raise ValueError("Q3 LoRA factor coverage differs")
    return {
        key: float(scaling) * pair["B"] @ pair["A"]
        for key, pair in factors.items()
    }


def _parse_adapter_key(key: str) -> tuple[int, str, str]:
    match = re.fullmatch(
        r"base_model\.model\.model\.layers\.(\d+)\.self_attn\."
        r"(q_proj|v_proj)\.lora_([AB])\.weight",
        key,
    )
    if match is None:
        raise ValueError(f"unrecognized Q3 adapter key: {key}")
    layer = int(match.group(1))
    if layer not in LAYERS:
        raise ValueError("Q3 adapter layer is outside model depth")
    return layer, match.group(2)[0], match.group(3)


def _head_energy(delta_w: torch.Tensor, projection: str) -> torch.Tensor:
    heads = 16 if projection == "q" else 8
    expected = (heads * 128, 1024)
    if tuple(delta_w.shape) != expected:
        raise ValueError(f"Q3 {projection} function shape differs")
    return delta_w.double().square().reshape(heads, 128, 1024).sum(dim=(1, 2))


def _distribution_metrics(energy: torch.Tensor | np.ndarray) -> dict[str, Any]:
    values = np.asarray(
        energy.detach().cpu().numpy() if isinstance(energy, torch.Tensor) else energy,
        dtype=np.float64,
    ).reshape(-1)
    if len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("Q3 LoRA energy distribution is invalid")
    total = float(math.fsum(float(value) for value in values))
    if total <= 0.0:
        raise ValueError("Q3 trained LoRA function has zero energy")
    probability = values / total
    positive = probability[probability > 0.0]
    entropy = -float(np.sum(positive * np.log(positive)))
    return {
        "groups": len(values),
        "total_update_energy": total,
        "normalized_participation_ratio": float(
            1.0 / (len(values) * np.sum(probability**2))
        ),
        "normalized_entropy_effective_count": float(math.exp(entropy) / len(values)),
        "top_group_energy_share": float(np.max(probability)),
        "maximum_to_mean_energy_ratio": float(len(values) * np.max(probability)),
        "zero_energy_fraction": float(np.mean(values == 0.0)),
    }


def _build_region_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        for projection in PROJECTIONS:
            for region, layers in REGIONS.items():
                selected = [
                    row
                    for row in rows
                    if row["state"] == state
                    and row["projection"] == projection
                    and row["layer_zero_based"] in layers
                ]
                result.append(
                    {
                        "state": state,
                        "projection": projection,
                        "region": region,
                        "layer_zero_based_indices": list(layers),
                        "mean_delta_w_frobenius": _mean(
                            [float(row["delta_w_frobenius"]) for row in selected]
                        ),
                        "mean_head_normalized_participation_ratio": _mean(
                            [
                                float(row["output_head"]["normalized_participation_ratio"])
                                for row in selected
                            ]
                        ),
                        "mean_head_normalized_entropy_effective_count": _mean(
                            [
                                float(row["output_head"]["normalized_entropy_effective_count"])
                                for row in selected
                            ]
                        ),
                        "mean_top_head_energy_share": _mean(
                            [float(row["output_head"]["top_group_energy_share"]) for row in selected]
                        ),
                        "mean_input_channel_normalized_participation_ratio": _mean(
                            [
                                float(row["input_channel"]["normalized_participation_ratio"])
                                for row in selected
                            ]
                        ),
                    }
                )
    return result


def _build_early_late(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for state in STATES:
        for projection in PROJECTIONS:
            early = next(
                row
                for row in rows
                if row["state"] == state
                and row["projection"] == projection
                and row["region"] == "blocks_00_06"
            )
            late = next(
                row
                for row in rows
                if row["state"] == state
                and row["projection"] == projection
                and row["region"] == "blocks_21_27"
            )
            result.append(
                {
                    "state": state,
                    "projection": projection,
                    "early_head_normalized_participation_ratio": early[
                        "mean_head_normalized_participation_ratio"
                    ],
                    "late_head_normalized_participation_ratio": late[
                        "mean_head_normalized_participation_ratio"
                    ],
                    "late_over_early_head_normalized_participation_ratio": (
                        late["mean_head_normalized_participation_ratio"]
                        / early["mean_head_normalized_participation_ratio"]
                    ),
                }
            )
    return result


def _build_q2_comparison(
    q3_regions: Sequence[Mapping[str, Any]], q2: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result = []
    q2_family = {"q": "q_proj", "v": "v_proj"}
    for projection in PROJECTIONS:
        for region in REGIONS:
            q3 = next(
                row
                for row in q3_regions
                if row["state"] == "frozen_final_checkpoint"
                and row["projection"] == projection
                and row["region"] == region
            )
            q2_row = next(
                row
                for row in q2["region_family_rows"]
                if row["family"] == q2_family[projection] and row["region"] == region
            )
            result.append(
                {
                    "projection": projection,
                    "region": region,
                    "q2_full_parameter_mean_head_normalized_participation_ratio": q2_row[
                        "mean_layer_normalized_participation_ratio"
                    ],
                    "q3_lora_mean_head_normalized_participation_ratio": q3[
                        "mean_head_normalized_participation_ratio"
                    ],
                    "q3_minus_q2": (
                        q3["mean_head_normalized_participation_ratio"]
                        - q2_row["mean_layer_normalized_participation_ratio"]
                    ),
                }
            )
    return result


def _build_flow_correlations(
    rows: Sequence[Mapping[str, Any]], block_flow: Mapping[str, Any]
) -> dict[str, Any]:
    flow_rows = sorted(
        (
            row
            for row in block_flow["block_rows"]
            if row["model_key"] == "q3" and row["normalized_query_fold"] == "all"
        ),
        key=lambda row: int(row["block_zero_based"]),
    )
    flow = [float(row["mean_update_common_energy_fraction"]) for row in flow_rows]
    result = {}
    for state in ("step_500", "frozen_final_checkpoint", "final_minus_step500"):
        result[state] = {}
        for projection in PROJECTIONS:
            selected = sorted(
                (row for row in rows if row["state"] == state and row["projection"] == projection),
                key=lambda row: int(row["layer_zero_based"]),
            )
            head_pr = [
                float(row["output_head"]["normalized_participation_ratio"])
                for row in selected
            ]
            result[state][projection] = _pearson(head_pr, flow)
    return {
        "descriptive_only": True,
        "flow_metric": "Q3 mean_update_common_energy_fraction",
        "weight_metric": "head normalized_participation_ratio",
        "pearson": result,
    }


def _region_for_layer(layer: int) -> str:
    for region, layers in REGIONS.items():
        if layer in layers:
            return region
    raise ValueError(f"Q3 layer outside registered regions: {layer}")


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return math.fsum(values) / len(values)


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson inputs must have equal length >=2")
    left_mean, right_mean = _mean(left), _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        math.fsum(value * value for value in left_centered)
        * math.fsum(value * value for value in right_centered)
    )
    if denominator <= 0.0:
        return None
    return math.fsum(x * y for x, y in zip(left_centered, right_centered)) / denominator


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
