#!/usr/bin/env python3
"""Replay every frozen Q2/Q3 input/final RMSNorm on qrels-blind states.

State 0 is the embedding output, states 1--28 are exact decoder-layer outputs
captured by hooks.  Therefore states 0--27 are inputs to blocks 0--27 and state
28 is the input to final RMSNorm.  On the frozen 482-request, 20,357-candidate
anchor this script replays the corresponding learned RMSNorm and measures how
it changes full-minus-null total, request-common, and candidate-relative
residual deltas across all 29 norm sites.

The source activations are stored FP16 and the replay uses float64 algebra.
This is a descriptive normalization-flow audit, not a causal intervention or
a substitute for registered D2 branch/node patches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from safetensors import safe_open


UPSTREAM_PATH = Path(
    "runs/20260718_kuaisearch_mech_d1_activation_anisotropy_v1/"
    "activation_anisotropy.json"
)
UPSTREAM_SHA256 = "ed02819400ff04ba7e6dde58117eef9d29ee0576d2b09aae28c5bcf72a44bfe8"
Q2_WEIGHTS = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q2_recranker_generalqwen_seed20260714/checkpoint_latest/model/model.safetensors"
)
Q2_WEIGHTS_SHA256 = "83e3467dc26a02e65a0a49efabf08273ddb6dc7bcea7b06fe5bb0aaf2825f7c9"
BASE_WEIGHTS = Path("models/huggingface/Qwen3-0.6B/model.safetensors")
BASE_WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
Q3_ADAPTER = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q3_tallrec_generalqwen_seed20260714/checkpoint_latest/model/adapter_model.safetensors"
)
Q3_ADAPTER_SHA256 = "fd51a9c6b9ee3a6651597c263a8120db52cb79d62e7c80e544666e46bc5e1cef"
Q3_ADAPTER_CONFIG = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q3_tallrec_generalqwen_seed20260714/checkpoint_latest/model/adapter_config.json"
)
Q3_ADAPTER_CONFIG_SHA256 = (
    "96398d25a5e76a1e22b26b9123127c0bf825291716762ef03f79b11a44ee8bdd"
)
STATES = tuple(range(29))
HIDDEN_SIZE = 1024
RMS_EPSILON = 1.0e-6
REGIONS = {
    "blocks_00_06": tuple(range(1, 8)),
    "blocks_07_13": tuple(range(8, 15)),
    "blocks_14_20": tuple(range(15, 22)),
    "blocks_21_27": tuple(range(22, 29)),
}
ENERGY_NAMES = (
    "pre_total",
    "post_total",
    "pre_common",
    "post_common",
    "pre_residual",
    "post_residual",
)
DOT_NAMES = ("total_dot", "common_dot", "residual_dot")
RMS_NAMES = ("full_input_rms", "null_input_rms", "full_output_rms", "null_output_rms")
REGION_METRICS = (
    "total_delta_rms_gain",
    "common_delta_rms_gain",
    "residual_delta_rms_gain",
    "residual_to_common_gain_ratio",
    "total_delta_pre_post_cosine",
    "common_delta_pre_post_cosine",
    "residual_delta_pre_post_cosine",
    "common_energy_fraction_pre",
    "common_energy_fraction_post",
    "common_energy_fraction_change",
    "full_over_null_input_rms",
    "full_over_null_output_rms",
    "rmsnorm_weight_rms",
    "rmsnorm_weight_channel_participation_ratio",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d2_rmsnorm_flow_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, expected, label in (
        (root / UPSTREAM_PATH, UPSTREAM_SHA256, "activation anisotropy upstream"),
        (root / Q2_WEIGHTS, Q2_WEIGHTS_SHA256, "Q2 final weights"),
        (root / BASE_WEIGHTS, BASE_WEIGHTS_SHA256, "Qwen base weights"),
        (root / Q3_ADAPTER, Q3_ADAPTER_SHA256, "Q3 LoRA weights"),
        (root / Q3_ADAPTER_CONFIG, Q3_ADAPTER_CONFIG_SHA256, "Q3 adapter config"),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"{label} hash drift")
    upstream = _read_json(root / UPSTREAM_PATH)
    if (
        upstream.get("status") != "completed"
        or upstream.get("qrels_read") is not False
        or upstream.get("model_scores_read") is not False
        or upstream.get("source_test_opened") is not False
        or upstream.get("hidden_state_indices") != list(STATES)
    ):
        raise ValueError("activation anisotropy upstream boundary differs")
    adapter_config = _read_json(root / Q3_ADAPTER_CONFIG)
    if (
        adapter_config.get("peft_type") != "LORA"
        or set(adapter_config.get("target_modules", [])) != {"q_proj", "v_proj"}
        or adapter_config.get("modules_to_save") is not None
        or adapter_config.get("bias") != "none"
    ):
        raise ValueError("Q3 adapter may modify RMSNorm unexpectedly")
    with safe_open(root / Q3_ADAPTER, framework="np") as adapter:
        if any("norm" in key.casefold() for key in adapter.keys()):
            raise ValueError("Q3 adapter contains a norm parameter")

    state_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    weight_specs = {"q2": root / Q2_WEIGHTS, "q3": root / BASE_WEIGHTS}
    for model_key in ("q2", "q3"):
        norm_weights = _load_norm_weights(weight_specs[model_key])
        rows, audit = _analyze_model(
            root,
            model_key,
            upstream["sources"][model_key],
            norm_weights,
        )
        state_rows.extend(rows)
        source_audit[model_key] = {
            **audit,
            "norm_weight_source": weight_specs[model_key].relative_to(root).as_posix(),
            "norm_weight_source_sha256": (
                Q2_WEIGHTS_SHA256 if model_key == "q2" else BASE_WEIGHTS_SHA256
            ),
            "q3_adapter_norm_parameters": 0 if model_key == "q3" else None,
        }
    region_rows = _region_rows(state_rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_rmsnorm_flow",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_node_selector": False,
        "interpretation_boundary": (
            "Offline RMSNorm replay on FP16 activation snapshots measures local "
            "normalization geometry. It cannot establish a causal rescue, replace "
            "BF16 runtime node patches, or select a block after outcomes."
        ),
        "state_semantics": {
            "0_to_27": "embedding/layer output serving as input to block-index RMSNorm",
            "28": "block-27 output serving as input to final RMSNorm",
        },
        "rmsnorm_formula": "x * rsqrt(mean(x^2) + 1e-6) * learned_weight",
        "weighting": (
            "Candidate energy is averaged within request, then each frozen request "
            "receives equal weight in all/fold summaries."
        ),
        "upstream_path": UPSTREAM_PATH.as_posix(),
        "upstream_sha256": UPSTREAM_SHA256,
        "q3_adapter_path": Q3_ADAPTER.as_posix(),
        "q3_adapter_sha256": Q3_ADAPTER_SHA256,
        "q3_adapter_norm_parameters": 0,
        "hidden_state_indices": list(STATES),
        "hidden_size": HIDDEN_SIZE,
        "rms_epsilon": RMS_EPSILON,
        "sources": source_audit,
        "state_rows": state_rows,
        "fixed_region_rows": region_rows,
        "qrels_read": False,
        "model_scores_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": " ".join(os.sys.argv),
    }
    output_path = output_dir / "rmsnorm_flow.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "state_rows": len(state_rows),
                "region_rows": len(region_rows),
                "sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _load_norm_weights(path: Path) -> np.ndarray:
    rows = []
    # Frozen full-parameter checkpoints store RMSNorm weights as BF16. NumPy
    # has no native BF16 dtype in this environment, so use safetensors' torch
    # backend solely for lossless CPU decoding before the declared float64
    # descriptive replay.
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        expected = {
            *(f"model.layers.{state}.input_layernorm.weight" for state in range(28)),
            "model.norm.weight",
        }
        if not expected.issubset(keys):
            raise ValueError("RMSNorm weight keys are incomplete")
        for state in range(28):
            rows.append(
                handle.get_tensor(f"model.layers.{state}.input_layernorm.weight")
                .float()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
        rows.append(
            handle.get_tensor("model.norm.weight")
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    weights = np.stack(rows)
    if weights.shape != (len(STATES), HIDDEN_SIZE) or not np.isfinite(weights).all():
        raise ValueError("RMSNorm weight matrix is invalid")
    return weights


def _analyze_model(
    root: Path,
    model_key: str,
    source: Mapping[str, Any],
    norm_weights: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_dir = root / str(source["full_bundle"])
    null_dir = root / str(source["null_bundle"])
    for directory, prefix in ((full_dir, "full"), (null_dir, "null")):
        if _sha256_file(directory / "metadata.json") != source[f"{prefix}_metadata_sha256"]:
            raise ValueError(f"{model_key} {prefix} metadata hash drift")
        if _sha256_file(directory / "index.json") != source[f"{prefix}_index_sha256"]:
            raise ValueError(f"{model_key} {prefix} index hash drift")
    full_index = _read_json(full_dir / "index.json")
    null_index = _read_json(null_dir / "index.json")
    if len(full_index["shards"]) != len(null_index["shards"]):
        raise ValueError(f"{model_key} representation shard count differs")
    selected = set(_selected_request_ids(root))
    accumulators = {
        "all": _empty_accumulator(),
        "fold0": _empty_accumulator(),
        "fold1": _empty_accumulator(),
    }
    seen: set[str] = set()
    selected_shards = 0
    verified_bytes = 0
    maximum_pre_identity_error = 0.0
    maximum_post_identity_error = 0.0
    for full_entry, null_entry in zip(full_index["shards"], null_index["shards"]):
        if full_entry["path"] != null_entry["path"]:
            raise ValueError(f"{model_key} shard partitions differ")
        full_path = full_dir / "shards" / str(full_entry["path"])
        null_path = null_dir / "shards" / str(null_entry["path"])
        with np.load(full_path, allow_pickle=False) as full:
            request_ids = tuple(str(value) for value in full["request_ids"].tolist())
        if not selected.intersection(request_ids):
            continue
        if _sha256_file(full_path) != full_entry["sha256"] or _sha256_file(
            null_path
        ) != null_entry["sha256"]:
            raise ValueError(f"{model_key} selected shard hash drift")
        selected_shards += 1
        verified_bytes += full_path.stat().st_size + null_path.stat().st_size
        with np.load(full_path, allow_pickle=False) as full, np.load(
            null_path, allow_pickle=False
        ) as null:
            for key in (
                "request_ids",
                "normalized_queries",
                "candidate_offsets",
                "candidate_ids",
                "hidden_state_indices",
            ):
                if not np.array_equal(full[key], null[key]):
                    raise ValueError(f"{model_key} full/null alignment differs: {key}")
            request_ids = tuple(str(value) for value in full["request_ids"].tolist())
            queries = tuple(str(value) for value in full["normalized_queries"].tolist())
            offsets = np.asarray(full["candidate_offsets"], dtype=np.int64)
            for local, (request_id, query) in enumerate(zip(request_ids, queries)):
                if request_id not in selected:
                    continue
                if request_id in seen:
                    raise ValueError(f"{model_key} duplicate selected request")
                seen.add(request_id)
                start, end = int(offsets[local]), int(offsets[local + 1])
                full_values = np.asarray(
                    full["candidate_activations"][start:end], dtype=np.float64
                )
                null_values = np.asarray(
                    null["candidate_activations"][start:end], dtype=np.float64
                )
                if full_values.shape[1:] != (len(STATES), HIDDEN_SIZE) or len(
                    full_values
                ) < 2:
                    raise ValueError(f"{model_key} candidate activation shape differs")
                metrics = _request_norm_metrics(
                    full_values, null_values, norm_weights, epsilon=RMS_EPSILON
                )
                maximum_pre_identity_error = max(
                    maximum_pre_identity_error,
                    float(metrics.pop("maximum_pre_energy_identity_error")),
                )
                maximum_post_identity_error = max(
                    maximum_post_identity_error,
                    float(metrics.pop("maximum_post_energy_identity_error")),
                )
                fold = _fold(query)
                for group in ("all", f"fold{fold}"):
                    _update_accumulator(
                        accumulators[group], metrics, candidate_count=len(full_values)
                    )
    if seen != selected or len(seen) != 482:
        raise ValueError(f"{model_key} selected request coverage differs")
    if accumulators["all"]["candidate_count"] != 20357:
        raise ValueError(f"{model_key} selected candidate coverage differs")
    weight_metrics = [_weight_metrics(norm_weights[state]) for state in STATES]
    rows = []
    for group, accumulator in accumulators.items():
        rows.extend(
            _finalize_accumulator(model_key, group, accumulator, weight_metrics)
        )
    return rows, {
        "method_id": source["method_id"],
        "checkpoint_id": source["checkpoint_id"],
        "full_bundle": source["full_bundle"],
        "null_bundle": source["null_bundle"],
        "full_metadata_sha256": source["full_metadata_sha256"],
        "null_metadata_sha256": source["null_metadata_sha256"],
        "full_index_sha256": source["full_index_sha256"],
        "null_index_sha256": source["null_index_sha256"],
        "selected_shards_verified": selected_shards,
        "selected_shard_bytes_verified": verified_bytes,
        "selected_requests": accumulators["all"]["request_count"],
        "selected_candidate_rows": accumulators["all"]["candidate_count"],
        "maximum_pre_channelwise_energy_identity_error": maximum_pre_identity_error,
        "maximum_post_channelwise_energy_identity_error": maximum_post_identity_error,
        "qrels_read": False,
        "source_test_opened": False,
    }


def _request_norm_metrics(
    full: np.ndarray,
    null: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float,
) -> dict[str, np.ndarray | float]:
    if full.shape != null.shape or full.ndim != 3 or weights.shape != full.shape[1:]:
        raise ValueError("RMSNorm replay shapes differ")
    post_full = _rmsnorm(full, weights, epsilon=epsilon)
    post_null = _rmsnorm(null, weights, epsilon=epsilon)
    pre = full - null
    post = post_full - post_null
    pre_common = pre.mean(axis=0)
    post_common = post.mean(axis=0)
    pre_residual = pre - pre_common[None, :, :]
    post_residual = post - post_common[None, :, :]
    metrics: dict[str, np.ndarray | float] = {
        "pre_total": np.mean(np.square(pre), axis=(0, 2)),
        "post_total": np.mean(np.square(post), axis=(0, 2)),
        "pre_common": np.mean(np.square(pre_common), axis=1),
        "post_common": np.mean(np.square(post_common), axis=1),
        "pre_residual": np.mean(np.square(pre_residual), axis=(0, 2)),
        "post_residual": np.mean(np.square(post_residual), axis=(0, 2)),
        "total_dot": np.mean(pre * post, axis=(0, 2)),
        "common_dot": np.mean(pre_common * post_common, axis=1),
        "residual_dot": np.mean(pre_residual * post_residual, axis=(0, 2)),
        "full_input_rms": np.mean(np.sqrt(np.mean(np.square(full), axis=2)), axis=0),
        "null_input_rms": np.mean(np.sqrt(np.mean(np.square(null), axis=2)), axis=0),
        "full_output_rms": np.mean(
            np.sqrt(np.mean(np.square(post_full), axis=2)), axis=0
        ),
        "null_output_rms": np.mean(
            np.sqrt(np.mean(np.square(post_null), axis=2)), axis=0
        ),
    }
    metrics["maximum_pre_energy_identity_error"] = float(
        np.max(
            np.abs(
                np.asarray(metrics["pre_total"])
                - np.asarray(metrics["pre_common"])
                - np.asarray(metrics["pre_residual"])
            )
        )
    )
    metrics["maximum_post_energy_identity_error"] = float(
        np.max(
            np.abs(
                np.asarray(metrics["post_total"])
                - np.asarray(metrics["post_common"])
                - np.asarray(metrics["post_residual"])
            )
        )
    )
    return metrics


def _rmsnorm(values: np.ndarray, weights: np.ndarray, *, epsilon: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.shape[-2:] != weights.shape or epsilon <= 0:
        raise ValueError("RMSNorm values/weights/epsilon are invalid")
    inverse_rms = 1.0 / np.sqrt(np.mean(np.square(values), axis=-1, keepdims=True) + epsilon)
    return values * inverse_rms * weights[None, :, :]


def _empty_accumulator() -> dict[str, Any]:
    return {
        "request_count": 0,
        "candidate_count": 0,
        **{
            name: np.zeros(len(STATES), dtype=np.float64)
            for name in (*ENERGY_NAMES, *DOT_NAMES, *RMS_NAMES)
        },
    }


def _update_accumulator(
    accumulator: dict[str, Any], metrics: Mapping[str, Any], *, candidate_count: int
) -> None:
    accumulator["request_count"] += 1
    accumulator["candidate_count"] += int(candidate_count)
    for name in (*ENERGY_NAMES, *DOT_NAMES, *RMS_NAMES):
        values = np.asarray(metrics[name], dtype=np.float64)
        if values.shape != (len(STATES),) or not np.isfinite(values).all():
            raise FloatingPointError(f"RMSNorm request metric is invalid: {name}")
        accumulator[name] += values


def _finalize_accumulator(
    model_key: str,
    fold: str,
    accumulator: Mapping[str, Any],
    weight_metrics: Sequence[Mapping[str, float]],
) -> list[dict[str, Any]]:
    count = int(accumulator["request_count"])
    if count <= 0:
        raise ValueError("RMSNorm accumulator is empty")
    means = {
        name: np.asarray(accumulator[name], dtype=np.float64) / count
        for name in (*ENERGY_NAMES, *DOT_NAMES, *RMS_NAMES)
    }
    rows = []
    for state in STATES:
        total_gain = _sqrt_ratio(means["post_total"][state], means["pre_total"][state])
        common_gain = _sqrt_ratio(means["post_common"][state], means["pre_common"][state])
        residual_gain = _sqrt_ratio(
            means["post_residual"][state], means["pre_residual"][state]
        )
        pre_fraction = _safe_divide(
            means["pre_common"][state], means["pre_total"][state]
        )
        post_fraction = _safe_divide(
            means["post_common"][state], means["post_total"][state]
        )
        row = {
            "model_key": model_key,
            "normalized_query_fold": fold,
            "hidden_state_index": state,
            "norm_target": (
                f"block_{state}_input_rmsnorm" if state < 28 else "final_rmsnorm"
            ),
            "requests": count,
            "candidate_rows": int(accumulator["candidate_count"]),
            "total_delta_rms_gain": total_gain,
            "common_delta_rms_gain": common_gain,
            "residual_delta_rms_gain": residual_gain,
            "residual_to_common_gain_ratio": (
                None
                if residual_gain is None or common_gain is None
                else _safe_divide(residual_gain, common_gain)
            ),
            "total_delta_pre_post_cosine": _energy_cosine(
                means["total_dot"][state],
                means["pre_total"][state],
                means["post_total"][state],
            ),
            "common_delta_pre_post_cosine": _energy_cosine(
                means["common_dot"][state],
                means["pre_common"][state],
                means["post_common"][state],
            ),
            "residual_delta_pre_post_cosine": _energy_cosine(
                means["residual_dot"][state],
                means["pre_residual"][state],
                means["post_residual"][state],
            ),
            "common_energy_fraction_pre": pre_fraction,
            "common_energy_fraction_post": post_fraction,
            "common_energy_fraction_change": (
                None
                if pre_fraction is None or post_fraction is None
                else post_fraction - pre_fraction
            ),
            "full_input_rms": float(means["full_input_rms"][state]),
            "null_input_rms": float(means["null_input_rms"][state]),
            "full_output_rms": float(means["full_output_rms"][state]),
            "null_output_rms": float(means["null_output_rms"][state]),
            "full_over_null_input_rms": _safe_divide(
                means["full_input_rms"][state], means["null_input_rms"][state]
            ),
            "full_over_null_output_rms": _safe_divide(
                means["full_output_rms"][state], means["null_output_rms"][state]
            ),
            **weight_metrics[state],
        }
        rows.append(row)
    return rows


def _weight_metrics(weights: np.ndarray) -> dict[str, float]:
    values = np.asarray(weights, dtype=np.float64)
    energy = np.square(values)
    total = float(energy.sum())
    return {
        "rmsnorm_weight_rms": float(np.sqrt(np.mean(energy))),
        "rmsnorm_weight_minimum": float(values.min()),
        "rmsnorm_weight_maximum": float(values.max()),
        "rmsnorm_weight_channel_participation_ratio": (
            total * total / (len(values) * float(np.dot(energy, energy)))
        ),
    }


def _energy_cosine(dot: float, left_energy: float, right_energy: float) -> float | None:
    denominator = math.sqrt(max(0.0, left_energy * right_energy))
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(dot / denominator, -1.0, 1.0))


def _sqrt_ratio(numerator: float, denominator: float) -> float | None:
    ratio = _safe_divide(numerator, denominator)
    return None if ratio is None else math.sqrt(max(0.0, ratio))


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator <= 1.0e-30:
        return None
    return float(numerator / denominator)


def _region_rows(state_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    lookup = {
        (str(row["model_key"]), str(row["normalized_query_fold"]), int(row["hidden_state_index"])): row
        for row in state_rows
    }
    rows = []
    for model_key in ("q2", "q3"):
        for fold in ("all", "fold0", "fold1"):
            for region, states in REGIONS.items():
                selected = [lookup[(model_key, fold, state)] for state in states]
                row: dict[str, Any] = {
                    "model_key": model_key,
                    "normalized_query_fold": fold,
                    "region": region,
                    "hidden_state_indices": list(states),
                }
                for metric in REGION_METRICS:
                    values = [item[metric] for item in selected if item[metric] is not None]
                    row[metric] = None if not values else _mean(values)
                rows.append(row)
    return rows


def _selected_request_ids(root: Path) -> tuple[str, ...]:
    upstream = _read_json(root / UPSTREAM_PATH)
    sample_path = root / str(upstream["request_anchor"]["sample_rows_path"])
    if _sha256_file(sample_path) != upstream["request_anchor"]["sample_rows_sha256"]:
        raise ValueError("RMSNorm frozen request-anchor hash drift")
    request_ids = []
    seen = set()
    for row in _iter_jsonl(sample_path):
        request_id = str(row["request_id"])
        if request_id not in seen:
            request_ids.append(request_id)
            seen.add(request_id)
    if len(request_ids) != 482:
        raise ValueError("RMSNorm frozen request-anchor count drift")
    return tuple(request_ids)


def _fold(normalized_query: str) -> int:
    return int(hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(), 16) % 2


def _mean(values: Sequence[float]) -> float:
    return float(math.fsum(float(value) for value in values) / len(values))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL object required: {path}")
                yield value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
