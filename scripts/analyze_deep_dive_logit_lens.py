#!/usr/bin/env python3
"""Trace full-minus-null geometry through frozen Q2/Q3 Yes-No logit lenses.

At every captured hidden state this qrels-blind analysis applies the model's
frozen final RMSNorm and tied Yes-minus-No unembedding direction.  Q2 state 28
is its exact single-position native readout algebra (up to FP16 snapshot
precision).  Q3 uses only the first Yes/No answer-token contrast and is never
presented as the complete two-token teacher-forced likelihood score.

Intermediate-state logit lenses are descriptive and may be affected by basis
misalignment.  They do not select a layer or replace registered D2/D6 causal
patches and native scoring.
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
    "runs/20260718_kuaisearch_mech_d1_causal_floor_v1/causal_floor.json"
)
UPSTREAM_SHA256 = "7c50239e78dab0e1a2978ba61ba9109be3deee0a06ff642536d10e04a886e284"
DEEP_DIVE_MANIFEST = Path(
    "experiments/motivation/transformer_deep_dive_manifest.yaml"
)
DEEP_DIVE_MANIFEST_SHA256 = (
    "76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758"
)
Q2_WEIGHTS = Path(
    "artifacts/motivation_v1_2/checkpoints/"
    "q2_recranker_generalqwen_seed20260714/checkpoint_latest/model/model.safetensors"
)
Q2_WEIGHTS_SHA256 = "83e3467dc26a02e65a0a49efabf08273ddb6dc7bcea7b06fe5bb0aaf2825f7c9"
BASE_WEIGHTS = Path("models/huggingface/Qwen3-0.6B/model.safetensors")
BASE_WEIGHTS_SHA256 = "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b"
MODEL_SPECS = {
    "q2": {
        "weights": Q2_WEIGHTS,
        "weights_sha256": Q2_WEIGHTS_SHA256,
        "yes_token_id": 9693,
        "no_token_id": 2152,
        "readout_scope": "exact_q2_native_single_position_yes_no_logit_difference_at_state28",
    },
    "q3": {
        "weights": BASE_WEIGHTS,
        "weights_sha256": BASE_WEIGHTS_SHA256,
        "yes_token_id": 9454,
        "no_token_id": 2753,
        "readout_scope": "q3_first_answer_token_only_not_complete_native_score",
    },
}
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
    "score_total_delta",
    "score_common_delta",
    "score_residual_delta",
    "score_history_delta",
    "score_query_floor_delta",
    "score_full_candidate_relative",
    "score_null_candidate_relative",
    "activation_total_delta",
    "activation_common_delta",
    "activation_residual_delta",
)
DOT_NAMES = ("score_common_history", "score_common_query")
REGION_METRICS = (
    "total_score_delta_rms",
    "common_score_delta_rms",
    "residual_score_delta_rms",
    "history_score_delta_rms",
    "query_floor_score_delta_rms",
    "common_score_rms_over_query_floor",
    "residual_score_rms_over_query_floor",
    "full_candidate_relative_score_rms",
    "null_candidate_relative_score_rms",
    "full_over_null_candidate_relative_score_rms",
    "score_common_energy_fraction",
    "total_native_direction_energy_fraction",
    "common_native_direction_energy_fraction",
    "residual_native_direction_energy_fraction",
    "total_native_direction_isotropic_multiple",
    "common_native_direction_isotropic_multiple",
    "residual_native_direction_isotropic_multiple",
    "common_history_score_cosine",
    "common_query_floor_score_cosine",
    "common_history_same_sign_fraction",
    "common_query_floor_same_sign_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d6_logit_lens_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, expected, label in (
        (root / UPSTREAM_PATH, UPSTREAM_SHA256, "causal-floor upstream"),
        (
            root / DEEP_DIVE_MANIFEST,
            DEEP_DIVE_MANIFEST_SHA256,
            "deep-dive manifest",
        ),
        (root / Q2_WEIGHTS, Q2_WEIGHTS_SHA256, "Q2 final weights"),
        (root / BASE_WEIGHTS, BASE_WEIGHTS_SHA256, "Qwen base weights"),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"{label} hash drift")
    upstream = _read_json(root / UPSTREAM_PATH)
    if (
        upstream.get("status") != "completed"
        or upstream.get("qrels_read") is not False
        or upstream.get("model_scores_read") is not False
        or upstream.get("source_test_opened") is not False
    ):
        raise ValueError("causal-floor upstream boundary differs")
    selected = _selected_request_ids(root, upstream)

    state_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    readout_identity: dict[str, Any] = {}
    for model_key in ("q2", "q3"):
        spec = MODEL_SPECS[model_key]
        final_norm, direction, rows = _load_readout(
            root / spec["weights"],
            yes_token_id=int(spec["yes_token_id"]),
            no_token_id=int(spec["no_token_id"]),
        )
        model_rows, audit = _analyze_model(
            root,
            model_key,
            upstream["sources"][model_key],
            selected,
            final_norm,
            direction,
        )
        state_rows.extend(model_rows)
        source_audit[model_key] = audit
        readout_identity[model_key] = {
            "weight_path": spec["weights"].as_posix(),
            "weight_sha256": spec["weights_sha256"],
            "tied_embedding_key": "model.embed_tokens.weight",
            "final_norm_key": "model.norm.weight",
            "yes_token_id": spec["yes_token_id"],
            "no_token_id": spec["no_token_id"],
            "readout_scope": spec["readout_scope"],
            "yes_row_rms": float(np.sqrt(np.mean(np.square(rows[0])))),
            "no_row_rms": float(np.sqrt(np.mean(np.square(rows[1])))),
            "yes_no_row_cosine": _vector_cosine(rows[0], rows[1]),
            "direction_norm": float(np.linalg.norm(direction)),
            "final_norm_weight_rms": float(np.sqrt(np.mean(np.square(final_norm)))),
        }
    region_rows = _region_rows(state_rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d6_frozen_logit_lens",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_layer_selector": False,
        "interpretation_boundary": (
            "Intermediate logit lenses apply the final norm/readout outside their "
            "native depth and are descriptive. Q3 covers only the first Yes/No "
            "answer token, not the complete teacher-forced native likelihood."
        ),
        "lens_formula": "(final_rmsnorm(h) dot (E_yes - E_no))",
        "projection_definition": (
            "score-delta energy divided by squared readout norm and corresponding "
            "post-finalnorm activation-delta energy; isotropic multiple is fraction*1024"
        ),
        "weighting": (
            "Candidates are averaged within request, then each frozen request has "
            "equal weight in all/fold summaries."
        ),
        "upstream_path": UPSTREAM_PATH.as_posix(),
        "upstream_sha256": UPSTREAM_SHA256,
        "deep_dive_manifest_sha256": DEEP_DIVE_MANIFEST_SHA256,
        "readout_identity": readout_identity,
        "hidden_state_indices": list(STATES),
        "sources": source_audit,
        "state_rows": state_rows,
        "fixed_region_rows": region_rows,
        "qrels_read": False,
        "model_scores_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": " ".join(os.sys.argv),
    }
    output_path = output_dir / "logit_lens.json"
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


def _load_readout(
    path: Path, *, yes_token_id: int, no_token_id: int
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if "model.embed_tokens.weight" not in keys or "model.norm.weight" not in keys:
            raise ValueError("tied embedding/final norm weight is missing")
        embedding = handle.get_tensor("model.embed_tokens.weight")
        if not 0 <= yes_token_id < embedding.shape[0] or not 0 <= no_token_id < embedding.shape[0]:
            raise ValueError("readout token ID is outside tied embedding")
        yes = embedding[yes_token_id].float().cpu().numpy().astype(np.float64)
        no = embedding[no_token_id].float().cpu().numpy().astype(np.float64)
        final_norm = (
            handle.get_tensor("model.norm.weight")
            .float()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
    direction = yes - no
    if yes.shape != (HIDDEN_SIZE,) or no.shape != yes.shape or final_norm.shape != yes.shape:
        raise ValueError("readout tensor shape differs")
    if not np.isfinite(direction).all() or np.linalg.norm(direction) <= 1.0e-20:
        raise ValueError("readout direction is invalid")
    return final_norm, direction, (yes, no)


def _analyze_model(
    root: Path,
    model_key: str,
    source: Mapping[str, Any],
    selected_requests: Sequence[str],
    final_norm: np.ndarray,
    direction: np.ndarray,
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
    selected = set(selected_requests)
    accumulators = {
        "all": _empty_accumulator(),
        "fold0": _empty_accumulator(),
        "fold1": _empty_accumulator(),
    }
    seen: set[str] = set()
    selected_shards = 0
    verified_bytes = 0
    maximum_lens_identity_error = 0.0
    for full_entry, null_entry in zip(full_index["shards"], null_index["shards"]):
        if full_entry["path"] != null_entry["path"]:
            raise ValueError(f"{model_key} shard partition differs")
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
                "request_positions",
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
                full_candidate = np.asarray(
                    full["candidate_activations"][start:end], dtype=np.float64
                )
                null_candidate = np.asarray(
                    null["candidate_activations"][start:end], dtype=np.float64
                )
                full_request = np.asarray(
                    full["request_activations"][local], dtype=np.float64
                )
                null_request = np.asarray(
                    null["request_activations"][local], dtype=np.float64
                )
                metrics = _request_lens_metrics(
                    full_candidate,
                    null_candidate,
                    full_request,
                    null_request,
                    final_norm,
                    direction,
                )
                maximum_lens_identity_error = max(
                    maximum_lens_identity_error,
                    float(metrics.pop("maximum_lens_identity_error")),
                )
                fold = _fold(query)
                for group in ("all", f"fold{fold}"):
                    _update_accumulator(
                        accumulators[group], metrics, candidate_count=len(full_candidate)
                    )
    if seen != selected or len(seen) != 482:
        raise ValueError(f"{model_key} selected request coverage differs")
    rows = []
    direction_norm_squared = float(np.dot(direction, direction))
    for fold, accumulator in accumulators.items():
        rows.extend(
            _finalize_accumulator(
                model_key,
                fold,
                accumulator,
                direction_norm_squared=direction_norm_squared,
            )
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
        "maximum_lens_recomposition_error": maximum_lens_identity_error,
        "qrels_read": False,
        "source_test_opened": False,
    }


def _request_lens_metrics(
    full_candidate: np.ndarray,
    null_candidate: np.ndarray,
    full_request: np.ndarray,
    null_request: np.ndarray,
    final_norm: np.ndarray,
    direction: np.ndarray,
) -> dict[str, Any]:
    full_candidate_norm = _final_rmsnorm(full_candidate, final_norm)
    null_candidate_norm = _final_rmsnorm(null_candidate, final_norm)
    full_request_norm = _final_rmsnorm(full_request, final_norm)
    null_request_norm = _final_rmsnorm(null_request, final_norm)
    full_score = np.tensordot(full_candidate_norm, direction, axes=([-1], [0]))
    null_score = np.tensordot(null_candidate_norm, direction, axes=([-1], [0]))
    request_full_score = np.tensordot(full_request_norm, direction, axes=([-1], [0]))
    request_null_score = np.tensordot(null_request_norm, direction, axes=([-1], [0]))
    score_delta = full_score - null_score
    score_common = score_delta.mean(axis=0)
    score_residual = score_delta - score_common[None, :]
    full_relative = full_score - full_score.mean(axis=0)[None, :]
    null_relative = null_score - null_score.mean(axis=0)[None, :]
    activation_delta = full_candidate_norm - null_candidate_norm
    activation_common = activation_delta.mean(axis=0)
    activation_residual = activation_delta - activation_common[None, :, :]
    direct_score_delta = np.tensordot(activation_delta, direction, axes=([-1], [0]))
    identity_error = float(np.max(np.abs(direct_score_delta - score_delta)))
    history_delta = request_full_score[1] - request_null_score[1]
    query_delta = request_full_score[0] - request_null_score[0]
    return {
        "energy_score_total_delta": np.mean(np.square(score_delta), axis=0),
        "energy_score_common_delta": np.square(score_common),
        "energy_score_residual_delta": np.mean(np.square(score_residual), axis=0),
        "energy_score_history_delta": np.square(history_delta),
        "energy_score_query_floor_delta": np.square(query_delta),
        "energy_score_full_candidate_relative": np.mean(np.square(full_relative), axis=0),
        "energy_score_null_candidate_relative": np.mean(np.square(null_relative), axis=0),
        "energy_activation_total_delta": np.mean(
            np.sum(np.square(activation_delta), axis=2), axis=0
        ),
        "energy_activation_common_delta": np.sum(
            np.square(activation_common), axis=1
        ),
        "energy_activation_residual_delta": np.mean(
            np.sum(np.square(activation_residual), axis=2), axis=0
        ),
        "dot_score_common_history": score_common * history_delta,
        "dot_score_common_query": score_common * query_delta,
        "same_sign_common_history": (score_common * history_delta > 0).astype(np.float64),
        "same_sign_common_query": (score_common * query_delta > 0).astype(np.float64),
        "maximum_lens_identity_error": identity_error,
    }


def _final_rmsnorm(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape[-1] != len(weights):
        raise ValueError("logit-lens hidden width differs")
    inverse_rms = 1.0 / np.sqrt(
        np.mean(np.square(values), axis=-1, keepdims=True) + RMS_EPSILON
    )
    return values * inverse_rms * weights


def _empty_accumulator() -> dict[str, Any]:
    return {
        "request_count": 0,
        "candidate_count": 0,
        **{
            f"energy_{name}": np.zeros(len(STATES), dtype=np.float64)
            for name in ENERGY_NAMES
        },
        **{
            f"dot_{name}": np.zeros(len(STATES), dtype=np.float64)
            for name in DOT_NAMES
        },
        "same_sign_common_history": np.zeros(len(STATES), dtype=np.float64),
        "same_sign_common_query": np.zeros(len(STATES), dtype=np.float64),
    }


def _update_accumulator(
    accumulator: dict[str, Any], metrics: Mapping[str, Any], *, candidate_count: int
) -> None:
    accumulator["request_count"] += 1
    accumulator["candidate_count"] += int(candidate_count)
    for name in ENERGY_NAMES:
        accumulator[f"energy_{name}"] += np.asarray(metrics[f"energy_{name}"])
    for name in DOT_NAMES:
        accumulator[f"dot_{name}"] += np.asarray(metrics[f"dot_{name}"])
    for name in ("same_sign_common_history", "same_sign_common_query"):
        accumulator[name] += np.asarray(metrics[name])


def _finalize_accumulator(
    model_key: str,
    fold: str,
    accumulator: Mapping[str, Any],
    *,
    direction_norm_squared: float,
) -> list[dict[str, Any]]:
    count = int(accumulator["request_count"])
    if count <= 0 or direction_norm_squared <= 0:
        raise ValueError("logit-lens accumulator/direction is invalid")
    energy = {
        name: np.asarray(accumulator[f"energy_{name}"]) / count
        for name in ENERGY_NAMES
    }
    dots = {
        name: np.asarray(accumulator[f"dot_{name}"]) / count for name in DOT_NAMES
    }
    rows = []
    for state in STATES:
        projection = {}
        for name in ("total", "common", "residual"):
            fraction = _safe_divide(
                energy[f"score_{name}_delta"][state],
                direction_norm_squared * energy[f"activation_{name}_delta"][state],
            )
            projection[name] = fraction
        query_rms = math.sqrt(max(0.0, energy["score_query_floor_delta"][state]))
        full_relative_rms = math.sqrt(
            max(0.0, energy["score_full_candidate_relative"][state])
        )
        null_relative_rms = math.sqrt(
            max(0.0, energy["score_null_candidate_relative"][state])
        )
        row = {
            "model_key": model_key,
            "normalized_query_fold": fold,
            "hidden_state_index": state,
            "requests": count,
            "candidate_rows": int(accumulator["candidate_count"]),
            "total_score_delta_rms": math.sqrt(max(0.0, energy["score_total_delta"][state])),
            "common_score_delta_rms": math.sqrt(max(0.0, energy["score_common_delta"][state])),
            "residual_score_delta_rms": math.sqrt(max(0.0, energy["score_residual_delta"][state])),
            "history_score_delta_rms": math.sqrt(max(0.0, energy["score_history_delta"][state])),
            "query_floor_score_delta_rms": query_rms,
            "common_score_rms_over_query_floor": _rms_ratio(
                energy["score_common_delta"][state],
                energy["score_query_floor_delta"][state],
            ),
            "residual_score_rms_over_query_floor": _rms_ratio(
                energy["score_residual_delta"][state],
                energy["score_query_floor_delta"][state],
            ),
            "full_candidate_relative_score_rms": full_relative_rms,
            "null_candidate_relative_score_rms": null_relative_rms,
            "full_over_null_candidate_relative_score_rms": _safe_divide(
                full_relative_rms, null_relative_rms
            ),
            "score_common_energy_fraction": _safe_divide(
                energy["score_common_delta"][state], energy["score_total_delta"][state]
            ),
            "common_history_score_cosine": _energy_cosine(
                dots["score_common_history"][state],
                energy["score_common_delta"][state],
                energy["score_history_delta"][state],
            ),
            "common_query_floor_score_cosine": _energy_cosine(
                dots["score_common_query"][state],
                energy["score_common_delta"][state],
                energy["score_query_floor_delta"][state],
            ),
            "common_history_same_sign_fraction": float(
                accumulator["same_sign_common_history"][state] / count
            ),
            "common_query_floor_same_sign_fraction": float(
                accumulator["same_sign_common_query"][state] / count
            ),
        }
        for name, fraction in projection.items():
            row[f"{name}_native_direction_energy_fraction"] = fraction
            row[f"{name}_native_direction_isotropic_multiple"] = (
                None if fraction is None else fraction * HIDDEN_SIZE
            )
        rows.append(row)
    return rows


def _energy_cosine(dot: float, left_energy: float, right_energy: float) -> float | None:
    denominator = math.sqrt(max(0.0, float(left_energy * right_energy)))
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(dot / denominator, -1.0, 1.0))


def _rms_ratio(numerator: float, denominator: float) -> float | None:
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


def _selected_request_ids(
    root: Path, upstream: Mapping[str, Any]
) -> tuple[str, ...]:
    rmsnorm_path = root / str(upstream["upstream_path"])
    if _sha256_file(rmsnorm_path) != upstream["upstream_sha256"]:
        raise ValueError("RMSNorm-flow ancestor hash drift")
    rmsnorm = _read_json(rmsnorm_path)
    activation_path = root / str(rmsnorm["upstream_path"])
    if _sha256_file(activation_path) != rmsnorm["upstream_sha256"]:
        raise ValueError("activation-anisotropy ancestor hash drift")
    activation = _read_json(activation_path)
    sample_path = root / str(activation["request_anchor"]["sample_rows_path"])
    if _sha256_file(sample_path) != activation["request_anchor"]["sample_rows_sha256"]:
        raise ValueError("logit-lens request-anchor hash drift")
    ordered = []
    seen = set()
    for row in _iter_jsonl(sample_path):
        request_id = str(row["request_id"])
        if request_id not in seen:
            ordered.append(request_id)
            seen.add(request_id)
    if len(ordered) != 482:
        raise ValueError("logit-lens request-anchor count drift")
    return tuple(ordered)


def _fold(normalized_query: str) -> int:
    return int(hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(), 16) % 2


def _vector_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


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
