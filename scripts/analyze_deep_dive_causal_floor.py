#!/usr/bin/env python3
"""Calibrate history-conditioned activation geometry against query causal floor.

The query endpoint precedes the history span in the causal prompt.  Its
full-minus-null activation should therefore be invariant in exact arithmetic;
the small observed difference captures separate-batch, low-precision, and
common-position-offset numerical floor.  On the frozen 482-request anchor this
script compares that negative control with candidate-common, candidate-
relative residual, and history-summary deltas at every state.

The query delta is an empirical control, not an error term that may be
subtracted from scientific effects.  This analysis is qrels-blind and cannot
select a layer or replace causal interventions.
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


UPSTREAM_PATH = Path(
    "runs/20260718_kuaisearch_mech_d2_rmsnorm_flow_v1/rmsnorm_flow.json"
)
UPSTREAM_SHA256 = "7239f9e8247cb613c034e9c9ac138c696a2d8254b22a4814fd234e4729ffafdd"
STATES = tuple(range(29))
HIDDEN_SIZE = 1024
ENERGY_NAMES = ("total", "common", "residual", "history", "query_floor")
DOT_NAMES = ("common_history", "common_query", "history_query")
REGIONS = {
    "blocks_00_06": tuple(range(1, 8)),
    "blocks_07_13": tuple(range(8, 15)),
    "blocks_14_20": tuple(range(15, 22)),
    "blocks_21_27": tuple(range(22, 29)),
}
REGION_METRICS = (
    "query_floor_rms",
    "total_rms_over_query_floor",
    "common_rms_over_query_floor",
    "residual_rms_over_query_floor",
    "history_rms_over_query_floor",
    "common_history_cosine",
    "common_query_floor_cosine",
    "history_query_floor_cosine",
    "common_fraction_orthogonal_to_query_floor",
    "history_fraction_orthogonal_to_query_floor",
    "common_query_floor_channel_energy_cosine",
    "history_query_floor_channel_energy_cosine",
    "common_history_channel_energy_cosine",
    "common_query_floor_global_mean_cosine",
    "history_query_floor_global_mean_cosine",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d1_causal_floor_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if _sha256_file(root / UPSTREAM_PATH) != UPSTREAM_SHA256:
        raise ValueError("RMSNorm-flow upstream hash drift")
    upstream = _read_json(root / UPSTREAM_PATH)
    if (
        upstream.get("status") != "completed"
        or upstream.get("qrels_read") is not False
        or upstream.get("model_scores_read") is not False
        or upstream.get("source_test_opened") is not False
    ):
        raise ValueError("RMSNorm-flow upstream boundary differs")
    selected = _selected_request_ids(root, upstream)

    state_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for model_key in ("q2", "q3"):
        rows, audit = _analyze_model(
            root, model_key, upstream["sources"][model_key], selected
        )
        state_rows.extend(rows)
        source_audit[model_key] = audit
    region_rows = _region_rows(state_rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d1_query_causal_floor",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_layer_selector": False,
        "interpretation_boundary": (
            "Query-end full-minus-null is a causal-invariance numerical control. "
            "It is not an additive error estimate and is never subtracted from "
            "candidate/history activations or used to choose a layer."
        ),
        "query_control_definition": (
            "query_end occurs strictly before the history span; future history "
            "tokens cannot affect it under the causal mask in exact arithmetic"
        ),
        "weighting": (
            "Candidate energy is averaged within request; all group summaries "
            "give each frozen request equal weight."
        ),
        "upstream_path": UPSTREAM_PATH.as_posix(),
        "upstream_sha256": UPSTREAM_SHA256,
        "request_anchor": {
            "requests": len(selected),
            "selected_candidate_rows_per_model": 20357,
            "qrels_read": False,
        },
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
    output_path = output_dir / "causal_floor.json"
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


def _analyze_model(
    root: Path,
    model_key: str,
    source: Mapping[str, Any],
    selected_requests: Sequence[str],
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
        raise ValueError(f"{model_key} shard count differs")
    selected = set(selected_requests)
    accumulators = {
        "all": _empty_accumulator(),
        "fold0": _empty_accumulator(),
        "fold1": _empty_accumulator(),
    }
    seen: set[str] = set()
    selected_shards = 0
    verified_bytes = 0
    maximum_query_state0_delta = 0.0
    maximum_candidate_state0_delta = 0.0
    maximum_energy_identity_error = 0.0
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
                candidate = (
                    np.asarray(full["candidate_activations"][start:end], dtype=np.float64)
                    - np.asarray(null["candidate_activations"][start:end], dtype=np.float64)
                )
                request = (
                    np.asarray(full["request_activations"][local], dtype=np.float64)
                    - np.asarray(null["request_activations"][local], dtype=np.float64)
                )
                if candidate.shape[1:] != (len(STATES), HIDDEN_SIZE) or request.shape != (
                    2,
                    len(STATES),
                    HIDDEN_SIZE,
                ):
                    raise ValueError(f"{model_key} selected activation shape differs")
                common = candidate.mean(axis=0)
                residual = candidate - common[None, :, :]
                query_floor = request[0]
                history = request[1]
                metrics = _request_metrics(
                    candidate, common, residual, history, query_floor
                )
                maximum_energy_identity_error = max(
                    maximum_energy_identity_error,
                    float(metrics.pop("maximum_energy_identity_error")),
                )
                maximum_query_state0_delta = max(
                    maximum_query_state0_delta,
                    float(np.max(np.abs(query_floor[0]))),
                )
                maximum_candidate_state0_delta = max(
                    maximum_candidate_state0_delta,
                    float(np.max(np.abs(candidate[:, 0]))),
                )
                fold = _fold(query)
                for group in ("all", f"fold{fold}"):
                    _update_accumulator(
                        accumulators[group], metrics, candidate_count=len(candidate)
                    )
    if seen != selected or len(seen) != 482:
        raise ValueError(f"{model_key} selected request coverage differs")
    if maximum_query_state0_delta != 0.0 or maximum_candidate_state0_delta != 0.0:
        raise ValueError(f"{model_key} state-0 causal identity failed")
    rows = []
    for fold, accumulator in accumulators.items():
        rows.extend(_finalize_accumulator(model_key, fold, accumulator))
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
        "maximum_query_state0_delta": maximum_query_state0_delta,
        "maximum_candidate_state0_delta": maximum_candidate_state0_delta,
        "maximum_channelwise_candidate_energy_identity_error": maximum_energy_identity_error,
        "qrels_read": False,
        "source_test_opened": False,
    }


def _request_metrics(
    candidate: np.ndarray,
    common: np.ndarray,
    residual: np.ndarray,
    history: np.ndarray,
    query_floor: np.ndarray,
) -> dict[str, Any]:
    total_energy = np.mean(np.square(candidate), axis=(0, 2))
    common_energy = np.mean(np.square(common), axis=1)
    residual_energy = np.mean(np.square(residual), axis=(0, 2))
    identity_error = float(
        np.max(np.abs(total_energy - common_energy - residual_energy))
    )
    result: dict[str, Any] = {
        "energy_total": total_energy,
        "energy_common": common_energy,
        "energy_residual": residual_energy,
        "energy_history": np.mean(np.square(history), axis=1),
        "energy_query_floor": np.mean(np.square(query_floor), axis=1),
        "dot_common_history": np.mean(common * history, axis=1),
        "dot_common_query": np.mean(common * query_floor, axis=1),
        "dot_history_query": np.mean(history * query_floor, axis=1),
        "profile_common": np.square(common),
        "profile_history": np.square(history),
        "profile_query_floor": np.square(query_floor),
        "vector_common": common,
        "vector_history": history,
        "vector_query_floor": query_floor,
        "maximum_energy_identity_error": identity_error,
    }
    return result


def _empty_accumulator() -> dict[str, Any]:
    state_shape = (len(STATES),)
    profile_shape = (len(STATES), HIDDEN_SIZE)
    return {
        "request_count": 0,
        "candidate_count": 0,
        **{
            f"energy_{name}": np.zeros(state_shape, dtype=np.float64)
            for name in ENERGY_NAMES
        },
        **{
            f"dot_{name}": np.zeros(state_shape, dtype=np.float64)
            for name in DOT_NAMES
        },
        **{
            f"profile_{name}": np.zeros(profile_shape, dtype=np.float64)
            for name in ("common", "history", "query_floor")
        },
        **{
            f"vector_{name}": np.zeros(profile_shape, dtype=np.float64)
            for name in ("common", "history", "query_floor")
        },
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
    for name in ("common", "history", "query_floor"):
        accumulator[f"profile_{name}"] += np.asarray(metrics[f"profile_{name}"])
        accumulator[f"vector_{name}"] += np.asarray(metrics[f"vector_{name}"])


def _finalize_accumulator(
    model_key: str, fold: str, accumulator: Mapping[str, Any]
) -> list[dict[str, Any]]:
    count = int(accumulator["request_count"])
    if count <= 0:
        raise ValueError("causal-floor accumulator is empty")
    energy = {
        name: np.asarray(accumulator[f"energy_{name}"]) / count
        for name in ENERGY_NAMES
    }
    dots = {
        name: np.asarray(accumulator[f"dot_{name}"]) / count for name in DOT_NAMES
    }
    profiles = {
        name: np.asarray(accumulator[f"profile_{name}"]) / count
        for name in ("common", "history", "query_floor")
    }
    vectors = {
        name: np.asarray(accumulator[f"vector_{name}"]) / count
        for name in ("common", "history", "query_floor")
    }
    rows = []
    for state in STATES:
        common_query_cos = _energy_cosine(
            dots["common_query"][state],
            energy["common"][state],
            energy["query_floor"][state],
        )
        history_query_cos = _energy_cosine(
            dots["history_query"][state],
            energy["history"][state],
            energy["query_floor"][state],
        )
        row = {
            "model_key": model_key,
            "normalized_query_fold": fold,
            "hidden_state_index": state,
            "requests": count,
            "candidate_rows": int(accumulator["candidate_count"]),
            "query_floor_rms": math.sqrt(max(0.0, float(energy["query_floor"][state]))),
            "total_rms_over_query_floor": _rms_ratio(
                energy["total"][state], energy["query_floor"][state]
            ),
            "common_rms_over_query_floor": _rms_ratio(
                energy["common"][state], energy["query_floor"][state]
            ),
            "residual_rms_over_query_floor": _rms_ratio(
                energy["residual"][state], energy["query_floor"][state]
            ),
            "history_rms_over_query_floor": _rms_ratio(
                energy["history"][state], energy["query_floor"][state]
            ),
            "common_history_cosine": _energy_cosine(
                dots["common_history"][state],
                energy["common"][state],
                energy["history"][state],
            ),
            "common_query_floor_cosine": common_query_cos,
            "history_query_floor_cosine": history_query_cos,
            "common_fraction_orthogonal_to_query_floor": _orthogonal_fraction(
                common_query_cos
            ),
            "history_fraction_orthogonal_to_query_floor": _orthogonal_fraction(
                history_query_cos
            ),
            "common_query_floor_channel_energy_cosine": _profile_cosine(
                profiles["common"][state], profiles["query_floor"][state]
            ),
            "history_query_floor_channel_energy_cosine": _profile_cosine(
                profiles["history"][state], profiles["query_floor"][state]
            ),
            "common_history_channel_energy_cosine": _profile_cosine(
                profiles["common"][state], profiles["history"][state]
            ),
            "common_query_floor_global_mean_cosine": _vector_cosine(
                vectors["common"][state], vectors["query_floor"][state]
            ),
            "history_query_floor_global_mean_cosine": _vector_cosine(
                vectors["history"][state], vectors["query_floor"][state]
            ),
        }
        rows.append(row)
    return rows


def _energy_cosine(dot: float, left_energy: float, right_energy: float) -> float | None:
    denominator = math.sqrt(max(0.0, float(left_energy * right_energy)))
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(dot / denominator, -1.0, 1.0))


def _rms_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 1.0e-30:
        return None
    return math.sqrt(max(0.0, float(numerator / denominator)))


def _orthogonal_fraction(cosine: float | None) -> float | None:
    return None if cosine is None else float(max(0.0, 1.0 - cosine * cosine))


def _profile_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    return _vector_cosine(left, right)


def _vector_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


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
    activation_path = root / str(upstream["upstream_path"])
    if _sha256_file(activation_path) != upstream["upstream_sha256"]:
        raise ValueError("activation-anisotropy ancestor hash drift")
    activation = _read_json(activation_path)
    sample_path = root / str(activation["request_anchor"]["sample_rows_path"])
    if _sha256_file(sample_path) != activation["request_anchor"]["sample_rows_sha256"]:
        raise ValueError("causal-floor frozen request-anchor hash drift")
    ordered = []
    seen = set()
    for row in _iter_jsonl(sample_path):
        request_id = str(row["request_id"])
        if request_id not in seen:
            ordered.append(request_id)
            seen.add(request_id)
    if len(ordered) != 482:
        raise ValueError("causal-floor request-anchor count drift")
    return tuple(ordered)


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
