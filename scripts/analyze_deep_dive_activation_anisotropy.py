#!/usr/bin/env python3
"""Measure qrels-blind channel anisotropy of full-minus-null activations.

The frozen 512-row candidate control identifies 482 requests without labels.
For every one of their 20,357 candidates and all 29 hidden states, this script
decomposes the candidate displacement into a request-common vector and a
candidate-relative residual.  It then measures coordinate-channel energy
concentration and cross-request directional consensus for candidate-common and
history-summary deltas.

These are descriptive, basis-dependent activation diagnostics.  They neither
select a layer/channel nor replace the registered D2--D4 causal interventions.
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
    "runs/20260718_kuaisearch_mech_d1_candidate_residual_v1/metrics.json"
)
UPSTREAM_SHA256 = "e9055d5011d71e85df1582e15bc885f93e4744a07d0c2fc995ad5f634a1d7c5f"
SAMPLE_MANIFEST = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1/manifest.json"
)
SAMPLE_MANIFEST_SHA256 = (
    "84cdf68a0fabefcab055806bb690adf96f2a36ad2921c2d10c5d0aae8310aa61"
)
SAMPLE_ROWS = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1/candidate_rows.jsonl"
)
SAMPLE_ROWS_SHA256 = (
    "258f9303b15d0778d8ca7fe91883f694424f25bc18271cb76f4a9da2941eb985"
)
STATES = tuple(range(29))
HIDDEN_SIZE = 1024
REGIONS = {
    "blocks_00_06": tuple(range(1, 8)),
    "blocks_07_13": tuple(range(8, 15)),
    "blocks_14_20": tuple(range(15, 22)),
    "blocks_21_27": tuple(range(22, 29)),
}
PROFILE_NAMES = ("total", "common", "residual", "history")
PROFILE_METRICS = (
    "channel_participation_ratio",
    "top_1pct_channel_energy_share",
    "top_5pct_channel_energy_share",
    "top_10pct_channel_energy_share",
)
STATE_METRICS = (
    "common_energy_fraction",
    "common_global_mean_energy_fraction",
    "common_mean_pairwise_cosine",
    "history_global_mean_energy_fraction",
    "history_mean_pairwise_cosine",
    "common_residual_channel_energy_cosine",
    "common_history_channel_energy_cosine",
    *(f"{profile}_{metric}" for profile in PROFILE_NAMES for metric in PROFILE_METRICS),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d1_activation_anisotropy_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for path, expected, label in (
        (root / UPSTREAM_PATH, UPSTREAM_SHA256, "candidate residual analysis"),
        (root / SAMPLE_MANIFEST, SAMPLE_MANIFEST_SHA256, "sample manifest"),
        (root / SAMPLE_ROWS, SAMPLE_ROWS_SHA256, "sample rows"),
    ):
        if _sha256_file(path) != expected:
            raise ValueError(f"frozen {label} hash drift")
    upstream = _read_json(root / UPSTREAM_PATH)
    if (
        upstream.get("status") != "completed"
        or upstream.get("qrels_read") is not False
        or upstream.get("source_test_opened") is not False
        or upstream.get("request_anchor", {}).get("requests") != 482
    ):
        raise ValueError("upstream candidate residual boundary differs")
    selected_rows = list(_iter_jsonl(root / SAMPLE_ROWS))
    ordered_requests = tuple(
        dict.fromkeys(str(row["request_id"]) for row in selected_rows)
    )
    if len(selected_rows) != 512 or len(ordered_requests) != 482:
        raise ValueError("frozen qrels-blind request anchor differs")

    state_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for model_key in ("q2", "q3"):
        source = upstream["sources"][model_key]
        rows, audit = _analyze_model(root, model_key, source, ordered_requests)
        state_rows.extend(rows)
        source_audit[model_key] = audit
    region_rows = _region_rows(state_rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d1_activation_anisotropy",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_layer_selector": False,
        "channel_selector": False,
        "interpretation_boundary": (
            "Channel-energy participation is coordinate-basis dependent and "
            "cross-request consensus is descriptive. Neither establishes a causal "
            "head/MLP channel, an erasure layer, or a transferable architecture."
        ),
        "weighting": (
            "Candidate energy is averaged within request, then requests receive "
            "equal weight. Fold and all-request summaries use the same rule."
        ),
        "decomposition": (
            "delta_i=h_full_i-h_null_i; common=mean_candidates(delta_i); "
            "residual_i=delta_i-common, with channelwise total=common+residual energy."
        ),
        "direction_consensus": (
            "Global-mean energy fraction compares squared mean-vector norm with "
            "mean squared norm. Mean pairwise cosine is computed exactly from the "
            "sum of per-request unit vectors, excluding self-pairs."
        ),
        "request_anchor": upstream["request_anchor"],
        "upstream_path": UPSTREAM_PATH.as_posix(),
        "upstream_sha256": UPSTREAM_SHA256,
        "hidden_state_indices": list(STATES),
        "hidden_size": HIDDEN_SIZE,
        "sources": source_audit,
        "state_rows": state_rows,
        "fixed_region_rows": region_rows,
        "qrels_read": False,
        "model_scores_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": " ".join(os.sys.argv),
    }
    output_path = output_dir / "activation_anisotropy.json"
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
    ordered_requests: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_dir = root / str(source["full_bundle"])
    null_dir = root / str(source["null_bundle"])
    for directory, prefix in ((full_dir, "full"), (null_dir, "null")):
        if _sha256_file(directory / "metadata.json") != source[f"{prefix}_metadata_sha256"]:
            raise ValueError(f"{model_key} {prefix} metadata hash drift")
        if _sha256_file(directory / "index.json") != source[f"{prefix}_index_sha256"]:
            raise ValueError(f"{model_key} {prefix} index hash drift")
    full_metadata = _read_json(full_dir / "metadata.json")
    null_metadata = _read_json(null_dir / "metadata.json")
    for condition, metadata in (("full", full_metadata), ("null", null_metadata)):
        if (
            metadata.get("status") != "completed"
            or metadata.get("result_eligible") is not True
            or metadata.get("condition_id") != condition
            or metadata.get("method_id") != source["method_id"]
            or metadata.get("hidden_state_indices") != list(STATES)
            or metadata.get("qrels_read") is not False
            or metadata.get("source_test_opened") is not False
        ):
            raise ValueError(f"{model_key} {condition} representation boundary differs")
    full_index = _read_json(full_dir / "index.json")
    null_index = _read_json(null_dir / "index.json")
    for index in (full_index, null_index):
        if index.get("request_count") != 8000 or index.get("candidate_count") != 160753:
            raise ValueError(f"{model_key} representation population differs")
    if len(full_index["shards"]) != len(null_index["shards"]):
        raise ValueError(f"{model_key} full/null shard partitions differ")

    selected = set(ordered_requests)
    accumulators = {
        "all": _empty_accumulator(),
        "fold0": _empty_accumulator(),
        "fold1": _empty_accumulator(),
    }
    seen: set[str] = set()
    selected_shards = 0
    verified_bytes = 0
    maximum_energy_identity_error = 0.0
    for full_entry, null_entry in zip(full_index["shards"], null_index["shards"]):
        if full_entry["path"] != null_entry["path"]:
            raise ValueError(f"{model_key} full/null shard path differs")
        full_path = full_dir / "shards" / str(full_entry["path"])
        null_path = null_dir / "shards" / str(null_entry["path"])
        with np.load(full_path, allow_pickle=False) as full:
            request_ids = tuple(str(value) for value in full["request_ids"].tolist())
        if not selected.intersection(request_ids):
            continue
        if _sha256_file(full_path) != full_entry["sha256"]:
            raise ValueError(f"{model_key} selected full shard hash drift")
        if _sha256_file(null_path) != null_entry["sha256"]:
            raise ValueError(f"{model_key} selected null shard hash drift")
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
            if full["hidden_state_indices"].tolist() != list(STATES):
                raise ValueError(f"{model_key} hidden-state indices differ")
            if full["request_positions"].tolist() != [
                "query_end",
                "history_summary_end",
            ]:
                raise ValueError(f"{model_key} request positions differ")
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
                delta = (
                    np.asarray(full["candidate_activations"][start:end], dtype=np.float32)
                    - np.asarray(null["candidate_activations"][start:end], dtype=np.float32)
                )
                history = (
                    np.asarray(full["request_activations"][local, 1], dtype=np.float32)
                    - np.asarray(null["request_activations"][local, 1], dtype=np.float32)
                )
                if delta.shape[1:] != (len(STATES), HIDDEN_SIZE) or history.shape != (
                    len(STATES),
                    HIDDEN_SIZE,
                ):
                    raise ValueError(f"{model_key} selected activation shape differs")
                if len(delta) < 2 or not np.isfinite(delta).all() or not np.isfinite(history).all():
                    raise FloatingPointError(f"{model_key} selected activation is invalid")
                common = delta.mean(axis=0, dtype=np.float64)
                residual = np.asarray(delta, dtype=np.float64) - common[None, :, :]
                total_energy = np.mean(np.square(delta, dtype=np.float64), axis=0)
                common_energy = np.square(common)
                residual_energy = np.mean(np.square(residual), axis=0)
                identity_error = float(
                    np.max(np.abs(total_energy - common_energy - residual_energy))
                )
                maximum_energy_identity_error = max(
                    maximum_energy_identity_error, identity_error
                )
                tolerance = 2.0e-10 * max(1.0, float(np.max(total_energy)))
                if identity_error > tolerance:
                    raise FloatingPointError(
                        f"{model_key} candidate energy identity failed"
                    )
                fold = _fold(query)
                for group in ("all", f"fold{fold}"):
                    _update_accumulator(
                        accumulators[group],
                        total_energy=total_energy,
                        common=common,
                        common_energy=common_energy,
                        residual_energy=residual_energy,
                        history=np.asarray(history, dtype=np.float64),
                        candidate_count=len(delta),
                    )
    if seen != selected or len(seen) != 482:
        raise ValueError(f"{model_key} selected request coverage differs")
    if accumulators["all"]["candidate_count"] != 20357:
        raise ValueError(f"{model_key} selected candidate coverage differs")
    rows = []
    for group_name, accumulator in accumulators.items():
        rows.extend(_finalize_accumulator(model_key, group_name, accumulator))
    return rows, {
        "method_id": source["method_id"],
        "checkpoint_id": source["checkpoint_id"],
        "full_bundle": str(source["full_bundle"]),
        "null_bundle": str(source["null_bundle"]),
        "full_metadata_sha256": source["full_metadata_sha256"],
        "null_metadata_sha256": source["null_metadata_sha256"],
        "full_index_sha256": source["full_index_sha256"],
        "null_index_sha256": source["null_index_sha256"],
        "selected_shards_verified": selected_shards,
        "selected_shard_bytes_verified": verified_bytes,
        "selected_requests": accumulators["all"]["request_count"],
        "selected_candidate_rows": accumulators["all"]["candidate_count"],
        "maximum_channelwise_energy_identity_error": maximum_energy_identity_error,
        "qrels_read": False,
        "source_test_opened": False,
    }


def _empty_accumulator() -> dict[str, Any]:
    shape = (len(STATES), HIDDEN_SIZE)
    return {
        "request_count": 0,
        "candidate_count": 0,
        "energy": {
            profile: np.zeros(shape, dtype=np.float64) for profile in PROFILE_NAMES
        },
        "vector_sum": {
            "common": np.zeros(shape, dtype=np.float64),
            "history": np.zeros(shape, dtype=np.float64),
        },
        "unit_sum": {
            "common": np.zeros(shape, dtype=np.float64),
            "history": np.zeros(shape, dtype=np.float64),
        },
        "unit_count": {
            "common": np.zeros(len(STATES), dtype=np.int64),
            "history": np.zeros(len(STATES), dtype=np.int64),
        },
    }


def _update_accumulator(
    accumulator: dict[str, Any],
    *,
    total_energy: np.ndarray,
    common: np.ndarray,
    common_energy: np.ndarray,
    residual_energy: np.ndarray,
    history: np.ndarray,
    candidate_count: int,
) -> None:
    accumulator["request_count"] += 1
    accumulator["candidate_count"] += int(candidate_count)
    accumulator["energy"]["total"] += total_energy
    accumulator["energy"]["common"] += common_energy
    accumulator["energy"]["residual"] += residual_energy
    accumulator["energy"]["history"] += np.square(history)
    for name, values in (("common", common), ("history", history)):
        accumulator["vector_sum"][name] += values
        norms = np.linalg.norm(values, axis=1)
        valid = norms > 1.0e-20
        accumulator["unit_sum"][name][valid] += values[valid] / norms[valid, None]
        accumulator["unit_count"][name][valid] += 1


def _finalize_accumulator(
    model_key: str, group_name: str, accumulator: Mapping[str, Any]
) -> list[dict[str, Any]]:
    count = int(accumulator["request_count"])
    if count <= 0:
        raise ValueError("activation accumulator is empty")
    profiles = {
        name: np.asarray(values, dtype=np.float64) / count
        for name, values in accumulator["energy"].items()
    }
    rows = []
    for state in STATES:
        total_energy = float(profiles["total"][state].sum())
        common_energy = float(profiles["common"][state].sum())
        row: dict[str, Any] = {
            "model_key": model_key,
            "normalized_query_fold": group_name,
            "hidden_state_index": state,
            "requests": count,
            "candidate_rows": int(accumulator["candidate_count"]),
            "common_energy_fraction": _safe_divide(common_energy, total_energy),
            "common_residual_channel_energy_cosine": _profile_cosine(
                profiles["common"][state], profiles["residual"][state]
            ),
            "common_history_channel_energy_cosine": _profile_cosine(
                profiles["common"][state], profiles["history"][state]
            ),
        }
        for profile in PROFILE_NAMES:
            metrics = _channel_energy_metrics(profiles[profile][state])
            for metric, value in metrics.items():
                row[f"{profile}_{metric}"] = value
        for name in ("common", "history"):
            vector_mean = np.asarray(accumulator["vector_sum"][name][state]) / count
            denominator = float(profiles[name][state].sum())
            row[f"{name}_global_mean_energy_fraction"] = _safe_divide(
                float(np.dot(vector_mean, vector_mean)), denominator
            )
            row[f"{name}_mean_pairwise_cosine"] = _mean_pairwise_cosine_from_sum(
                np.asarray(accumulator["unit_sum"][name][state]),
                int(accumulator["unit_count"][name][state]),
            )
        rows.append(row)
    return rows


def _channel_energy_metrics(energy: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(energy, dtype=np.float64)
    if values.shape != (HIDDEN_SIZE,) or np.any(values < -1.0e-20):
        raise ValueError("channel energy profile is invalid")
    total = float(values.sum())
    if total <= 1.0e-30:
        return {
            "channel_participation_ratio": None,
            "top_1pct_channel_energy_share": None,
            "top_5pct_channel_energy_share": None,
            "top_10pct_channel_energy_share": None,
        }
    squared = float(np.dot(values, values))
    ordered = np.sort(values)[::-1]
    return {
        "channel_participation_ratio": total * total / (HIDDEN_SIZE * squared),
        "top_1pct_channel_energy_share": float(ordered[: math.ceil(HIDDEN_SIZE * 0.01)].sum() / total),
        "top_5pct_channel_energy_share": float(ordered[: math.ceil(HIDDEN_SIZE * 0.05)].sum() / total),
        "top_10pct_channel_energy_share": float(ordered[: math.ceil(HIDDEN_SIZE * 0.10)].sum() / total),
    }


def _mean_pairwise_cosine_from_sum(
    unit_vector_sum: np.ndarray, count: int
) -> float | None:
    if count < 2:
        return None
    squared_sum_norm = float(np.dot(unit_vector_sum, unit_vector_sum))
    value = (squared_sum_norm - count) / (count * (count - 1))
    return float(np.clip(value, -1.0, 1.0))


def _profile_cosine(left: np.ndarray, right: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-30:
        return None
    return float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))


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
                for metric in STATE_METRICS:
                    values = [value for item in selected if (value := item[metric]) is not None]
                    row[metric] = None if not values else _mean(values)
                rows.append(row)
    return rows


def _fold(normalized_query: str) -> int:
    if not normalized_query:
        raise ValueError("normalized query is empty")
    return int(hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(), 16) % 2


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
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
