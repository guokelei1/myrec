#!/usr/bin/env python3
"""Audit how each Transformer block updates candidate history deltas.

For the complete candidate slates of the 482 qrels-blind frozen request
anchors, this D1 diagnostic writes the exact residual-stream energy identity

    output_energy = input_energy + update_energy + 2<input, update>

separately for request-common and candidate-relative full-minus-null deltas.
It distinguishes absolute attenuation from relative drowning without opening
dev qrels or selecting a block from outcomes.  The result is descriptive and
does not replace the registered D2/D3 causal interventions.
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


FROZEN_SAMPLE_MANIFEST_SHA256 = (
    "84cdf68a0fabefcab055806bb690adf96f2a36ad2921c2d10c5d0aae8310aa61"
)
FROZEN_SAMPLE_ROWS_SHA256 = (
    "258f9303b15d0778d8ca7fe91883f694424f25bc18271cb76f4a9da2941eb985"
)
SAMPLE_DIR = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1"
)
MODELS = {
    "q2": {
        "method_id": "q2_recranker_generalqwen",
        "full": Path("runs/20260718_kuaisearch_mech_d1_q2_dev_full_all29"),
        "null": Path("runs/20260718_kuaisearch_mech_d1_q2_dev_null_all29"),
        "evaluation": Path("runs/20260718_kuaisearch_mech_d1_q2_eval_all29_v2"),
    },
    "q3": {
        "method_id": "q3_tallrec_generalqwen",
        "full": Path("runs/20260718_kuaisearch_mech_d1_q3_dev_full_all29"),
        "null": Path("runs/20260718_kuaisearch_mech_d1_q3_dev_null_all29"),
        "evaluation": Path("runs/20260718_kuaisearch_mech_d1_q3_eval_all29_v2"),
    },
}
STATES = tuple(range(29))
BLOCKS = tuple(range(28))
COMPONENTS = ("common", "candidate_relative")
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
        default="runs/20260718_kuaisearch_mech_d1_candidate_block_flow_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_manifest_path = root / SAMPLE_DIR / "manifest.json"
    sample_rows_path = root / SAMPLE_DIR / "candidate_rows.jsonl"
    if _sha256_file(sample_manifest_path) != FROZEN_SAMPLE_MANIFEST_SHA256:
        raise ValueError("frozen candidate-row manifest hash drift")
    if _sha256_file(sample_rows_path) != FROZEN_SAMPLE_ROWS_SHA256:
        raise ValueError("frozen candidate-row data hash drift")
    sample_manifest = _read_json(sample_manifest_path)
    if (
        sample_manifest.get("qrels_read") is not False
        or sample_manifest.get("source_test_opened") is not False
        or sample_manifest.get("selected_candidate_rows") != 512
    ):
        raise ValueError("frozen request-anchor safety boundary failed")
    sample_rows = list(_iter_jsonl(sample_rows_path))
    ordered_requests = tuple(
        dict.fromkeys(str(row["request_id"]) for row in sample_rows)
    )
    if len(sample_rows) != 512 or len(ordered_requests) != 482:
        raise ValueError("frozen request-anchor population drift")

    block_rows: list[dict[str, Any]] = []
    source_audit: dict[str, Any] = {}
    for model_key, spec in MODELS.items():
        request_rows, source = _load_request_flows(
            root, model_key, spec, ordered_requests
        )
        source_audit[model_key] = source
        block_rows.extend(_summarize_model(model_key, request_rows))
    region_rows = _build_region_rows(block_rows)

    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d1_candidate_block_flow",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_layer_selector": False,
        "interpretation_boundary": (
            "The paired residual-stream algebra localizes geometric construction, "
            "interference, and attenuation. Growing candidate-relative energy can "
            "still lose task alignment, and a negative cross term is not by itself "
            "a causal ranking effect. Registered D2/D3 interventions remain required."
        ),
        "request_anchor": {
            "selection": (
                "all candidates belonging to the 482 requests present in the "
                "qrels-blind frozen 512 candidate-row sample"
            ),
            "sample_manifest_path": SAMPLE_DIR.joinpath("manifest.json").as_posix(),
            "sample_manifest_sha256": FROZEN_SAMPLE_MANIFEST_SHA256,
            "sample_rows_path": SAMPLE_DIR.joinpath("candidate_rows.jsonl").as_posix(),
            "sample_rows_sha256": FROZEN_SAMPLE_ROWS_SHA256,
            "anchor_candidate_rows": len(sample_rows),
            "requests": len(ordered_requests),
            "qrels_read": False,
        },
        "flow_definition": (
            "delta_s=h_full_s-h_null_s; block_update_b=delta_(b+1)-delta_b. "
            "Each request is first decomposed across candidates into common and "
            "candidate-relative parts. All reported means weight requests equally."
        ),
        "hidden_state_indices": list(STATES),
        "block_zero_based_indices": list(BLOCKS),
        "components": list(COMPONENTS),
        "sources": source_audit,
        "block_rows": block_rows,
        "fixed_region_rows": region_rows,
        "qrels_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": " ".join(os.sys.argv),
    }
    output_path = output_dir / "metrics.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "block_rows": len(block_rows),
                "region_rows": len(region_rows),
                "output": str(output_path),
                "sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _load_request_flows(
    root: Path,
    model_key: str,
    spec: Mapping[str, Any],
    ordered_requests: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full_dir = root / spec["full"]
    null_dir = root / spec["null"]
    full_meta = _read_json(full_dir / "metadata.json")
    null_meta = _read_json(null_dir / "metadata.json")
    full_index = _read_json(full_dir / "index.json")
    null_index = _read_json(null_dir / "index.json")
    for condition, metadata in (("full", full_meta), ("null", null_meta)):
        expected = {
            "status": "completed",
            "result_eligible": True,
            "method_id": spec["method_id"],
            "condition_id": condition,
            "qrels_read": False,
            "source_test_opened": False,
            "hidden_state_indices": list(STATES),
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f"{model_key} {condition} metadata differs: {key}")
    invariants = (
        "method_id",
        "checkpoint_id",
        "config_sha256",
        "records_sha256",
        "candidate_manifest_sha256",
        "request_manifest_sha256",
        "dataset_manifest_sha256",
        "deep_dive_manifest_sha256",
        "request_positions",
        "candidate_positions",
        "hidden_state_indices",
    )
    for key in invariants:
        if full_meta.get(key) != null_meta.get(key):
            raise ValueError(f"{model_key} full/null invariant differs: {key}")
    for index in (full_index, null_index):
        if index.get("request_count") != 8000 or index.get("candidate_count") != 160753:
            raise ValueError(f"{model_key} representation population differs")
    if len(full_index.get("shards", [])) != len(null_index.get("shards", [])):
        raise ValueError(f"{model_key} full/null shard count differs")

    evaluation_dir = root / spec["evaluation"]
    evaluation_metrics = _read_json(evaluation_dir / "metrics.json")
    pre_qrels_path = evaluation_dir / "pre_qrels_audit.json"
    if evaluation_metrics.get("status") != "completed":
        raise ValueError(f"{model_key} D1 evaluation is incomplete")
    if evaluation_metrics.get("pre_qrels_audit_sha256") != _sha256_file(pre_qrels_path):
        raise ValueError(f"{model_key} D1 pre-qrels audit hash differs")
    pre_qrels = _read_json(pre_qrels_path)
    for condition, directory in (("full", full_dir), ("null", null_dir)):
        declared = pre_qrels["bundles"][condition]
        if declared.get("metadata_sha256") != _sha256_file(directory / "metadata.json"):
            raise ValueError(f"{model_key} evaluated metadata hash differs: {condition}")
        if declared.get("index_sha256") != _sha256_file(directory / "index.json"):
            raise ValueError(f"{model_key} evaluated index hash differs: {condition}")

    selected = set(ordered_requests)
    request_order = {request_id: i for i, request_id in enumerate(ordered_requests)}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    verified_shards = 0
    selected_candidate_rows = 0
    for full_shard, null_shard in zip(full_index["shards"], null_index["shards"]):
        if full_shard["path"] != null_shard["path"]:
            raise ValueError(f"{model_key} full/null shard partition differs")
        full_path = full_dir / "shards" / full_shard["path"]
        null_path = null_dir / "shards" / null_shard["path"]
        with np.load(full_path, allow_pickle=False) as full:
            request_ids = [str(value) for value in full["request_ids"].tolist()]
        if not selected.intersection(request_ids):
            continue
        if _sha256_file(full_path) != full_shard["sha256"]:
            raise ValueError(f"{model_key} selected full shard hash differs")
        if _sha256_file(null_path) != null_shard["sha256"]:
            raise ValueError(f"{model_key} selected null shard hash differs")
        verified_shards += 1
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
                    raise ValueError(f"{model_key} selected shard alignment differs: {key}")
            request_ids = [str(value) for value in full["request_ids"].tolist()]
            queries = [str(value) for value in full["normalized_queries"].tolist()]
            offsets = np.asarray(full["candidate_offsets"], dtype=np.int64)
            for local, request_id in enumerate(request_ids):
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
                if len(delta) < 2 or delta.shape[1:] != (len(STATES), 1024):
                    raise ValueError(f"{model_key} selected slate shape differs")
                if not np.isfinite(delta).all():
                    raise ValueError(f"{model_key} selected delta is non-finite")
                flow = _block_flow(delta)
                flow.update(
                    {
                        "request_id": request_id,
                        "normalized_query": queries[local],
                        "candidate_count": len(delta),
                    }
                )
                rows.append(flow)
                selected_candidate_rows += len(delta)
    if seen != selected or len(rows) != len(ordered_requests):
        raise ValueError(f"{model_key} selected request coverage is incomplete")
    rows.sort(key=lambda row: request_order[str(row["request_id"])])
    return rows, {
        "method_id": spec["method_id"],
        "checkpoint_id": full_meta["checkpoint_id"],
        "full_bundle": spec["full"].as_posix(),
        "null_bundle": spec["null"].as_posix(),
        "full_metadata_sha256": _sha256_file(full_dir / "metadata.json"),
        "null_metadata_sha256": _sha256_file(null_dir / "metadata.json"),
        "full_index_sha256": _sha256_file(full_dir / "index.json"),
        "null_index_sha256": _sha256_file(null_dir / "index.json"),
        "selected_shards_verified": verified_shards,
        "selected_requests": len(rows),
        "selected_candidate_rows": selected_candidate_rows,
        "qrels_read": False,
        "source_test_opened": False,
    }


def _block_flow(delta: np.ndarray) -> dict[str, Any]:
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (len(STATES), 1024):
        raise ValueError("candidate deltas have the wrong shape")
    common = values.mean(axis=0)
    candidate_relative = values - common[None, :, :]
    common_flow = _component_flow(common, state_axis=0)
    relative_flow = _component_flow(candidate_relative, state_axis=1)
    maximum_identity_error = max(
        common_flow["maximum_energy_identity_error"],
        relative_flow["maximum_energy_identity_error"],
    )
    output_total = common_flow["output_mse"] + relative_flow["output_mse"]
    update_total = common_flow["update_mse"] + relative_flow["update_mse"]
    return {
        "common": common_flow,
        "candidate_relative": relative_flow,
        "output_common_energy_fraction": _safe_ratio(
            common_flow["output_mse"], output_total
        ),
        "update_common_energy_fraction": _safe_ratio(
            common_flow["update_mse"], update_total
        ),
        "maximum_energy_identity_error": maximum_identity_error,
    }


def _component_flow(values: np.ndarray, *, state_axis: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    previous = np.take(array, BLOCKS, axis=state_axis)
    following = np.take(array, tuple(block + 1 for block in BLOCKS), axis=state_axis)
    update = following - previous
    reduction_axes = tuple(axis for axis in range(array.ndim) if axis != state_axis)
    input_mse = np.mean(previous**2, axis=reduction_axes)
    update_mse = np.mean(update**2, axis=reduction_axes)
    output_mse = np.mean(following**2, axis=reduction_axes)
    dot_mse = np.mean(previous * update, axis=reduction_axes)
    cross_mse = 2.0 * dot_mse
    identity_error = np.abs(output_mse - input_mse - update_mse - cross_mse)
    tolerance = 2.0e-10 * np.maximum(output_mse + input_mse + update_mse, 1.0)
    if np.any(identity_error > tolerance):
        raise ValueError("block residual-stream energy identity failed")
    input_update_cosine = _cosine_arrays(dot_mse, input_mse, update_mse)
    input_output_dot = input_mse + dot_mse
    input_output_cosine = _cosine_arrays(
        input_output_dot, input_mse, output_mse
    )
    update_projection_coefficient = _safe_ratio(dot_mse, input_mse)
    output_input_rms_ratio = np.sqrt(_safe_ratio(output_mse, input_mse))
    return {
        "input_mse": input_mse,
        "update_mse": update_mse,
        "output_mse": output_mse,
        "energy_change": output_mse - input_mse,
        "interaction_cross_mse": cross_mse,
        "input_update_cosine": input_update_cosine,
        "input_output_cosine": input_output_cosine,
        "update_projection_coefficient": update_projection_coefficient,
        "output_input_rms_ratio": output_input_rms_ratio,
        "energy_decreased": output_mse < input_mse,
        "maximum_energy_identity_error": float(np.max(identity_error)),
    }


def _cosine_arrays(
    dot: np.ndarray, left_mse: np.ndarray, right_mse: np.ndarray
) -> np.ndarray:
    denominator = np.sqrt(
        np.asarray(left_mse, dtype=np.float64)
        * np.asarray(right_mse, dtype=np.float64)
    )
    result = np.full_like(denominator, np.nan)
    valid = denominator > 1.0e-20
    result[valid] = np.clip(np.asarray(dot)[valid] / denominator[valid], -1.0, 1.0)
    return result


def _summarize_model(
    model_key: str, request_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for block in BLOCKS:
        for fold_name in ("all", "0", "1"):
            selected = [
                row
                for row in request_rows
                if fold_name == "all"
                or _fold(str(row["normalized_query"])) == int(fold_name)
            ]
            row: dict[str, Any] = {
                "model_key": model_key,
                "block_zero_based": block,
                "input_hidden_state_index": block,
                "output_hidden_state_index": block + 1,
                "normalized_query_fold": fold_name,
                "requests": len(selected),
                "candidate_rows": int(
                    sum(int(value["candidate_count"]) for value in selected)
                ),
                "mean_output_common_energy_fraction": _mean_finite(
                    [float(value["output_common_energy_fraction"][block]) for value in selected]
                ),
                "mean_update_common_energy_fraction": _mean_finite(
                    [float(value["update_common_energy_fraction"][block]) for value in selected]
                ),
                "maximum_energy_identity_error": max(
                    float(value["maximum_energy_identity_error"]) for value in selected
                ),
            }
            for component in COMPONENTS:
                prefix = "common" if component == "common" else "candidate_relative"
                flows = [value[component] for value in selected]
                for energy_name in ("input_mse", "update_mse", "output_mse"):
                    mean_energy = _mean(
                        [float(flow[energy_name][block]) for flow in flows]
                    )
                    row[f"{prefix}_{energy_name}"] = mean_energy
                    row[f"{prefix}_{energy_name[:-3]}rms"] = math.sqrt(mean_energy)
                for name in (
                    "energy_change",
                    "interaction_cross_mse",
                    "input_update_cosine",
                    "input_output_cosine",
                    "update_projection_coefficient",
                    "output_input_rms_ratio",
                ):
                    row[f"mean_{prefix}_{name}"] = _mean_finite(
                        [float(flow[name][block]) for flow in flows]
                    )
                row[f"fraction_requests_{prefix}_energy_decreased"] = _mean(
                    [float(flow["energy_decreased"][block]) for flow in flows]
                )
            rows.append(row)
    return rows


def _build_region_rows(
    block_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    lookup = {
        (
            str(row["model_key"]),
            str(row["normalized_query_fold"]),
            int(row["block_zero_based"]),
        ): row
        for row in block_rows
    }
    metrics = (
        "mean_output_common_energy_fraction",
        "mean_update_common_energy_fraction",
        "mean_common_energy_change",
        "mean_common_interaction_cross_mse",
        "mean_common_input_update_cosine",
        "fraction_requests_common_energy_decreased",
        "mean_candidate_relative_energy_change",
        "mean_candidate_relative_interaction_cross_mse",
        "mean_candidate_relative_input_update_cosine",
        "mean_candidate_relative_update_projection_coefficient",
        "fraction_requests_candidate_relative_energy_decreased",
    )
    rows = []
    for model_key in MODELS:
        for fold_name in ("all", "0", "1"):
            for region, blocks in REGIONS.items():
                selected = [lookup[(model_key, fold_name, block)] for block in blocks]
                row: dict[str, Any] = {
                    "model_key": model_key,
                    "normalized_query_fold": fold_name,
                    "region": region,
                    "block_zero_based_indices": list(blocks),
                }
                for metric in metrics:
                    values = [
                        value[metric]
                        for value in selected
                        if value[metric] is not None
                    ]
                    row[metric] = _mean_finite(
                        [float(value) for value in values]
                    )
                rows.append(row)
    return rows


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    numerator64 = np.asarray(numerator, dtype=np.float64)
    denominator64 = np.asarray(denominator, dtype=np.float64)
    result = np.full_like(numerator64, np.nan)
    valid = denominator64 > 1.0e-20
    result[valid] = numerator64[valid] / denominator64[valid]
    return result


def _fold(normalized_query: str) -> int:
    digest = hashlib.sha256(normalized_query.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 2


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return float(math.fsum(values) / len(values))


def _mean_finite(values: Sequence[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return None if not finite else _mean(finite)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


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
