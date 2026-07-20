#!/usr/bin/env python3
"""Audit the exact token-position geometry exercised by registered D5 RoPE.

This analysis reads only the frozen internal-dev records, Q2/Q3 prompt configs,
tokenizer, and qrels-blind content-neutral eligibility rows.  It reconstructs
every eligible candidate prompt and quantifies the natural history-to-readout
distances before the already-registered layer-local phase compression and
expansion.  It never reads model scores or qrels and is descriptive: the
registered D5 causal evaluator remains the only source of performance effects.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.mechanism.representation_probe import (
    instrument_pointwise_prompt,
    normalize_query,
)
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


DEEP_DIVE_PLAN = Path("experiments/motivation/transformer_deep_dive_plan.md")
DEEP_DIVE_PLAN_SHA256 = (
    "07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584"
)
DEEP_DIVE_MANIFEST = Path(
    "experiments/motivation/transformer_deep_dive_manifest.yaml"
)
DEEP_DIVE_MANIFEST_SHA256 = (
    "76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758"
)
CONTENT_DIR = Path(
    "artifacts/motivation_transformer_deep_dive/frozen_controls/content_neutral_v1"
)
CONTENT_MANIFEST_SHA256 = (
    "934cea39662585329fa5f8330f07b5a8decc0233d5a0ed610fb06cfb45dcbd24"
)
RECORDS_PATH = Path(
    "data/standardized/kuaisearch/full_confirm_preceding40k_v11/records_dev.jsonl"
)
RECORDS_SHA256 = "907046aa0ed69bced6a6115c15ce4f81ac6ada2937cdc810fc48a82822a2a06e"
TOKENIZER_PATH = Path("models/huggingface/Qwen3-0.6B")
TOKENIZER_SHA256 = "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
METHODS = {
    "q2": {
        "method_id": "q2_recranker_generalqwen",
        "config": Path(
            "configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
        ),
        "config_sha256": (
            "88a463fe48e5a884e99bf72cc3522a82031194f13cdd4b98966b160378e9a11e"
        ),
        "control_sha256": (
            "c38002af4d21356be2aaaef7f7b6873e025727ee4d3970d9ad5269e1ce3cd12f"
        ),
        "target_reserve": 0,
        "native_paths": 1,
        "native_positions": 1,
    },
    "q3": {
        "method_id": "q3_tallrec_generalqwen",
        "config": Path(
            "configs/methods/kuaisearch_motivation_v12_q3_tallrec_generalqwen.yaml"
        ),
        "config_sha256": (
            "ea8e0fb2d3421408cc51ecc216bfcfc7c7a0524e14a594d24009c9678235bd91"
        ),
        "control_sha256": (
            "9c9e1484d1fe910693e25be11b223fa19325181d3a3e5744ce82884432983093"
        ),
        "target_reserve": 2,
        "native_paths": 2,
        "native_positions": 2,
    },
}
REQUEST_METRICS = (
    "candidate_count",
    "history_tokens",
    "candidate_gap_tokens_mean",
    "prompt_tokens_mean",
    "natural_closest_distance_mean",
    "natural_center_distance_mean",
    "natural_farthest_distance_mean",
    "compression_span_to_center_ratio_mean",
    "compressed_center_distance_mean",
    "expanded_center_distance_mean",
    "compressed_absolute_distance_mean",
    "compression_absolute_closer_edge_fraction",
    "compression_absolute_equal_edge_fraction",
    "compression_absolute_farther_edge_fraction",
    "compression_negative_edge_fraction",
    "compression_zero_edge_fraction",
    "compression_nonpositive_edge_fraction",
    "readout_q_negative_phase_fraction",
    "history_k_beyond_readout_fraction",
    "rope_isotropic_expected_cosine",
    "rope_isotropic_relative_l2",
    "prompt_at_max_boundary_fraction",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d5_rope_geometry_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _assert_sha(root / DEEP_DIVE_PLAN, DEEP_DIVE_PLAN_SHA256, "deep-dive plan")
    _assert_sha(
        root / DEEP_DIVE_MANIFEST,
        DEEP_DIVE_MANIFEST_SHA256,
        "deep-dive manifest",
    )
    _assert_sha(
        root / CONTENT_DIR / "manifest.json",
        CONTENT_MANIFEST_SHA256,
        "content-neutral manifest",
    )
    _assert_sha(root / RECORDS_PATH, RECORDS_SHA256, "internal-dev records")
    _assert_sha(
        root / TOKENIZER_PATH / "tokenizer.json",
        TOKENIZER_SHA256,
        "frozen tokenizer",
    )
    content_manifest = _read_json(root / CONTENT_DIR / "manifest.json")
    if (
        content_manifest.get("qrels_read") is not False
        or content_manifest.get("model_scores_read") is not False
        or content_manifest.get("source_test_opened") is not False
        or content_manifest.get("target_records_sha256") != RECORDS_SHA256
    ):
        raise ValueError("content-neutral safety boundary differs")
    records = [
        sanitize_record_for_model(dict(row))
        for row in iter_jsonl(root / RECORDS_PATH)
    ]
    if len(records) != 8000 or sum(len(row.candidates) for row in records) != 160753:
        raise ValueError("frozen internal-dev population differs")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(root / TOKENIZER_PATH),
        local_files_only=True,
        trust_remote_code=False,
    )
    all_request_rows: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    by_model_request: dict[str, dict[str, dict[str, Any]]] = {}
    source_identity: dict[str, Any] = {}
    for model_key, spec in METHODS.items():
        config_path = root / spec["config"]
        _assert_sha(config_path, str(spec["config_sha256"]), f"{model_key} config")
        config = _read_yaml(config_path)
        if (
            config.get("method_id") != spec["method_id"]
            or config.get("model", {}).get("tokenizer_sha256") != TOKENIZER_SHA256
            or int(config.get("training", {}).get("history_budget", -1)) != 6
            or int(config.get("training", {}).get("max_length", -1)) != 2048
        ):
            raise ValueError(f"{model_key} frozen prompt config differs")
        declared = content_manifest["methods"][str(spec["method_id"])]
        control_path = root / str(declared["path"])
        expected_control_sha = str(spec["control_sha256"])
        _assert_sha(control_path, expected_control_sha, f"{model_key} controls")
        if (
            declared.get("sha256") != expected_control_sha
            or int(declared.get("eligible_requests", -1)) != 7254
            or int(declared.get("ineligible_requests", -1)) != 746
            or int(declared.get("max_prompt_length_after_target_reserve", -1))
            != 2048 - int(spec["target_reserve"])
        ):
            raise ValueError(f"{model_key} content-control declaration differs")
        controls = list(iter_jsonl(control_path))
        if len(controls) != len(records):
            raise ValueError(f"{model_key} control coverage differs")
        request_rows = _analyze_model(
            tokenizer,
            records,
            controls,
            model_key=model_key,
            method_id=str(spec["method_id"]),
            history_budget=6,
            prompt_max_length=2048 - int(spec["target_reserve"]),
            native_paths=int(spec["native_paths"]),
            native_positions=int(spec["native_positions"]),
        )
        by_model_request[model_key] = {
            str(row["request_id"]): row for row in request_rows
        }
        all_request_rows.extend(request_rows)
        model_summaries[model_key] = _group_summaries(request_rows)
        source_identity[model_key] = {
            "method_id": spec["method_id"],
            "config_path": spec["config"].as_posix(),
            "config_sha256": spec["config_sha256"],
            "control_path": str(declared["path"]),
            "control_sha256": expected_control_sha,
            "native_paths": int(spec["native_paths"]),
            "native_positions_per_path": int(spec["native_positions"]),
        }

    paired_rows = _paired_model_differences(
        by_model_request["q2"], by_model_request["q3"]
    )
    request_rows_path = output_dir / "request_position_geometry.jsonl"
    _write_jsonl_atomic(request_rows_path, all_request_rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d5_rope_position_geometry",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "causal_effect_claim": False,
        "interpretation_boundary": (
            "This audit measures the phase-distance support of the frozen D5 "
            "conditions. It cannot establish that RoPE causes a ranking effect, "
            "select a block/mode, or replace the shared post-qrels evaluator."
        ),
        "geometry_definition": {
            "natural_distance": "readout_query_position - history_key_position",
            "compression": "natural_distance - retained_history_span_length",
            "expansion": "natural_distance + retained_history_span_length",
            "negative_after_compression": (
                "compressed relative phase is negative; token indices and causal "
                "attention order remain unchanged"
            ),
            "weighting": (
                "candidate quantities are averaged within request; reported means "
                "and quantiles give every eligible request equal weight"
            ),
        },
        "registered_rope_blocks_zero_based": [13, 20, 27],
        "registered_modes": ["readout_q", "history_k", "paired_qk"],
        "source_identity": {
            "deep_dive_plan_sha256": DEEP_DIVE_PLAN_SHA256,
            "deep_dive_manifest_sha256": DEEP_DIVE_MANIFEST_SHA256,
            "content_manifest_sha256": CONTENT_MANIFEST_SHA256,
            "records_sha256": RECORDS_SHA256,
            "tokenizer_sha256": TOKENIZER_SHA256,
            "models": source_identity,
        },
        "model_summaries": model_summaries,
        "paired_q3_minus_q2": _group_summaries(paired_rows, paired=True),
        "request_rows_path": request_rows_path.relative_to(root).as_posix(),
        "request_rows_sha256": sha256_file(request_rows_path),
        "request_rows": len(all_request_rows),
        "eligible_requests_per_model": 7254,
        "ineligible_requests_per_model": 746,
        "qrels_read": False,
        "model_scores_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": " ".join(os.sys.argv),
    }
    metrics_path = output_dir / "metrics.json"
    _write_json_atomic(metrics_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "request_rows": len(all_request_rows),
                "metrics_sha256": sha256_file(metrics_path),
                "request_rows_sha256": sha256_file(request_rows_path),
            },
            sort_keys=True,
        )
    )


def _analyze_model(
    tokenizer: Any,
    records: Sequence[Any],
    controls: Sequence[Mapping[str, Any]],
    *,
    model_key: str,
    method_id: str,
    history_budget: int,
    prompt_max_length: int,
    native_paths: int,
    native_positions: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    eligible = 0
    for record, control in zip(records, controls):
        if str(control.get("request_id")) != record.request_id:
            raise ValueError(f"{model_key} record/control order differs")
        if control.get("eligible") is not True:
            if control.get("reason") != "no_visible_history":
                raise ValueError(f"{model_key} unexpected ineligibility reason")
            continue
        eligible += 1
        start = int(control["history_span_start"])
        end = int(control["history_span_end_exclusive"])
        history_tokens = int(control["history_span_tokens"])
        if end - start != history_tokens or history_tokens <= 0:
            raise ValueError(f"{model_key} invalid frozen history span")
        candidates = []
        for candidate in record.candidates:
            prompt = instrument_pointwise_prompt(
                tokenizer,
                method_id,
                record,
                candidate,
                history=record.history,
                history_budget=history_budget,
                max_length=prompt_max_length,
            )
            if end > prompt.candidate_readout:
                raise ValueError(f"{model_key} history is not before native readout")
            if native_positions == 1:
                positions = (prompt.candidate_readout,)
                sequence_length = len(prompt.token_ids)
            elif native_positions == 2:
                positions = (prompt.candidate_readout, prompt.candidate_readout + 1)
                sequence_length = len(prompt.token_ids) + 2
            else:
                raise AssertionError(native_positions)
            candidates.append(
                _candidate_geometry(
                    history_start=start,
                    history_end=end,
                    readout_positions=positions,
                    sequence_length=sequence_length,
                    candidate_start=prompt.candidate_start,
                    prompt_tokens=len(prompt.token_ids),
                    prompt_at_max_boundary=prompt.prompt_at_max_boundary,
                    native_paths=native_paths,
                )
            )
        request_row = _request_geometry(candidates)
        request_row.update(
            {
                "model_key": model_key,
                "method_id": method_id,
                "request_id": record.request_id,
                "normalized_query": normalize_query(record.query),
                "normalized_query_fold": _fold(normalize_query(record.query)),
                "history_start": start,
                "history_end_exclusive": end,
                "history_tokens": history_tokens,
                "native_paths": native_paths,
                "native_positions_per_path": native_positions,
            }
        )
        if int(request_row["candidate_count"]) != len(record.candidates):
            raise AssertionError("candidate geometry count drift")
        rows.append(request_row)
    if eligible != 7254 or len(rows) != 7254:
        raise ValueError(f"{model_key} eligible request count differs")
    return rows


def _candidate_geometry(
    *,
    history_start: int,
    history_end: int,
    readout_positions: Sequence[int],
    sequence_length: int,
    candidate_start: int,
    prompt_tokens: int,
    prompt_at_max_boundary: bool,
    native_paths: int,
) -> dict[str, float]:
    history_tokens = history_end - history_start
    if history_tokens <= 0 or not readout_positions or native_paths <= 0:
        raise ValueError("position geometry requires a non-empty span/path")
    negative_edges = 0
    zero_edges = 0
    absolute_closer_edges = 0
    absolute_equal_edges = 0
    compressed_absolute_sum = 0.0
    natural_edges = len(readout_positions) * history_tokens * native_paths
    center_distances = []
    closest_distances = []
    farthest_distances = []
    negative_q = 0
    history_k_beyond = 0
    for query_position in readout_positions:
        if not history_end <= query_position < sequence_length:
            raise ValueError("history/readout position order is invalid")
        closest = query_position - (history_end - 1)
        farthest = query_position - history_start
        center = query_position - (history_start + history_end - 1) / 2.0
        closest_distances.append(float(closest))
        farthest_distances.append(float(farthest))
        center_distances.append(float(center))
        negative_q += int(query_position - history_tokens < 0) * native_paths
        history_k_beyond += int(history_end - 1 + history_tokens > query_position) * native_paths
        first_negative_key = max(history_start, query_position - history_tokens + 1)
        negative_edges += max(0, history_end - first_negative_key) * native_paths
        zero_key = query_position - history_tokens
        zero_edges += int(history_start <= zero_key < history_end) * native_paths
        compressed_absolute_sum += (
            _absolute_integer_range_sum(
                closest - history_tokens, farthest - history_tokens
            )
            * native_paths
        )
        closer_threshold = history_tokens // 2 + 1
        first_closer_distance = max(closest, closer_threshold)
        absolute_closer_edges += (
            max(0, farthest - first_closer_distance + 1) * native_paths
        )
        if history_tokens % 2 == 0 and closest <= history_tokens // 2 <= farthest:
            absolute_equal_edges += native_paths
    query_rows = len(readout_positions) * native_paths
    center_mean = _mean(center_distances)
    rotation_cosine, rotation_l2 = _rope_rotation_geometry(history_tokens)
    absolute_farther_edges = (
        natural_edges - absolute_closer_edges - absolute_equal_edges
    )
    return {
        "history_tokens": float(history_tokens),
        "candidate_gap_tokens": float(candidate_start - history_end),
        "prompt_tokens": float(prompt_tokens),
        "natural_closest_distance": _mean(closest_distances),
        "natural_center_distance": center_mean,
        "natural_farthest_distance": _mean(farthest_distances),
        "compression_span_to_center_ratio": history_tokens / center_mean,
        "compressed_center_distance": center_mean - history_tokens,
        "expanded_center_distance": center_mean + history_tokens,
        "compressed_absolute_distance": compressed_absolute_sum / natural_edges,
        "compression_absolute_closer_edge_fraction": (
            absolute_closer_edges / natural_edges
        ),
        "compression_absolute_equal_edge_fraction": (
            absolute_equal_edges / natural_edges
        ),
        "compression_absolute_farther_edge_fraction": (
            absolute_farther_edges / natural_edges
        ),
        "compression_negative_edge_fraction": negative_edges / natural_edges,
        "compression_zero_edge_fraction": zero_edges / natural_edges,
        "compression_nonpositive_edge_fraction": (
            negative_edges + zero_edges
        )
        / natural_edges,
        "readout_q_negative_phase_fraction": negative_q / query_rows,
        "history_k_beyond_readout_fraction": history_k_beyond / query_rows,
        "rope_isotropic_expected_cosine": rotation_cosine,
        "rope_isotropic_relative_l2": rotation_l2,
        "prompt_at_max_boundary_fraction": float(bool(prompt_at_max_boundary)),
    }


def _request_geometry(candidates: Sequence[Mapping[str, float]]) -> dict[str, Any]:
    if not candidates:
        raise ValueError("request geometry requires candidates")
    history_lengths = {float(row["history_tokens"]) for row in candidates}
    if len(history_lengths) != 1:
        raise ValueError("candidate history spans differ within request")
    row: dict[str, Any] = {"candidate_count": len(candidates)}
    for metric in REQUEST_METRICS:
        if metric in {"candidate_count", "history_tokens"}:
            continue
        source = metric.removesuffix("_mean")
        row[metric] = _mean([float(candidate[source]) for candidate in candidates])
    return row


def _paired_model_differences(
    q2: Mapping[str, Mapping[str, Any]], q3: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if set(q2) != set(q3) or len(q2) != 7254:
        raise ValueError("Q2/Q3 eligible request identities differ")
    rows = []
    for request_id in q2:
        left = q2[request_id]
        right = q3[request_id]
        if (
            left["candidate_count"] != right["candidate_count"]
            or left["normalized_query"] != right["normalized_query"]
            or left["normalized_query_fold"] != right["normalized_query_fold"]
        ):
            raise ValueError("Q2/Q3 request geometry identity differs")
        row = {
            "request_id": request_id,
            "normalized_query_fold": int(left["normalized_query_fold"]),
        }
        for metric in REQUEST_METRICS:
            row[metric] = float(right[metric]) - float(left[metric])
        rows.append(row)
    return rows


def _group_summaries(
    rows: Sequence[Mapping[str, Any]], *, paired: bool = False
) -> dict[str, Any]:
    groups = {
        "all": list(rows),
        "fold0": [row for row in rows if int(row["normalized_query_fold"]) == 0],
        "fold1": [row for row in rows if int(row["normalized_query_fold"]) == 1],
    }
    result = {}
    for name, values in groups.items():
        result[name] = {
            "requests": len(values),
            "metrics": {
                metric: _summary([float(row[metric]) for row in values])
                for metric in REQUEST_METRICS
            },
        }
        if not paired:
            result[name]["candidate_rows"] = sum(
                int(row["candidate_count"]) for row in values
            )
    return result


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(float(value)) for value in values):
        raise ValueError("summary values must be finite and non-empty")
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": _nearest_rank(ordered, 0.10),
        "p25": _nearest_rank(ordered, 0.25),
        "median": _median(ordered),
        "mean": _mean(ordered),
        "p75": _nearest_rank(ordered, 0.75),
        "p90": _nearest_rank(ordered, 0.90),
        "p99": _nearest_rank(ordered, 0.99),
        "max": ordered[-1],
    }


def _absolute_integer_range_sum(start: int, end: int) -> int:
    """Return sum(abs(value) for value in range(start, end + 1))."""

    if end < start:
        raise ValueError("integer range is reversed")
    if start >= 0:
        return (start + end) * (end - start + 1) // 2
    if end <= 0:
        return ((-end) + (-start)) * (end - start + 1) // 2
    return (-start) * (-start + 1) // 2 + end * (end + 1) // 2


def _rope_rotation_geometry(
    delta: int, *, head_dim: int = 128, theta: float = 1_000_000.0
) -> tuple[float, float]:
    """Expected isotropic cosine/L2 of the exact registered RoPE rotation."""

    if delta < 0 or head_dim <= 0 or head_dim % 2 or theta <= 1.0:
        raise ValueError("invalid RoPE rotation geometry")
    cosines = [
        math.cos(delta / (theta ** (index / head_dim)))
        for index in range(0, head_dim, 2)
    ]
    expected_cosine = _mean(cosines)
    relative_l2 = math.sqrt(max(0.0, 2.0 - 2.0 * expected_cosine))
    return expected_cosine, relative_l2


def _nearest_rank(ordered: Sequence[float], quantile: float) -> float:
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def _median(ordered: Sequence[float]) -> float:
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2.0)


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return float(math.fsum(float(value) for value in values) / len(values))


def _fold(normalized_query: str) -> int:
    if not normalized_query:
        raise ValueError("normalized query is empty")
    return int(hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(), 16) % 2


def _assert_sha(path: Path, expected: str, label: str) -> None:
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"{label} hash drift: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"YAML object required: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    temporary.replace(path)


if __name__ == "__main__":
    main()
