"""Registered 12-cell synthesis for Q2 RankNet/ListNet gradient conflict."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.mechanism.deep_dive_native_evaluator import (
    benjamini_hochberg,
    cluster_mean_inference,
)
from myrec.mechanism.gradient_diagnostic import REQUESTS_PER_SURFACE, SURFACES
from myrec.mechanism.objective_conflict_runtime import SUPPORTED_STATES
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


REGISTERED_FAMILY_SIZE = 12
COSINE_SESOI = 0.1


def synthesize_q2_objective_conflict(
    state_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    *,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    if set(state_dirs) != set(SUPPORTED_STATES):
        raise ValueError("objective-conflict synthesis requires base and final states")
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"objective synthesis output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = {
        state: _load_run(path, expected_state=state)
        for state, path in state_dirs.items()
    }
    selection_hashes = {
        _state_invariant_selection_sha256(Path(state_dirs[state]))
        for state in SUPPORTED_STATES
    }
    if len(selection_hashes) != 1:
        raise ValueError("objective conflict states used different frozen selections")
    family_rows = []
    results: dict[str, Any] = {}
    for state in SUPPORTED_STATES:
        metadata, rows = runs[state]
        results[state] = {}
        for surface in SURFACES:
            observed = {
                row["request_id"]: row
                for row in rows
                if row["surface"] == surface and row["control"] == "observed"
            }
            shuffled = {
                row["request_id"]: row
                for row in rows
                if row["surface"] == surface
                and row["control"] == "within_request_label_shuffle"
            }
            if set(observed) != set(shuffled) or len(observed) != REQUESTS_PER_SURFACE:
                raise ValueError("objective conflict observed/shuffle pairing is incomplete")
            request_ids = sorted(observed)
            clusters = np.asarray(
                [observed[request_id]["normalized_query"] for request_id in request_ids],
                dtype=np.str_,
            )
            observed_values = np.asarray(
                [observed[request_id]["cosine"] for request_id in request_ids],
                dtype=np.float64,
            )
            shuffle_values = np.asarray(
                [shuffled[request_id]["cosine"] for request_id in request_ids],
                dtype=np.float64,
            )
            endpoints = {
                "ranknet_listnet_cosine": observed_values,
                "observed_minus_label_shuffle_cosine": observed_values
                - shuffle_values,
            }
            surface_results = {}
            for endpoint, values in endpoints.items():
                inference = cluster_mean_inference(values, clusters)
                family_row = {
                    "state": state,
                    "surface": surface,
                    "endpoint": endpoint,
                    **inference,
                }
                family_rows.append(family_row)
                surface_results[endpoint] = family_row
            surface_results["descriptive"] = {
                "label_shuffle_cosine_mean": float(shuffle_values.mean()),
                "request_count": len(request_ids),
            }
            results[state][surface] = surface_results
    if len(family_rows) != REGISTERED_FAMILY_SIZE:
        raise AssertionError("objective conflict family size is not 12")
    q_values = benjamini_hochberg([row["two_sided_p"] for row in family_rows])
    for row, q_value in zip(family_rows, q_values):
        row["bh_q"] = float(q_value)
        lower, upper = map(float, row["ci95"])
        if row["endpoint"] == "ranknet_listnet_cosine":
            row["conflict_beyond_sesoi"] = upper < -COSINE_SESOI
            row["practical_equivalence_within_sesoi"] = (
                lower >= -COSINE_SESOI and upper <= COSINE_SESOI
            )
        else:
            row["observed_more_conflicted_than_shuffle"] = upper < 0.0
        row["bh_q_below_0.05"] = q_value < 0.05
    report = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d7_q2_objective_conflict",
        "family": {
            "definition": "state_x_surface_x_endpoint",
            "registered_size": REGISTERED_FAMILY_SIZE,
            "observed_size": len(family_rows),
            "multiple_testing": "benjamini_hochberg",
        },
        "cosine_sesoi": [-COSINE_SESOI, COSINE_SESOI],
        "bootstrap": {
            "cluster": "normalized_query",
            "samples": 5000,
            "seed": 20260715,
        },
        "selection_sha256": next(iter(selection_hashes)),
        "source_runs": {
            state: {
                "path": str(Path(state_dirs[state])),
                "metadata_sha256": sha256_file(Path(state_dirs[state]) / "metadata.json"),
                "per_request_sha256": metadata["per_request_sha256"],
            }
            for state, (metadata, _rows) in runs.items()
        },
        "family_rows": family_rows,
        "results": results,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": list(command or []),
        "status": "completed",
    }
    _write_json(output_dir / "metrics.json", report)
    return report


def _load_run(path: str | Path, *, expected_state: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    root = Path(path)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if (
        metadata.get("analysis_stage")
        != "transformer_deep_dive_d7_q2_objective_conflict"
        or metadata.get("state") != expected_state
        or metadata.get("status") != "completed"
        or metadata.get("result_eligible") is not True
        or metadata.get("dev_confirmation_test_qrels_read") is not False
        or metadata.get("per_request_sha256") != sha256_file(root / "per_request.jsonl")
    ):
        raise ValueError(f"objective conflict source run failed audit: {root}")
    rows = list(iter_jsonl(root / "per_request.jsonl"))
    if len(rows) != 2 * len(SURFACES) * REQUESTS_PER_SURFACE:
        raise ValueError("objective conflict source row count is incomplete")
    if any(
        not math.isfinite(float(row[key]))
        for row in rows
        for key in ("cosine", "left_norm", "right_norm")
    ):
        raise FloatingPointError("objective conflict source contains non-finite values")
    return metadata, rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _state_invariant_selection_sha256(root: Path) -> str:
    """Hash the frozen task selection while excluding only the state label."""

    selection = json.loads((root / "selection_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(selection, dict) or "state" not in selection:
        raise ValueError("objective conflict selection manifest lacks its state label")
    payload = {key: value for key, value in selection.items() if key != "state"}
    import hashlib

    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
