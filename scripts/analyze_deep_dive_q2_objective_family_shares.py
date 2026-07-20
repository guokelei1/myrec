#!/usr/bin/env python3
"""Describe Q2 RankNet/ListNet gradient-energy allocation by parameter family.

This is a full-cell descriptive audit of the already frozen D7 objective-conflict
bundles.  It does not add a confirmatory family or infer signed per-family
gradient conflict: family squared-norm shares contain no per-family dot product.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_FAMILIES = (
    "attention_k",
    "attention_o",
    "attention_q",
    "attention_v",
    "embedding_readout",
    "mlp_down",
    "mlp_gate",
    "mlp_up",
    "rmsnorm",
)
EXPECTED_CONTROLS = ("observed", "within_request_label_shuffle")
EXPECTED_SURFACES = ("recurrence", "strict_transfer", "other_overlap")
STATE_PATHS = {
    "base_initialization": Path(
        "runs/20260718_kuaisearch_mech_d7_q2_objective_conflict_base_v1"
    ),
    "frozen_final_checkpoint": Path(
        "runs/20260718_kuaisearch_mech_d7_q2_objective_conflict_final_v1"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output",
        default=(
            "runs/20260718_kuaisearch_mech_d7_q2_objective_conflict_synthesis_v1/"
            "parameter_family_share_analysis.json"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()

    rows_by_state: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, Any]] = {}
    for state, relative in STATE_PATHS.items():
        run_dir = root / relative
        metadata_path = run_dir / "metadata.json"
        rows_path = run_dir / "per_request.jsonl"
        metadata = _read_json(metadata_path)
        if metadata.get("status") != "completed":
            raise ValueError(f"source run is not completed: {relative}")
        if metadata.get("dev_confirmation_test_qrels_read") is not False:
            raise ValueError(f"source dev/confirmation/test qrels boundary failed: {relative}")
        if metadata.get("qrels_access") != "train_only_before_model_load_for_frozen_selection":
            raise ValueError(f"source train-only qrels declaration failed: {relative}")
        if metadata.get("source_test_opened") is not False:
            raise ValueError(f"source-test boundary failed: {relative}")
        rows = list(_iter_jsonl(rows_path))
        _audit_rows(rows, state)
        observed_sha = _sha256_file(rows_path)
        if metadata.get("per_request_sha256") != observed_sha:
            raise ValueError(f"per-request SHA differs from metadata: {relative}")
        rows_by_state[state] = rows
        sources[state] = {
            "run_dir": relative.as_posix(),
            "metadata_sha256": _sha256_file(metadata_path),
            "per_request_sha256": observed_sha,
            "rows": len(rows),
        }

    _audit_cross_state_identity(rows_by_state)
    cells = []
    for state in STATE_PATHS:
        for control in EXPECTED_CONTROLS:
            for surface in EXPECTED_SURFACES:
                rows = [
                    row
                    for row in rows_by_state[state]
                    if row["control"] == control and row["surface"] == surface
                ]
                cells.append(_summarize_cell(state, control, surface, rows))

    state_changes = []
    base_index = _index_rows(rows_by_state["base_initialization"])
    final_index = _index_rows(rows_by_state["frozen_final_checkpoint"])
    for control in EXPECTED_CONTROLS:
        for surface in EXPECTED_SURFACES:
            keys = sorted(
                key
                for key in base_index
                if key[0] == control and key[1] == surface
            )
            pairs = [(base_index[key], final_index[key]) for key in keys]
            state_changes.append(_summarize_state_change(control, surface, pairs))

    observed_cells = [cell for cell in cells if cell["control"] == "observed"]
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d7_q2_objective_family_shares",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "interpretation_boundary": (
            "Squared-gradient family shares diagnose allocation magnitude only. "
            "They cannot establish signed within-family RankNet/ListNet conflict, "
            "layer causality, or an effective optimizer update."
        ),
        "objective_labels": {"left": "pairwise_ranknet", "right": "listwise_softmax"},
        "share_definition": "family_squared_l2_over_full_gradient_squared_l2",
        "descriptive_absolute_difference_band": 0.05,
        "families": list(EXPECTED_FAMILIES),
        "sources": sources,
        "cells": cells,
        "paired_final_minus_base": state_changes,
        "overview": {
            "cells": len(cells),
            "requests_per_cell": 96,
            "observed_cells_with_any_mean_share_difference_abs_ge_0_05": sum(
                bool(cell["families_at_or_beyond_abs_0_05"]) for cell in observed_cells
            ),
            "maximum_observed_mean_share_difference_abs": max(
                max(abs(value) for value in cell["mean_share_difference"].values())
                for cell in observed_cells
            ),
            "maximum_observed_request_mean_total_variation": max(
                cell["request_share_total_variation"]["mean"] for cell in observed_cells
            ),
        },
        "qrels_access": "train_only_in_frozen_source_bundles",
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "command": " ".join(os.sys.argv),
    }
    output = root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output, result)
    print(json.dumps(result["overview"], ensure_ascii=False, sort_keys=True))


def _audit_rows(rows: Sequence[Mapping[str, Any]], state: str) -> None:
    if len(rows) != 576:
        raise ValueError(f"{state} expected 576 rows, observed {len(rows)}")
    counts: defaultdict[tuple[str, str], int] = defaultdict(int)
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        control = str(row.get("control"))
        surface = str(row.get("surface"))
        request_id = str(row.get("request_id"))
        key = (control, surface, request_id)
        if key in seen:
            raise ValueError(f"duplicate row in {state}: {key}")
        seen.add(key)
        counts[(control, surface)] += 1
        for share_key in ("left_family_share", "right_family_share"):
            shares = row.get(share_key)
            if not isinstance(shares, Mapping) or tuple(sorted(shares)) != EXPECTED_FAMILIES:
                raise ValueError(f"family boundary drift in {state}:{share_key}")
            values = [float(shares[family]) for family in EXPECTED_FAMILIES]
            if not all(math.isfinite(value) and value >= 0.0 for value in values):
                raise ValueError(f"invalid family share in {state}:{share_key}")
            if abs(sum(values) - 1.0) > 1e-9:
                raise ValueError(f"family shares do not sum to one in {state}:{share_key}")
        for scalar in ("cosine", "left_norm", "right_norm"):
            if not math.isfinite(float(row[scalar])):
                raise ValueError(f"non-finite {scalar} in {state}")
    expected = {
        (control, surface): 96
        for control in EXPECTED_CONTROLS
        for surface in EXPECTED_SURFACES
    }
    if dict(counts) != expected:
        raise ValueError(f"cell coverage differs in {state}: {dict(counts)}")


def _audit_cross_state_identity(rows_by_state: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    base = _index_rows(rows_by_state["base_initialization"])
    final = _index_rows(rows_by_state["frozen_final_checkpoint"])
    if set(base) != set(final):
        raise ValueError("base/final request cells differ")
    for key in base:
        if base[key]["normalized_query"] != final[key]["normalized_query"]:
            raise ValueError(f"normalized-query identity differs across states: {key}")


def _index_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    return {
        (str(row["control"]), str(row["surface"]), str(row["request_id"])): row
        for row in rows
    }


def _summarize_cell(
    state: str,
    control: str,
    surface: str,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = {
        family: _mean([float(row["left_family_share"][family]) for row in rows])
        for family in EXPECTED_FAMILIES
    }
    right = {
        family: _mean([float(row["right_family_share"][family]) for row in rows])
        for family in EXPECTED_FAMILIES
    }
    difference = {family: left[family] - right[family] for family in EXPECTED_FAMILIES}
    tv = [
        0.5
        * sum(
            abs(float(row["left_family_share"][family]) - float(row["right_family_share"][family]))
            for family in EXPECTED_FAMILIES
        )
        for row in rows
    ]
    fold_differences = {}
    for fold in (0, 1):
        fold_rows = [row for row in rows if _fold(str(row["normalized_query"])) == fold]
        fold_differences[str(fold)] = {
            "requests": len(fold_rows),
            "mean_share_difference": {
                family: _mean(
                    [
                        float(row["left_family_share"][family])
                        - float(row["right_family_share"][family])
                        for row in fold_rows
                    ]
                )
                for family in EXPECTED_FAMILIES
            },
        }
    return {
        "state": state,
        "control": control,
        "surface": surface,
        "requests": len(rows),
        "normalized_query_clusters": len({row["normalized_query"] for row in rows}),
        "mean_cosine": _mean([float(row["cosine"]) for row in rows]),
        "mean_ranknet_family_share": left,
        "mean_listnet_family_share": right,
        "mean_share_difference": difference,
        "families_at_or_beyond_abs_0_05": [
            family for family in EXPECTED_FAMILIES if abs(difference[family]) >= 0.05
        ],
        "request_share_total_variation": {
            "mean": _mean(tv),
            "median": float(median(tv)),
            "p90": _quantile(tv, 0.90),
            "max": max(tv),
            "pearson_with_global_cosine": _pearson(
                tv, [float(row["cosine"]) for row in rows]
            ),
        },
        "folds": fold_differences,
    }


def _summarize_state_change(
    control: str,
    surface: str,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> dict[str, Any]:
    return {
        "control": control,
        "surface": surface,
        "requests": len(pairs),
        "mean_global_cosine_change": _mean(
            [float(final["cosine"]) - float(base["cosine"]) for base, final in pairs]
        ),
        "ranknet_family_share_change": {
            family: _mean(
                [
                    float(final["left_family_share"][family])
                    - float(base["left_family_share"][family])
                    for base, final in pairs
                ]
            )
            for family in EXPECTED_FAMILIES
        },
        "listnet_family_share_change": {
            family: _mean(
                [
                    float(final["right_family_share"][family])
                    - float(base["right_family_share"][family])
                    for base, final in pairs
                ]
            )
            for family in EXPECTED_FAMILIES
        },
    }


def _fold(normalized_query: str) -> int:
    if not normalized_query:
        raise ValueError("normalized query is empty")
    return int(hashlib.sha256(normalized_query.encode("utf-8")).hexdigest(), 16) % 2


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    return float(sum(values) / len(values))


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    left_mean = _mean(left)
    right_mean = _mean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    numerator = sum(a * b for a, b in zip(left_centered, right_centered, strict=True))
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    return float(numerator / denominator) if denominator > 0.0 else None


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError(f"expected JSON object row: {path}")
                yield value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
