#!/usr/bin/env python3
"""Verify the exact per-request common-score nullspace of Q2 objectives.

The analysis reconstructs the already frozen 288 train groups, uses only
train qrels, and evaluates the actual RankNet/ListNet implementations in
float64 at deterministic label-independent score anchors.  It is descriptive
algebra: no model score, dev/confirmation/test qrel, or outcome-based sample is
read, and no loss modification is proposed or trained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_ranker import (
    listwise_softmax_loss,
    load_training_groups,
    load_v12_ranker_config,
    pairwise_index_pairs,
    pairwise_ranknet_loss,
)
from myrec.mechanism.gradient_diagnostic import (
    CONTROLS,
    REQUESTS_PER_SURFACE,
    SELECTION_SEED,
    SURFACES,
    _load_train_gains,
    deterministic_label_shuffle,
    select_surface_training_groups,
)


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
STANDARDIZED_DIR = Path(
    "data/standardized/kuaisearch/full_confirm_preceding40k_v11"
)
RECORDS_TRAIN_SHA256 = (
    "3bcc86ff867ada54f90289153b1b1cf3f9277efdabba6672b73c865a2433beea"
)
QRELS_TRAIN_SHA256 = (
    "ffffd39ee1e3fe1290c0797c11730ae11945b3a36a60fa24cae9a11e2a311a5c"
)
Q2_CONFIG = Path(
    "configs/methods/kuaisearch_motivation_v12_q2_recranker_generalqwen.yaml"
)
Q2_CONFIG_SHA256 = (
    "88a463fe48e5a884e99bf72cc3522a82031194f13cdd4b98966b160378e9a11e"
)
SELECTION_MANIFEST = Path(
    "runs/20260718_kuaisearch_mech_d7_q2_objective_conflict_base_v1/"
    "selection_manifest.json"
)
SELECTION_MANIFEST_SHA256 = (
    "123099d64e75f7a6f16af1a5bb544253a17b7d732ea339610840070180b98811"
)
COMMON_SHIFT = 137.0
OBJECTIVES = ("pairwise_ranknet", "listwise_softmax", "combined_half_half")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument(
        "--output-dir",
        default="runs/20260718_kuaisearch_mech_d7_objective_nullspace_v1",
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for relative, expected, label in (
        (DEEP_DIVE_PLAN, DEEP_DIVE_PLAN_SHA256, "deep-dive plan"),
        (DEEP_DIVE_MANIFEST, DEEP_DIVE_MANIFEST_SHA256, "deep-dive manifest"),
        (STANDARDIZED_DIR / "records_train.jsonl", RECORDS_TRAIN_SHA256, "train records"),
        (STANDARDIZED_DIR / "qrels_train.jsonl", QRELS_TRAIN_SHA256, "train qrels"),
        (Q2_CONFIG, Q2_CONFIG_SHA256, "Q2 config"),
        (SELECTION_MANIFEST, SELECTION_MANIFEST_SHA256, "frozen D7 selection"),
    ):
        if _sha256_file(root / relative) != expected:
            raise ValueError(f"{label} hash drift")
    selection_manifest = _read_json(root / SELECTION_MANIFEST)
    if (
        selection_manifest.get("method_id") != "q2_recranker_generalqwen"
        or selection_manifest.get("requests_per_surface") != REQUESTS_PER_SURFACE
        or selection_manifest.get("selection_seed") != SELECTION_SEED
        or selection_manifest.get("frozen_hashes", {}).get("qrels_train_sha256")
        != QRELS_TRAIN_SHA256
        or selection_manifest.get("frozen_hashes", {}).get("records_train_sha256")
        != RECORDS_TRAIN_SHA256
    ):
        raise ValueError("frozen objective selection boundary differs")
    config = load_v12_ranker_config(root / Q2_CONFIG)
    if config["_config_sha256"] != Q2_CONFIG_SHA256:
        raise ValueError("Q2 objective config digest drift")
    gains = _load_train_gains(root / STANDARDIZED_DIR / "qrels_train.jsonl")
    groups, group_stats = load_training_groups(
        root / STANDARDIZED_DIR / "records_train.jsonl",
        root / STANDARDIZED_DIR / "qrels_train.jsonl",
        seed=int(config["training"]["seed"]),
        negatives_per_positive=int(config["training"]["negatives_per_positive"]),
        max_group_size=int(config["training"].get("list_size", 8)),
    )
    selected, regenerated = select_surface_training_groups(
        groups,
        gains,
        requests_per_surface=REQUESTS_PER_SURFACE,
        selection_seed=SELECTION_SEED,
    )
    for surface in SURFACES:
        expected = selection_manifest["surfaces"][surface]
        observed = regenerated["surfaces"][surface]
        for key in (
            "request_count",
            "request_ids",
            "request_ids_sha256",
            "training_groups_sha256",
        ):
            if observed[key] != expected[key]:
                raise ValueError(f"objective selection regeneration differs: {surface}.{key}")

    rows = []
    for surface in SURFACES:
        for group in selected[surface]:
            for control in CONTROLS:
                active = (
                    group
                    if control == "observed"
                    else deterministic_label_shuffle(group)[0]
                )
                score_values = _deterministic_scores(
                    active.record.request_id,
                    [str(candidate["item_id"]) for candidate in active.candidates],
                )
                objective_rows = {
                    name: _nullspace_metrics(score_values, active.gains, name)
                    for name in OBJECTIVES
                }
                rows.append(
                    {
                        "surface": surface,
                        "control": control,
                        "request_id_sha256": hashlib.sha256(
                            active.record.request_id.encode()
                        ).hexdigest(),
                        "candidates": len(active.candidates),
                        "distinct_gains": len(set(map(float, active.gains))),
                        "grade_different_pairs": len(pairwise_index_pairs(active.gains)),
                        "objectives": objective_rows,
                        "pairwise_listwise_score_gradient_cosine": _gradient_cosine(
                            score_values, active.gains
                        ),
                    }
                )
    expected_rows = len(SURFACES) * REQUESTS_PER_SURFACE * len(CONTROLS)
    if len(rows) != expected_rows:
        raise AssertionError("objective nullspace row coverage differs")
    cells = _aggregate_cells(rows)
    result = {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d7_objective_common_nullspace",
        "status": "completed",
        "descriptive_only": True,
        "confirmatory_family_membership": False,
        "method_id": "q2_recranker_generalqwen",
        "objectives": list(OBJECTIVES),
        "controls": list(CONTROLS),
        "surfaces": list(SURFACES),
        "groups_per_surface": REQUESTS_PER_SURFACE,
        "rows": len(rows),
        "common_shift": COMMON_SHIFT,
        "score_anchor": "sha256(request_id,candidate_item_id) mapped to [-1,1] independent of labels/model",
        "deep_dive_plan_sha256": DEEP_DIVE_PLAN_SHA256,
        "deep_dive_manifest_sha256": DEEP_DIVE_MANIFEST_SHA256,
        "q2_config_sha256": Q2_CONFIG_SHA256,
        "selection_manifest_path": SELECTION_MANIFEST.as_posix(),
        "selection_manifest_sha256": SELECTION_MANIFEST_SHA256,
        "selection_regenerated_exactly": True,
        "group_construction": group_stats,
        "train_qrels_read": True,
        "train_qrels_sha256": QRELS_TRAIN_SHA256,
        "qrels_read": True,
        "qrels_scope": "train_only",
        "model_scores_read": False,
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "cells": cells,
        "interpretation_boundary": (
            "An additive per-request common score is an exact loss null direction "
            "and cannot itself change within-request ranking. This audit can explain "
            "why common response is unconstrained, but it cannot attribute harmful "
            "candidate-relative direction or justify subtracting a common offset as "
            "a ranking fix. Curvature is evaluated at deterministic non-model scores."
        ),
        "command": "scripts/analyze_deep_dive_objective_nullspace.py",
    }
    output_path = output_dir / "objective_nullspace.json"
    _write_json_atomic(output_path, result)
    print(
        json.dumps(
            {
                "status": "completed",
                "rows": len(rows),
                "cells": len(cells),
                "sha256": _sha256_file(output_path),
            },
            sort_keys=True,
        )
    )


def _deterministic_scores(request_id: str, candidate_ids: Sequence[str]) -> np.ndarray:
    values = []
    for candidate_id in candidate_ids:
        digest = hashlib.sha256(
            f"objective-nullspace\0{request_id}\0{candidate_id}".encode()
        ).digest()
        integer = int.from_bytes(digest[:8], "big")
        values.append(2.0 * integer / float(2**64 - 1) - 1.0)
    array = np.asarray(values, dtype=np.float64)
    if len(array) < 2 or not np.isfinite(array).all():
        raise ValueError("objective score anchor differs")
    return array


def _objective_loss(scores: Any, gains: Sequence[float], objective: str) -> Any:
    if objective == "pairwise_ranknet":
        return pairwise_ranknet_loss(scores, gains)
    if objective == "listwise_softmax":
        return listwise_softmax_loss(scores, gains)
    if objective == "combined_half_half":
        return 0.5 * pairwise_ranknet_loss(scores, gains) + 0.5 * listwise_softmax_loss(scores, gains)
    raise ValueError(f"unknown objective={objective}")


def _nullspace_metrics(
    score_values: Sequence[float], gains: Sequence[float], objective: str
) -> dict[str, Any]:
    import torch

    scores = torch.tensor(score_values, dtype=torch.float64, requires_grad=True)
    loss = _objective_loss(scores, gains, objective)
    gradient = torch.autograd.grad(loss, scores, create_graph=False)[0]
    hessian = torch.autograd.functional.hessian(
        lambda value: _objective_loss(value, gains, objective),
        scores.detach(),
    )
    shifted = _objective_loss(scores.detach() + COMMON_SHIFT, gains, objective)
    ones = torch.ones_like(scores)
    hessian_ones = hessian @ ones
    eigenvalues = torch.linalg.eigvalsh(hessian).detach().cpu().numpy()
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    threshold = 1.0e-10 * scale
    positive = eigenvalues[eigenvalues > threshold]
    if not len(positive):
        raise ValueError("objective has no candidate-relative positive curvature")
    return {
        "loss": float(loss.detach().item()),
        "common_shift_loss_delta": float((shifted - loss.detach()).item()),
        "gradient_sum": float(gradient.sum().item()),
        "gradient_l2": float(gradient.norm().item()),
        "hessian_times_ones_l2": float(hessian_ones.norm().item()),
        "common_direction_rayleigh": float(
            (ones @ hessian_ones / (ones @ ones)).item()
        ),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "null_eigenvalue_count": int(np.sum(np.abs(eigenvalues) <= threshold)),
        "positive_eigenvalue_count": int(len(positive)),
        "minimum_positive_eigenvalue": float(positive.min()),
        "maximum_positive_eigenvalue": float(positive.max()),
        "positive_condition_number": float(positive.max() / positive.min()),
    }


def _gradient_cosine(score_values: Sequence[float], gains: Sequence[float]) -> float:
    import torch

    scores = torch.tensor(score_values, dtype=torch.float64, requires_grad=True)
    left = torch.autograd.grad(pairwise_ranknet_loss(scores, gains), scores, retain_graph=True)[0]
    right = torch.autograd.grad(listwise_softmax_loss(scores, gains), scores)[0]
    denominator = float(left.norm().item() * right.norm().item())
    if denominator <= 1.0e-30:
        raise ValueError("objective score gradient cosine is undefined")
    return float(torch.clamp((left @ right) / denominator, -1.0, 1.0).item())


def _aggregate_cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    cells = []
    for surface in SURFACES:
        for control in CONTROLS:
            selected = [
                row
                for row in rows
                if row["surface"] == surface and row["control"] == control
            ]
            if len(selected) != REQUESTS_PER_SURFACE:
                raise ValueError("objective nullspace cell coverage differs")
            cell = {
                "surface": surface,
                "control": control,
                "requests": len(selected),
                "candidate_count": _summary([row["candidates"] for row in selected]),
                "distinct_gain_count": _summary([row["distinct_gains"] for row in selected]),
                "grade_different_pairs": _summary(
                    [row["grade_different_pairs"] for row in selected]
                ),
                "pairwise_listwise_score_gradient_cosine": _summary(
                    [row["pairwise_listwise_score_gradient_cosine"] for row in selected]
                ),
                "objectives": {},
            }
            for objective in OBJECTIVES:
                values = [row["objectives"][objective] for row in selected]
                cell["objectives"][objective] = {
                    "maximum_absolute_common_shift_loss_delta": max(
                        abs(value["common_shift_loss_delta"]) for value in values
                    ),
                    "maximum_absolute_gradient_sum": max(
                        abs(value["gradient_sum"]) for value in values
                    ),
                    "maximum_hessian_times_ones_l2": max(
                        value["hessian_times_ones_l2"] for value in values
                    ),
                    "maximum_absolute_common_direction_rayleigh": max(
                        abs(value["common_direction_rayleigh"]) for value in values
                    ),
                    "null_eigenvalue_count": sorted(
                        {value["null_eigenvalue_count"] for value in values}
                    ),
                    "positive_eigenvalue_count": _summary(
                        [value["positive_eigenvalue_count"] for value in values]
                    ),
                    "minimum_positive_eigenvalue": _summary(
                        [value["minimum_positive_eigenvalue"] for value in values]
                    ),
                    "maximum_positive_eigenvalue": _summary(
                        [value["maximum_positive_eigenvalue"] for value in values]
                    ),
                    "positive_condition_number": _summary(
                        [value["positive_condition_number"] for value in values]
                    ),
                }
            cells.append(cell)
    return cells


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not array.size or not np.isfinite(array).all():
        raise ValueError("objective nullspace summary differs")
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "minimum": float(array.min()),
        "median": float(np.median(array)),
        "maximum": float(array.max()),
    }


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


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
