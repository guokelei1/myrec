"""Shared-metric admission and evaluation for the Q2 matched control.

All four score bundles are audited without qrels first.  Cross-checkpoint
comparison is admitted only when the original-mixture and surface-balanced
checkpoints share an exact base/config/recipe signature and differ only in
their registered sampling declarations.  Graded dev labels are then opened
solely by :func:`myrec.mechanism.evaluator.evaluate_mechanism_probe`.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_ranker import (
    _git_revision,
    load_v12_ranker_config,
)
from myrec.mechanism.evaluator import (
    audit_mechanism_score_bundle,
    evaluate_mechanism_probe,
)
from myrec.mechanism.matched_control_scorer import (
    EXPECTED_REQUESTS,
    EXPECTED_SCORE_ROWS,
    validate_matched_training_checkpoint_metadata,
)
from myrec.mechanism.matched_training_control import (
    CONDITIONS,
    METHOD_ID,
    ROLE,
    _canonical_sha256,
    _read_json,
)
from myrec.utils.hashing import sha256_file


def admit_q2_matched_control_score_runs(
    *,
    original_full_run_id: str,
    original_null_run_id: str,
    balanced_full_run_id: str,
    balanced_null_run_id: str,
    standardized_dir: str | Path,
    candidate_manifest_path: str | Path | None = None,
    runs_dir: str | Path = "runs",
) -> dict[str, Any]:
    """Perform the complete four-run, qrels-free admission boundary."""

    runs_dir = Path(runs_dir)
    standardized_dir = Path(standardized_dir)
    pair_ids = {
        "original_mixture": {
            "full": original_full_run_id,
            "null": original_null_run_id,
        },
        "surface_balanced": {
            "full": balanced_full_run_id,
            "null": balanced_null_run_id,
        },
    }
    pair_audits = {}
    score_metadata: dict[str, dict[str, dict[str, Any]]] = {}
    training_metadata: dict[str, dict[str, Any]] = {}
    scorer_identities = []
    normalized_scoring_signatures = []
    for sampling_condition in CONDITIONS:
        run_ids = pair_ids[sampling_condition]
        audit = audit_mechanism_score_bundle(
            treatment_run_id=run_ids["full"],
            control_run_id=run_ids["null"],
            standardized_dir=standardized_dir,
            candidate_manifest_path=candidate_manifest_path,
            split="dev",
            runs_dir=runs_dir,
        )
        if int(audit.get("num_requests", -1)) != EXPECTED_REQUESTS or int(
            audit.get("score_rows_per_run", -1)
        ) != EXPECTED_SCORE_ROWS:
            raise ValueError(
                "matched-control score pair must cover exactly 8000/160753"
            )
        pair_audits[sampling_condition] = audit
        score_metadata[sampling_condition] = {}
        pair_training_hash = None
        pair_checkpoint_id = None
        for history_condition in ("full", "null"):
            run_id = run_ids[history_condition]
            metadata_path = runs_dir / run_id / "metadata.json"
            metadata = _read_json(metadata_path)
            _validate_score_metadata(
                metadata,
                run_id=run_id,
                sampling_condition=sampling_condition,
                history_condition=history_condition,
            )
            score_metadata[sampling_condition][history_condition] = metadata
            scorer_identities.append(
                metadata["matched_control_scorer_implementation_identity"]
            )
            normalized_signature = dict(metadata["base_scoring_signature"])
            normalized_signature.pop("checkpoint_id", None)
            normalized_scoring_signatures.append(normalized_signature)
            declaration = metadata["matched_training_control"]
            observed_training_hash = str(declaration["training_metadata_sha256"])
            observed_checkpoint_id = str(metadata["checkpoint_id"])
            if pair_training_hash is None:
                pair_training_hash = observed_training_hash
                pair_checkpoint_id = observed_checkpoint_id
            elif pair_training_hash != observed_training_hash or (
                pair_checkpoint_id != observed_checkpoint_id
            ):
                raise ValueError(
                    f"{sampling_condition} full/null do not share one checkpoint"
                )
        if score_metadata[sampling_condition]["full"][
            "matched_training_control"
        ] != score_metadata[sampling_condition]["null"][
            "matched_training_control"
        ]:
            raise ValueError(
                f"{sampling_condition} full/null training provenance differs"
            )
        full_metadata = score_metadata[sampling_condition]["full"]
        declaration = full_metadata["matched_training_control"]
        training_path = _resolve_declared_path(
            str(declaration["training_metadata_path"])
        )
        if sha256_file(training_path) != declaration["training_metadata_sha256"]:
            raise ValueError("training metadata differs from score-run binding")
        train_metadata = _read_json(training_path)
        config_path = _resolve_declared_path(str(full_metadata["config_path"]))
        config = load_v12_ranker_config(config_path)
        validate_matched_training_checkpoint_metadata(
            train_metadata,
            sampling_condition=sampling_condition,
            config=config,
            training_metadata_path=training_path,
        )
        if train_metadata["checkpoint_id"] != full_metadata["checkpoint_id"]:
            raise ValueError("score and training checkpoint identity differ")
        if train_metadata["matched_recipe"] != declaration["matched_recipe"]:
            raise ValueError("score run copied a different matched recipe")
        if train_metadata["sampling"] != declaration["sampling"]:
            raise ValueError("score run copied a different sampling declaration")
        training_metadata[sampling_condition] = {
            "metadata": train_metadata,
            "metadata_path": str(training_path),
            "metadata_sha256": sha256_file(training_path),
        }

    fixed_recipe_admission = validate_cross_checkpoint_fixed_recipe(
        training_metadata["original_mixture"]["metadata"],
        training_metadata["surface_balanced"]["metadata"],
    )
    original_path = Path(training_metadata["original_mixture"]["metadata_path"])
    balanced_path = Path(training_metadata["surface_balanced"]["metadata_path"])
    if original_path.resolve() == balanced_path.resolve():
        raise ValueError("sampling conditions must use independent checkpoint roots")
    if any(identity != scorer_identities[0] for identity in scorer_identities[1:]):
        raise ValueError("four score bundles use different scorer implementations")
    if any(
        signature != normalized_scoring_signatures[0]
        for signature in normalized_scoring_signatures[1:]
    ):
        raise ValueError(
            "cross-checkpoint scoring recipes differ beyond checkpoint identity"
        )

    return {
        "schema_version": 1,
        "analysis_type": "q2_matched_control_pre_qrels_admission",
        "checks": {
            "base_config_recipe_equal_except_sampling": True,
            "complete_finite_score_coverage": True,
            "full_null_checkpoint_identity_equal_within_condition": True,
            "independent_checkpoint_roots": True,
            "qrels_read_false_attested": True,
            "registered_sampling_conditions_exact": True,
            "scorer_implementation_equal": True,
            "scoring_recipe_equal_except_checkpoint": True,
            "score_training_provenance_exact": True,
        },
        "fixed_recipe_admission": fixed_recipe_admission,
        "generated_at": _utc_now(),
        "pair_audits": pair_audits,
        "qrels_read": False,
        "run_ids": pair_ids,
        "status": "passed",
        "training_checkpoints": {
            condition: {
                "checkpoint_id": training_metadata[condition]["metadata"][
                    "checkpoint_id"
                ],
                "metadata_path": training_metadata[condition]["metadata_path"],
                "metadata_sha256": training_metadata[condition]["metadata_sha256"],
                "sampling_selection_sha256": training_metadata[condition][
                    "metadata"
                ]["sampling"]["selection_sha256"],
            }
            for condition in CONDITIONS
        },
    }


def validate_cross_checkpoint_fixed_recipe(
    original: Mapping[str, Any], balanced: Mapping[str, Any]
) -> dict[str, Any]:
    """Hand-auditable equality check with sampling as the sole free factor."""

    expected_conditions = {
        str(original.get("condition")),
        str(balanced.get("condition")),
    }
    if expected_conditions != set(CONDITIONS):
        raise ValueError("cross-checkpoint conditions are not the registered pair")
    for metadata in (original, balanced):
        if metadata.get("method_id") != METHOD_ID or metadata.get("role") != ROLE:
            raise ValueError("cross-checkpoint method/role mismatch")
        recipe = metadata.get("matched_recipe")
        if not isinstance(recipe, Mapping) or _canonical_sha256(recipe) != metadata.get(
            "matched_recipe_sha256"
        ):
            raise ValueError("cross-checkpoint matched recipe hash mismatch")
        sampling = metadata.get("sampling")
        if not isinstance(sampling, Mapping) or sampling.get("condition") != metadata.get(
            "condition"
        ):
            raise ValueError("cross-checkpoint sampling declaration mismatch")
    if original["matched_recipe"] != balanced["matched_recipe"]:
        raise ValueError(
            "base/config/optimizer/LR/visible-fields/steps differ across checkpoints"
        )
    if original["matched_recipe_sha256"] != balanced["matched_recipe_sha256"]:
        raise ValueError("matched recipe digests differ across checkpoints")
    if original.get("config_sha256") != balanced.get("config_sha256"):
        raise ValueError("config hash differs across checkpoints")
    original_selection = original["sampling"].get("selection_sha256")
    balanced_selection = balanced["sampling"].get("selection_sha256")
    if not original_selection or not balanced_selection:
        raise ValueError("sampling selection hashes are required")
    if original_selection == balanced_selection:
        raise ValueError("registered sampling conditions share one selection hash")
    recipe = original["matched_recipe"]
    return {
        "allowed_cross_checkpoint_difference": [
            "sampling.condition",
            "sampling.selection_sha256",
            "sampling.selected_surface_counts",
            "checkpoint_and_run_output_identity",
            "timestamps_elapsed_loss_and_resume_lineage",
        ],
        "backbone_initialization": recipe["backbone_initialization"],
        "config_sha256": original["config_sha256"],
        "fixed_fields_verified": [
            "backbone_initialization",
            "optimizer",
            "optimizer.learning_rate",
            "input_fields_used",
            "group_exposures",
            "optimizer_steps",
            "objective",
            "scheduler",
            "surface_classifier",
            "runtime_contract",
        ],
        "matched_recipe_sha256": original["matched_recipe_sha256"],
        "sampling_selection_sha256": {
            "original_mixture": original_selection,
            "surface_balanced": balanced_selection,
        },
        "status": "passed",
    }


def evaluate_q2_matched_control(
    *,
    analysis_run_id: str,
    original_full_run_id: str,
    original_null_run_id: str,
    balanced_full_run_id: str,
    balanced_null_run_id: str,
    standardized_dir: str | Path,
    candidate_manifest_path: str | Path | None = None,
    runs_dir: str | Path = "runs",
    dev_eval_log_path: str | Path = "reports/dev_eval_log.jsonl",
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Admit four bundles, invoke the shared evaluator twice, compare effects."""

    recorded_command = list(sys.argv if command is None else command)
    if not recorded_command:
        raise ValueError("matched-control evaluator command must be non-empty")
    code_revision = _git_revision()
    implementation_identity = matched_control_evaluator_implementation_identity()
    runs_dir = Path(runs_dir)
    analysis_dir = runs_dir / analysis_run_id
    if analysis_dir.exists():
        raise FileExistsError(f"analysis run already exists: {analysis_dir}")
    admission = admit_q2_matched_control_score_runs(
        original_full_run_id=original_full_run_id,
        original_null_run_id=original_null_run_id,
        balanced_full_run_id=balanced_full_run_id,
        balanced_null_run_id=balanced_null_run_id,
        standardized_dir=standardized_dir,
        candidate_manifest_path=candidate_manifest_path,
        runs_dir=runs_dir,
    )
    # This artifact is persisted before either shared evaluator is allowed to
    # open qrels_dev.jsonl.
    analysis_dir.mkdir(parents=True, exist_ok=False)
    admission_path = analysis_dir / "pre_qrels_cross_checkpoint_admission.json"
    _write_json_atomic(admission_path, admission)

    original_analysis_id = f"{analysis_run_id}_original_mixture_full_vs_null"
    balanced_analysis_id = f"{analysis_run_id}_surface_balanced_full_vs_null"
    for pair_analysis_id in (original_analysis_id, balanced_analysis_id):
        if (runs_dir / pair_analysis_id).exists():
            raise FileExistsError(
                f"shared pair analysis run already exists: {pair_analysis_id}"
            )
    original_metrics = evaluate_mechanism_probe(
        analysis_run_id=original_analysis_id,
        treatment_run_id=original_full_run_id,
        control_run_id=original_null_run_id,
        standardized_dir=standardized_dir,
        candidate_manifest_path=candidate_manifest_path,
        split="dev",
        runs_dir=runs_dir,
        dev_eval_log_path=dev_eval_log_path,
    )
    balanced_metrics = evaluate_mechanism_probe(
        analysis_run_id=balanced_analysis_id,
        treatment_run_id=balanced_full_run_id,
        control_run_id=balanced_null_run_id,
        standardized_dir=standardized_dir,
        candidate_manifest_path=candidate_manifest_path,
        split="dev",
        runs_dir=runs_dir,
        dev_eval_log_path=dev_eval_log_path,
    )
    _validate_shared_pair_metrics(
        original_metrics,
        analysis_run_id=original_analysis_id,
        treatment_run_id=original_full_run_id,
        control_run_id=original_null_run_id,
    )
    _validate_shared_pair_metrics(
        balanced_metrics,
        analysis_run_id=balanced_analysis_id,
        treatment_run_id=balanced_full_run_id,
        control_run_id=balanced_null_run_id,
    )
    _validate_cross_pair_metrics(original_metrics, balanced_metrics)
    pair_analysis_artifacts = {
        "original_mixture": _pair_analysis_artifact_lineage(
            runs_dir=runs_dir,
            analysis_run_id=original_analysis_id,
            metrics=original_metrics,
        ),
        "surface_balanced": _pair_analysis_artifact_lineage(
            runs_dir=runs_dir,
            analysis_run_id=balanced_analysis_id,
            metrics=balanced_metrics,
        ),
    }
    cross_surface = {}
    if set(original_metrics["surfaces"]) != set(balanced_metrics["surfaces"]):
        raise ValueError("shared evaluator surface coverage differs")
    for surface in original_metrics["surfaces"]:
        left = original_metrics["surfaces"][surface]
        right = balanced_metrics["surfaces"][surface]
        if left.get("num_requests") != right.get("num_requests"):
            raise ValueError("shared evaluator surface request counts differ")
        cross_surface[surface] = {
            "balanced_minus_original_history_response_ndcg@10": _difference(
                right.get("mean_treatment_minus_control_ndcg@10"),
                left.get("mean_treatment_minus_control_ndcg@10"),
            ),
            "balanced_minus_original_target_margin_change": _difference(
                right.get("mean_target_margin_change"),
                left.get("mean_target_margin_change"),
            ),
            "num_requests": left.get("num_requests"),
            "original_mixture": {
                "history_response_ndcg@10": left.get(
                    "mean_treatment_minus_control_ndcg@10"
                ),
                "target_margin_change": left.get("mean_target_margin_change"),
            },
            "surface_balanced": {
                "history_response_ndcg@10": right.get(
                    "mean_treatment_minus_control_ndcg@10"
                ),
                "target_margin_change": right.get("mean_target_margin_change"),
            },
        }
    metrics = {
        "schema_version": 1,
        "analysis_run_id": analysis_run_id,
        "analysis_type": "q2_matched_training_control_cross_checkpoint",
        "balanced_minus_original_history_response_ndcg@10": _difference(
            balanced_metrics["mean_treatment_minus_control_ndcg@10"],
            original_metrics["mean_treatment_minus_control_ndcg@10"],
        ),
        "balanced_minus_original_target_margin_change": _difference(
            balanced_metrics["mean_target_margin_change"],
            original_metrics["mean_target_margin_change"],
        ),
        "code_revision": code_revision,
        "command": recorded_command,
        "fixed_recipe_admission": admission["fixed_recipe_admission"],
        "generated_at": _utc_now(),
        "label_mode": "graded",
        "matched_control_evaluator_implementation_identity": implementation_identity,
        "method_id": METHOD_ID,
        "original_mixture_analysis_run_id": original_analysis_id,
        "original_mixture_metrics": original_metrics,
        "pair_analysis_artifacts": pair_analysis_artifacts,
        "pre_qrels_admission_path": str(admission_path),
        "pre_qrels_admission_sha256": sha256_file(admission_path),
        "qrels_read": True,
        "qrels_sha256": original_metrics["qrels_sha256"],
        "request_count": EXPECTED_REQUESTS,
        "role": ROLE,
        "score_rows_per_run": EXPECTED_SCORE_ROWS,
        "split": "dev",
        "status": "completed",
        "surface_balanced_analysis_run_id": balanced_analysis_id,
        "surface_balanced_metrics": balanced_metrics,
        "surfaces": cross_surface,
    }
    metrics_path = analysis_dir / "metrics.json"
    _write_json_atomic(metrics_path, metrics)
    metadata = {
        "schema_version": 1,
        "analysis_run_id": analysis_run_id,
        "analysis_type": metrics["analysis_type"],
        "code_revision": code_revision,
        "command": recorded_command,
        "dev_eval_log_path": str(dev_eval_log_path),
        "generated_at": metrics["generated_at"],
        "label_mode": metrics["label_mode"],
        "matched_control_evaluator_implementation_identity": (
            implementation_identity
        ),
        "method_id": metrics["method_id"],
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "pair_analysis_artifacts": pair_analysis_artifacts,
        "pre_qrels_admission_path": str(admission_path),
        "pre_qrels_admission_sha256": sha256_file(admission_path),
        "qrels_read": metrics["qrels_read"],
        "qrels_reader": "myrec.mechanism.evaluator.evaluate_mechanism_probe",
        "qrels_sha256": metrics["qrels_sha256"],
        "request_count": metrics["request_count"],
        "role": ROLE,
        "score_rows_per_run": metrics["score_rows_per_run"],
        "split": metrics["split"],
        "status": metrics["status"],
    }
    _append_cross_checkpoint_ledger(
        dev_eval_log_path, metrics=metrics, metrics_path=metrics_path
    )
    # ``metadata.json`` is the final completion marker.  A failed or
    # non-durable ledger append must leave the top-level analysis incomplete.
    _write_json_atomic(analysis_dir / "metadata.json", metadata)
    return metrics


def matched_control_evaluator_implementation_identity() -> dict[str, Any]:
    """Bind results to the matched wrapper, shared evaluator, and CLI."""

    root = Path(__file__).resolve().parents[3]
    paths = {
        "scripts/evaluate_q2_matched_control.py": (
            root / "scripts/evaluate_q2_matched_control.py"
        ),
        "src/myrec/mechanism/evaluator.py": (
            root / "src/myrec/mechanism/evaluator.py"
        ),
        "src/myrec/mechanism/matched_control_evaluator.py": Path(__file__).resolve(),
    }
    files = []
    for relative, path in sorted(paths.items()):
        if not path.is_file():
            raise FileNotFoundError(f"missing matched-control evaluator: {path}")
        files.append({"path": relative, "sha256": sha256_file(path)})
    return {"digest": _canonical_sha256(files), "files": files}


def _validate_shared_pair_metrics(
    metrics: Mapping[str, Any],
    *,
    analysis_run_id: str,
    treatment_run_id: str,
    control_run_id: str,
) -> None:
    expected = {
        "analysis_run_id": analysis_run_id,
        "analysis_type": "motivation_mechanism_paired_probe",
        "control_run_id": control_run_id,
        "label_mode": "graded",
        "method_id": METHOD_ID,
        "num_requests": EXPECTED_REQUESTS,
        "split": "dev",
        "treatment_run_id": treatment_run_id,
    }
    for key, value in expected.items():
        if metrics.get(key) != value:
            raise ValueError(
                f"shared pair metrics {analysis_run_id} {key} mismatch"
            )
    for key in (
        "candidate_manifest_sha256",
        "qrels_sha256",
        "request_manifest_sha256",
    ):
        _require_sha256(metrics.get(key), field=f"{analysis_run_id}.{key}")
    for key in ("dataset_id", "dataset_version"):
        value = metrics.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"shared pair metrics {analysis_run_id} {key} is empty"
            )
    surfaces = metrics.get("surfaces")
    if not isinstance(surfaces, Mapping):
        raise ValueError(f"shared pair metrics {analysis_run_id} lacks surfaces")
    all_surface = surfaces.get("all")
    if not isinstance(all_surface, Mapping) or all_surface.get(
        "num_requests"
    ) != EXPECTED_REQUESTS:
        raise ValueError(
            f"shared pair metrics {analysis_run_id} all-surface request count mismatch"
        )


def _validate_cross_pair_metrics(
    original: Mapping[str, Any], balanced: Mapping[str, Any]
) -> None:
    for key in (
        "candidate_manifest_sha256",
        "dataset_id",
        "dataset_version",
        "label_mode",
        "method_id",
        "num_requests",
        "qrels_sha256",
        "request_manifest_sha256",
        "split",
    ):
        if original.get(key) != balanced.get(key):
            raise ValueError(f"shared pair metrics cross-checkpoint {key} mismatch")


def _pair_analysis_artifact_lineage(
    *,
    runs_dir: Path,
    analysis_run_id: str,
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    analysis_dir = runs_dir / analysis_run_id
    metrics_path = analysis_dir / "metrics.json"
    metadata_path = analysis_dir / "metadata.json"
    if not metrics_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            f"shared pair analysis artifacts are incomplete: {analysis_run_id}"
        )
    persisted_metrics = _read_json(metrics_path)
    if persisted_metrics != dict(metrics):
        raise ValueError(
            f"shared pair analysis returned/persisted metrics differ: {analysis_run_id}"
        )
    metadata = _read_json(metadata_path)
    expected_metadata = {
        "analysis_run_id": analysis_run_id,
        "analysis_type": "motivation_mechanism_paired_probe",
        "label_mode": "graded",
        "metrics_path": str(metrics_path),
        "qrels_read": True,
        "qrels_sha256": metrics["qrels_sha256"],
        "split": "dev",
    }
    for key, value in expected_metadata.items():
        if metadata.get(key) != value:
            raise ValueError(
                f"shared pair metadata {analysis_run_id} {key} mismatch"
            )
    invariants = metadata.get("invariants")
    if not isinstance(invariants, Mapping) or invariants.get("method_id") != METHOD_ID:
        raise ValueError(
            f"shared pair metadata {analysis_run_id} method_id mismatch"
        )
    return {
        "analysis_run_id": analysis_run_id,
        "metadata_path": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
    }


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value.lower()
    ):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return value


def _validate_score_metadata(
    metadata: Mapping[str, Any],
    *,
    run_id: str,
    sampling_condition: str,
    history_condition: str,
) -> None:
    expected = {
        "condition_id": f"{sampling_condition}__{history_condition}",
        "evidence_mode": "mechanism_diagnostic",
        "history_condition": history_condition,
        "method_id": METHOD_ID,
        "qrels_read": False,
        "request_count": EXPECTED_REQUESTS,
        "result_eligible": True,
        "run_id": run_id,
        "sampling_condition": sampling_condition,
        "score_rows": EXPECTED_SCORE_ROWS,
        "split": "dev",
        "status": "completed",
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"score-run admission failed for {run_id}: {key}")
    declaration = metadata.get("matched_training_control")
    if not isinstance(declaration, Mapping):
        raise ValueError(f"score run {run_id} lacks matched-control provenance")
    if declaration.get("role") != ROLE or declaration.get(
        "sampling_condition"
    ) != sampling_condition:
        raise ValueError(f"score run {run_id} matched-control role/sampling mismatch")
    recipe = declaration.get("matched_recipe")
    if not isinstance(recipe, Mapping) or _canonical_sha256(recipe) != declaration.get(
        "matched_recipe_sha256"
    ):
        raise ValueError(f"score run {run_id} matched recipe hash mismatch")
    scorer_identity = metadata.get(
        "matched_control_scorer_implementation_identity"
    )
    if not isinstance(scorer_identity, Mapping) or not scorer_identity.get("digest"):
        raise ValueError(f"score run {run_id} lacks scorer implementation identity")
    signature = metadata.get("base_scoring_signature")
    if not isinstance(signature, Mapping) or signature.get(
        "checkpoint_id"
    ) != metadata.get("checkpoint_id"):
        raise ValueError(f"score run {run_id} base scoring signature mismatch")
    signature_expected = {
        "config_sha256": metadata.get("config_sha256"),
        "matched_recipe_sha256": declaration.get("matched_recipe_sha256"),
        "method_id": METHOD_ID,
    }
    for key, value in signature_expected.items():
        if signature.get(key) != value:
            raise ValueError(
                f"score run {run_id} base scoring signature mismatch: {key}"
            )


def _append_cross_checkpoint_ledger(
    path: str | Path, *, metrics: Mapping[str, Any], metrics_path: Path
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1,
        "analysis_type": metrics["analysis_type"],
        "balanced_minus_original_history_response_ndcg@10": metrics[
            "balanced_minus_original_history_response_ndcg@10"
        ],
        "balanced_minus_original_target_margin_change": metrics[
            "balanced_minus_original_target_margin_change"
        ],
        "code_revision": metrics["code_revision"],
        "command": metrics["command"],
        "label_mode": metrics["label_mode"],
        "matched_control_evaluator_implementation_digest": metrics[
            "matched_control_evaluator_implementation_identity"
        ]["digest"],
        "method_id": "shared_mechanism_evaluator",
        "metrics_path": str(metrics_path),
        "metrics_sha256": sha256_file(metrics_path),
        "original_mixture_analysis_run_id": metrics[
            "original_mixture_analysis_run_id"
        ],
        "pair_analysis_artifacts": metrics["pair_analysis_artifacts"],
        "pre_qrels_admission_path": metrics["pre_qrels_admission_path"],
        "pre_qrels_admission_sha256": metrics["pre_qrels_admission_sha256"],
        "qrels_read": metrics["qrels_read"],
        "qrels_sha256": metrics["qrels_sha256"],
        "request_count": metrics["request_count"],
        "role": ROLE,
        "score_rows_per_run": metrics["score_rows_per_run"],
        "run_id": metrics["analysis_run_id"],
        "split": metrics["split"],
        "status": metrics["status"],
        "subject_method_id": metrics["method_id"],
        "surface_balanced_analysis_run_id": metrics[
            "surface_balanced_analysis_run_id"
        ],
        "timestamp": _utc_now(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())


def _difference(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _resolve_declared_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "admit_q2_matched_control_score_runs",
    "evaluate_q2_matched_control",
    "matched_control_evaluator_implementation_identity",
    "validate_cross_checkpoint_fixed_recipe",
]
