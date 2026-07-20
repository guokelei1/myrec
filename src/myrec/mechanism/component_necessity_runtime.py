"""Resumable qrels-blind runtime for reverse component-state removal."""

from __future__ import annotations

import hashlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from myrec.baselines.motivation_v12_contracts import sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _checkpoint_identity,
    _git_revision,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_runtime import (
    BASELINE_SCORE_DIRS,
    DEEP_DIVE_MANIFEST_PATH,
    MAX_WALL_SECONDS,
    _assert_native_targets,
    _canonical_sha256,
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.component_necessity_scoring import (
    NECESSITY_NODES,
    component_necessity_conditions,
    score_component_necessity_chunk,
)
from myrec.mechanism.representation_probe import normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.mechanism.selected_branch_runtime import NULL_BASELINE_DIRS
from myrec.mechanism.selected_branch_scoring import SELECTED_NODES
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


SUPPORTED_METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
EXTENSION_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_component_necessity_extension_manifest_v2.yaml"
)
EXTENSION_MANIFEST_SHA256 = (
    "6b784682239e5ce8e6f1c37fed1c648658912ed03801750fe99d256b0777c0e3"
)


def write_component_necessity_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    branch_contract_path: str | Path,
    parent_selected_branch_dir: str | Path,
    runs_dir: str | Path = "runs",
    deep_dive_manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    extension_manifest_path: str | Path = EXTENSION_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score four reverse-removal nodes at the immutable D2 selected block."""

    _validate_run_id(run_id)
    if not str(device).strip():
        raise ValueError("an explicit component-necessity device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("component-necessity max_wall_seconds must be in (0,13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("component-necessity max_requests must be positive")
    formal = max_requests is None

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    deep_dive_manifest = _load_manifest(deep_dive_manifest_path)
    extension_manifest = _load_extension_manifest(extension_manifest_path)
    _audit_parent_manifest_binding(
        extension_manifest,
        deep_dive_manifest_path=Path(deep_dive_manifest_path),
        deep_dive_manifest=deep_dive_manifest,
    )

    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, extension_manifest["frozen_inputs"]["records_dev_sha256"]),
        (
            standardized_dir / "manifest.json",
            extension_manifest["frozen_inputs"]["dataset_manifest_sha256"],
        ),
        (
            standardized_dir / "request_manifest.json",
            extension_manifest["frozen_inputs"]["request_manifest_sha256"],
        ),
        (
            standardized_dir / "candidate_manifest.json",
            extension_manifest["frozen_inputs"]["candidate_manifest_sha256"],
        ),
    ):
        if not path.is_file() or sha256_file(path) != str(expected):
            raise ValueError(f"component-necessity frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("component-necessity scoring requires frozen 8000-request dev")

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("component-necessity scoring supports only Q2/Q3")
    frozen = extension_manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen["config_sha256"]:
        raise ValueError("component-necessity config differs from frozen extension")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, method_id
    )
    if checkpoint_id != frozen["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("component-necessity checkpoint differs from frozen binding")

    content_controls, content_control_identity = _load_content_controls(
        deep_dive_manifest, method_id, records
    )
    registered_content = extension_manifest["frozen_inputs"]["content_neutral"]
    expected_rows_sha = registered_content[
        "q2_rows_sha256" if method_id.startswith("q2_") else "q3_rows_sha256"
    ]
    if (
        content_control_identity["manifest_sha256"]
        != registered_content["manifest_sha256"]
        or content_control_identity["rows_sha256"] != expected_rows_sha
        or content_control_identity["eligible_requests"]
        != int(registered_content["eligible_requests_each_model"])
    ):
        raise ValueError("component-necessity content-neutral binding drift")

    branch_contract_path = Path(branch_contract_path)
    branch_contract = _audit_branch_contract(
        branch_contract_path, method_id=method_id, checkpoint_id=checkpoint_id
    )
    selected_block = int(branch_contract["selected_block"])
    parent_identity = _audit_parent_selected_branch(
        parent_selected_branch_dir,
        method_id=method_id,
        checkpoint_id=checkpoint_id,
        selected_block=selected_block,
        branch_contract_path=branch_contract_path,
    )
    branch_contract_identity = {
        "path": str(branch_contract_path),
        "sha256": sha256_file(branch_contract_path),
        "evidence_role": branch_contract["evidence_role"],
    }

    baseline_full, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    baseline_null, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    fold1_records = [
        record for record in records if normalized_query_fold(record.query) == 1
    ]
    target_records = (
        fold1_records
        if formal
        else _stable_smoke_records(fold1_records, int(max_requests))
    )
    conditions = component_necessity_conditions()
    fold1_content_eligible = sum(
        content_controls[record.request_id].get("eligible") is True
        for record in fold1_records
    )
    target_content_eligible = sum(
        content_controls[record.request_id].get("eligible") is True
        for record in target_records
    )
    implementation = component_necessity_implementation_identity()
    evidence_mode = (
        "registered_component_necessity_diagnostic"
        if formal
        else "mechanical_smoke_non_result"
    )
    run_contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "selected_block": selected_block,
        "branch_contract": branch_contract_identity,
        "parent_selected_branch": parent_identity,
        "normalized_query_fold": 1,
        "target_requests": len(target_records),
        "score_conditions": list(conditions),
        "records_sha256": sha256_file(records_path),
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "content_control_rows_sha256": content_control_identity["rows_sha256"],
        "deep_dive_manifest_sha256": deep_dive_manifest["_sha256"],
        "extension_manifest_sha256": extension_manifest["_sha256"],
        "device": str(device),
        "implementation_digest": implementation["digest"],
        "evidence_mode": evidence_mode,
    }
    run_contract_sha256 = _canonical_sha256(run_contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_component_necessity_extension",
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "selected_block": selected_block,
        "selected_nodes": list(NECESSITY_NODES),
        "branch_contract": branch_contract_identity,
        "parent_selected_branch": parent_identity,
        "evidence_role": branch_contract["evidence_role"],
        "normalized_query_fold": 1,
        "full_population_request_count": len(records),
        "fold1_request_count": len(fold1_records),
        "target_request_count": len(target_records),
        "fold1_content_neutral_eligible_requests": fold1_content_eligible,
        "target_content_neutral_eligible_requests": target_content_eligible,
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            standardized_dir / "request_manifest.json"
        ),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": deep_dive_manifest["_sha256"],
        "extension_manifest_path": str(extension_manifest_path),
        "extension_manifest_sha256": extension_manifest["_sha256"],
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "content_neutral_control": content_control_identity,
        "score_conditions": list(conditions),
        "identity_tolerance": 1.0e-5,
        "causal_intervention_role": "full_context_component_state_removal",
        "operator_necessity_tested": False,
        "exclusive_origin_tested": False,
        "implementation_identity": implementation,
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": evidence_mode,
        "result_eligible": formal,
        "run_contract": run_contract,
        "run_contract_sha256": run_contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    prepared = prepare_scalar_bundle(
        run_dir,
        metadata=metadata,
        contract_sha256=run_contract_sha256,
        records=target_records,
        conditions=conditions,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    full_delta = float(
        prepared.metadata.get("maximum_full_baseline_delta", 0.0)
    )
    null_delta = float(
        prepared.metadata.get("maximum_null_baseline_delta", 0.0)
    )
    baseline_ratio = float(
        prepared.metadata.get("maximum_baseline_low_precision_ratio", 0.0)
    )
    shared_delta = float(
        prepared.metadata.get("shared_prompt_path_max_abs_delta", 0.0)
    )
    if completed >= len(target_records):
        return finalize_scalar_bundle(
            run_dir,
            prepared,
            target_records,
            conditions,
            maximum_identity_delta=identity_delta,
        )

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config,
            device=str(device),
            training=False,
            checkpoint_model_dir=checkpoint_model_dir,
        )
        model.eval()
        _assert_native_targets(tokenizer, method_id, deep_dive_manifest)
        prepared.metadata.update(_runtime_metadata(method_id, torch, transformers))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    _store_progress(
                        prepared.metadata,
                        started,
                        identity_delta,
                        full_delta,
                        null_delta,
                        baseline_ratio,
                        shared_delta,
                        status="wall_time_exhausted",
                    )
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                rows = []
                for start in range(0, len(record.candidates), batch_size):
                    candidates = list(record.candidates[start : start + batch_size])
                    result = score_component_necessity_chunk(
                        model,
                        tokenizer,
                        record,
                        candidates,
                        config,
                        content_control=content_controls[record.request_id],
                        block=selected_block,
                        device=str(device),
                    )
                    identity_delta = max(
                        identity_delta, float(result["maximum_identity_delta"])
                    )
                    shared_delta = max(
                        shared_delta,
                        float(result.get("shared_prompt_path_max_abs_delta", 0.0)),
                    )
                    score_values = result["conditions"]
                    expected_eligible = (
                        content_controls[record.request_id].get("eligible") is True
                    )
                    if (
                        result.get("content_neutral_eligible") is not expected_eligible
                        or result.get("neutral_path_identity_passed")
                        is not expected_eligible
                    ):
                        raise RuntimeError(
                            "component-necessity neutral eligibility/path audit drift"
                        )
                    for local, candidate in enumerate(candidates):
                        values = {
                            name: float(score_values[name][local].item())
                            for name in conditions
                        }
                        if not all(math.isfinite(value) for value in values.values()):
                            raise FloatingPointError(
                                "component-necessity score is non-finite"
                            )
                        key = (record.request_id, str(candidate["item_id"]))
                        full_delta = max(
                            full_delta,
                            abs(values["baseline_full"] - baseline_full[key]),
                        )
                        null_delta = max(
                            null_delta,
                            abs(values["baseline_null"] - baseline_null[key]),
                        )
                        for condition, reference in (
                            ("baseline_full", baseline_full[key]),
                            ("baseline_null", baseline_null[key]),
                        ):
                            bound = 8.0 * (2.0**-7) * max(1.0, abs(reference))
                            baseline_ratio = max(
                                baseline_ratio,
                                abs(values[condition] - reference) / bound,
                            )
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "content_neutral_eligible": expected_eligible,
                                "conditions": values,
                            }
                        )
                append_scalar_request(
                    run_dir,
                    {
                        "ordinal": ordinal,
                        "request_id": record.request_id,
                        "rows": rows,
                        "rows_sha256": _canonical_sha256(rows),
                    },
                    prepared,
                )
    except Exception as exc:
        _store_progress(
            prepared.metadata,
            started,
            identity_delta,
            full_delta,
            null_delta,
            baseline_ratio,
            shared_delta,
            status="mechanical_failure",
        )
        prepared.metadata["error"] = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    if baseline_ratio > 1.0:
        message = (
            "component-necessity recomputed baseline exceeded path-local BF16 "
            f"bound: ratio={baseline_ratio}"
        )
        _store_progress(
            prepared.metadata,
            started,
            identity_delta,
            full_delta,
            null_delta,
            baseline_ratio,
            shared_delta,
            status="mechanical_failure",
        )
        prepared.metadata["error"] = {"type": "ValueError", "message": message}
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise ValueError(message)
    _store_progress(
        prepared.metadata,
        started,
        identity_delta,
        full_delta,
        null_delta,
        baseline_ratio,
        shared_delta,
        status="running",
    )
    return finalize_scalar_bundle(
        run_dir,
        prepared,
        target_records,
        conditions,
        maximum_identity_delta=identity_delta,
    )


def _load_extension_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != EXTENSION_MANIFEST_SHA256:
        raise ValueError("component-necessity manifest differs from frozen digest")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("component-necessity manifest is not a mapping")
    plan = value.get("scientific_plan", {})
    plan_path = Path(str(plan.get("path") or ""))
    if not plan_path.is_file() or sha256_file(plan_path) != plan.get("sha256"):
        raise ValueError("component-necessity plan differs from frozen binding")
    if (
        value.get("status") != "frozen_before_component_necessity_outcomes"
        or plan.get("scientific_effect_values_read_before_freeze") is not False
        or plan.get("qrels_read_before_freeze") is not False
        or plan.get("source_test_opened_before_freeze") is not False
    ):
        raise ValueError("component-necessity freeze attestation is inadmissible")
    superseded = value.get("supersedes_before_execution", {})
    for key in ("plan_path", "manifest_path"):
        frozen_path = Path(str(superseded.get(key) or ""))
        expected_key = "plan_sha256" if key == "plan_path" else "manifest_sha256"
        if not frozen_path.is_file() or sha256_file(frozen_path) != superseded.get(
            expected_key
        ):
            raise ValueError("component-necessity V1 supersession binding drift")
    if superseded.get("v1_score_bundles_started") is not False:
        raise ValueError("component-necessity V1 was not superseded before execution")
    value["_sha256"] = observed
    return value


def _audit_parent_manifest_binding(
    extension: Mapping[str, Any],
    *,
    deep_dive_manifest_path: Path,
    deep_dive_manifest: Mapping[str, Any],
) -> None:
    parent = extension.get("parent_deep_dive", {})
    plan_path = Path(str(parent.get("plan_path") or ""))
    if (
        deep_dive_manifest_path != Path(str(parent.get("manifest_path") or ""))
        or sha256_file(deep_dive_manifest_path) != parent.get("manifest_sha256")
        or deep_dive_manifest.get("_sha256") != parent.get("manifest_sha256")
        or not plan_path.is_file()
        or sha256_file(plan_path) != parent.get("plan_sha256")
        or parent.get("parent_families_modified") is not False
    ):
        raise ValueError("component-necessity parent deep-dive binding drift")


def _audit_branch_contract(
    path: Path, *, method_id: str, checkpoint_id: str
) -> dict[str, Any]:
    contract = _read_json(path)
    if (
        contract.get("contract_type")
        != "transformer_deep_dive_d2_selected_branch_contract"
        or contract.get("status") != "completed"
        or contract.get("branch_scoring_eligible") is not True
        or contract.get("fold1_negative_transition_reproduced") is not True
        or contract.get("evidence_role")
        != "registered_confirmatory_branch_localization"
        or contract.get("method_id") != method_id
        or contract.get("checkpoint_id") != checkpoint_id
        or contract.get("scoring_population") != "normalized_query_fold_1"
        or contract.get("selected_nodes") != list(SELECTED_NODES)
        or contract.get("qrels_values_exposed_to_scorer") is not False
        or not 13 <= int(contract.get("selected_block", -1)) <= 27
    ):
        raise ValueError("component-necessity branch contract is inadmissible")
    return contract


def _audit_parent_selected_branch(
    root: str | Path,
    *,
    method_id: str,
    checkpoint_id: str,
    selected_block: int,
    branch_contract_path: Path,
) -> dict[str, Any]:
    root = Path(root)
    metadata_path = root / "metadata.json"
    scores_path = root / "scores.jsonl"
    metadata = _read_json(metadata_path)
    expected_contract_sha = sha256_file(branch_contract_path)
    if (
        metadata.get("analysis_stage")
        != "transformer_deep_dive_d2_selected_branch"
        or metadata.get("status") != "completed"
        or metadata.get("result_eligible") is not True
        or metadata.get("complete_finite_score_coverage") is not True
        or metadata.get("identity_passed") is not True
        or metadata.get("method_id") != method_id
        or metadata.get("checkpoint_id") != checkpoint_id
        or int(metadata.get("selected_block", -1)) != selected_block
        or metadata.get("branch_contract", {}).get("sha256")
        != expected_contract_sha
        or metadata.get("qrels_read") is not False
        or metadata.get("source_test_opened") is not False
        or not scores_path.is_file()
        or metadata.get("scores_sha256") != sha256_file(scores_path)
    ):
        raise ValueError("parent selected-branch bundle is incomplete or unbound")
    return {
        "path": str(root),
        "metadata_sha256": sha256_file(metadata_path),
        "scores_sha256": sha256_file(scores_path),
    }


def _stable_smoke_records(records: Sequence[Any], limit: int) -> list[Any]:
    return sorted(
        records,
        key=lambda record: hashlib.sha256(
            (
                "component-necessity-extension-smoke-v1\0" + record.request_id
            ).encode("utf-8")
        ).digest(),
    )[:limit]


def component_necessity_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "experiments/motivation/transformer_component_necessity_extension_plan_v2.md",
        "experiments/motivation/transformer_component_necessity_extension_manifest_v2.yaml",
        "src/myrec/mechanism/component_necessity_runtime.py",
        "src/myrec/mechanism/component_necessity_scoring.py",
        "src/myrec/mechanism/attention_edge_runtime.py",
        "src/myrec/mechanism/attention_edge_scoring.py",
        "src/myrec/mechanism/deep_dive_assignments.py",
        "src/myrec/mechanism/selected_branch_scoring.py",
        "src/myrec/mechanism/transformer_instrumentation.py",
        "src/myrec/mechanism/deep_dive_native_patch.py",
        "src/myrec/mechanism/native_readout_scoring.py",
        "src/myrec/mechanism/scalar_condition_bundle.py",
        "scripts/score_deep_dive_component_necessity.py",
    )
    files = [
        {
            "path": path,
            "sha256": sha256_file(root / path),
            "size_bytes": (root / path).stat().st_size,
        }
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _store_progress(
    metadata: dict[str, Any],
    started: float,
    identity: float,
    full_delta: float,
    null_delta: float,
    baseline_ratio: float,
    shared_delta: float,
    *,
    status: str,
) -> None:
    metadata.update(
        {
            "status": status,
            "resumable": status != "running",
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "maximum_identity_delta": identity,
            "maximum_full_baseline_delta": full_delta,
            "maximum_null_baseline_delta": null_delta,
            "maximum_baseline_low_precision_ratio": baseline_ratio,
            "shared_prompt_path_max_abs_delta": shared_delta,
        }
    )
