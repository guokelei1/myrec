"""Resumable qrels-blind runtime for the N8 joint component probe."""

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
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.attention_edge_runtime import (
    BASELINE_SCORE_DIRS,
    DEEP_DIVE_MANIFEST_PATH,
    _assert_native_targets,
    _canonical_sha256,
    _load_content_controls,
    _load_frozen_baseline,
    _load_manifest,
    _read_json,
    _write_json,
)
from myrec.mechanism.component_composition_scoring import (
    composition_conditions,
    score_component_composition_chunk,
)
from myrec.mechanism.component_necessity_runtime import (
    _audit_branch_contract,
    _audit_parent_selected_branch,
)
from myrec.mechanism.representation_probe import normalized_query_fold
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)
from myrec.mechanism.selected_branch_scoring import SELECTED_NODES
from myrec.mechanism.selected_branch_runtime import NULL_BASELINE_DIRS
from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl


NEXT_WAVE_MANIFEST = Path(
    "experiments/motivation/transformer_next_wave_manifest_v1.yaml"
)
NEXT_WAVE_MANIFEST_SHA256 = (
    "7c01947295d19a36c744ed9ea319b92c15c3b99ee9bd52c3b23405cfb2aa54c4"
)
SUPPORTED_METHODS = ("q2_recranker_generalqwen", "q3_tallrec_generalqwen")
MAX_WALL_SECONDS = 13_500.0


def write_component_composition_bundle(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    device: str,
    branch_contract_path: str | Path,
    parent_selected_branch_dir: str | Path,
    runs_dir: str | Path = "runs",
    next_wave_manifest_path: str | Path = NEXT_WAVE_MANIFEST,
    deep_dive_manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Write one Q2/Q3 fold-1 joint composition bundle."""

    if not str(device).strip():
        raise ValueError("component-composition requires an explicit device")
    max_wall_seconds = float(max_wall_seconds)
    if not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("component-composition max_wall_seconds is out of range")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("component-composition max_requests must be positive")
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    next_manifest = _load_next_wave_manifest(next_wave_manifest_path)
    deep_manifest = _load_manifest(deep_dive_manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    for path, expected in (
        (records_path, deep_manifest["frozen_inputs"]["records_dev_sha256"]),
        (standardized_dir / "manifest.json", deep_manifest["frozen_inputs"]["dataset_manifest_sha256"]),
        (standardized_dir / "request_manifest.json", deep_manifest["frozen_inputs"]["request_manifest_sha256"]),
        (standardized_dir / "candidate_manifest.json", deep_manifest["frozen_inputs"]["candidate_manifest_sha256"]),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"component-composition frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("component-composition requires the frozen 8000-request dev")
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in SUPPORTED_METHODS:
        raise ValueError("component-composition supports only Q2/Q3")
    frozen_model = deep_manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("component-composition config differs from deep-dive manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("component-composition checkpoint differs from frozen manifest")
    content_controls, content_identity = _load_content_controls(deep_manifest, method_id, records)
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
    full_scores, full_identity = _load_frozen_baseline(
        BASELINE_SCORE_DIRS[method_id], method_id, checkpoint_id, records
    )
    null_scores, null_identity = _load_frozen_baseline(
        NULL_BASELINE_DIRS[method_id], method_id, checkpoint_id, records
    )
    fold1_records = [record for record in records if normalized_query_fold(record.query) == 1]
    target_records = fold1_records
    if max_requests is not None:
        target_records = sorted(
            fold1_records,
            key=lambda record: hashlib.sha256(
                ("component-composition-smoke-v1\0" + record.request_id).encode()
            ).digest(),
        )[: int(max_requests)]
    conditions = composition_conditions()
    branch_contract_identity = {
        "path": str(branch_contract_path),
        "sha256": sha256_file(branch_contract_path),
        "evidence_role": branch_contract["evidence_role"],
    }
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "selected_block": selected_block,
        "selected_nodes": list(SELECTED_NODES),
        "composition_nodes": ["attention_o_projection", "mlp_down_projection"],
        "conditions": list(conditions),
        "normalized_query_fold": 1,
        "target_requests": len(target_records),
        "records_sha256": sha256_file(records_path),
        "full_scores_sha256": full_identity["scores_sha256"],
        "null_scores_sha256": null_identity["scores_sha256"],
        "content_control_rows_sha256": content_identity["rows_sha256"],
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "next_wave_manifest_sha256": next_manifest["_sha256"],
        "device": str(device),
        "evidence_mode": "registered_component_composition" if max_requests is None else "mechanical_smoke_non_result",
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_component_composition_next_wave",
        "run_id": run_id,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "selected_block": selected_block,
        "selected_nodes": list(SELECTED_NODES),
        "composition_nodes": ["attention_o_projection", "mlp_down_projection"],
        "branch_contract": branch_contract_identity,
        "parent_selected_branch": parent_identity,
        "normalized_query_fold": 1,
        "full_population_request_count": len(records),
        "fold1_request_count": len(fold1_records),
        "target_request_count": len(target_records),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(standardized_dir / "candidate_manifest.json"),
        "request_manifest_sha256": sha256_file(standardized_dir / "request_manifest.json"),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_sha256": deep_manifest["_sha256"],
        "next_wave_manifest_path": str(next_wave_manifest_path),
        "next_wave_manifest_sha256": next_manifest["_sha256"],
        "frozen_full_baseline": full_identity,
        "frozen_null_baseline": null_identity,
        "content_neutral_control": content_identity,
        "score_conditions": list(conditions),
        "identity_tolerance": 1.0e-5,
        "qrels_read": False,
        "source_test_opened": False,
        "result_eligible": max_requests is None,
        "evidence_mode": contract["evidence_mode"],
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    prepared = prepare_scalar_bundle(
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        conditions=conditions,
        resume=resume,
    )
    completed = int(prepared.progress["completed_requests"])
    identity_delta = float(prepared.metadata.get("maximum_identity_delta", 0.0))
    full_delta = float(prepared.metadata.get("maximum_full_baseline_delta", 0.0))
    null_delta = float(prepared.metadata.get("maximum_null_baseline_delta", 0.0))
    if completed >= len(target_records):
        return _finalize(
            run_dir,
            prepared,
            target_records,
            conditions,
            identity_delta=identity_delta,
            full_delta=full_delta,
            null_delta=null_delta,
        )
    import torch

    started = time.monotonic()
    try:
        tokenizer, model = _load_model_and_tokenizer(
            config,
            device=str(device),
            training=False,
            checkpoint_model_dir=checkpoint_model_dir,
        )
        model.eval()
        _assert_native_targets(tokenizer, method_id, deep_manifest)
        prepared.metadata.update(_runtime_metadata(method_id, torch, __import__("transformers")))
        prepared.metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", prepared.metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    _store_progress(prepared.metadata, started, identity_delta, full_delta, null_delta, "wall_time_exhausted")
                    _write_json(run_dir / "metadata.json", prepared.metadata)
                    return prepared.metadata
                record = target_records[ordinal]
                rows = []
                eligible = content_controls[record.request_id].get("eligible") is True
                if eligible:
                    scored = score_component_composition_chunk(
                        model,
                        tokenizer,
                        record,
                        record.candidates,
                        config,
                        content_control=content_controls[record.request_id],
                        block=selected_block,
                        device=str(device),
                    )
                    values = scored["conditions"]
                    identity_delta = max(identity_delta, float(scored["maximum_identity_delta"]))
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        key = (record.request_id, str(candidate["item_id"]))
                        full_value = float(values["baseline_full"][candidate_ordinal].item())
                        null_value = float(values["baseline_null"][candidate_ordinal].item())
                        full_delta = max(full_delta, abs(full_value - full_scores[key]))
                        null_delta = max(null_delta, abs(null_value - null_scores[key]))
                        rows.append({
                            "request_id": record.request_id,
                            "candidate_ordinal": candidate_ordinal,
                            "candidate_item_id": str(candidate["item_id"]),
                            "conditions": {name: float(values[name][candidate_ordinal].item()) for name in conditions},
                        })
                else:
                    for candidate_ordinal, candidate in enumerate(record.candidates):
                        key = (record.request_id, str(candidate["item_id"]))
                        baseline_full = float(full_scores[key])
                        baseline_null = float(null_scores[key])
                        rows.append({
                            "request_id": record.request_id,
                            "candidate_ordinal": candidate_ordinal,
                            "candidate_item_id": str(candidate["item_id"]),
                            "conditions": {name: (baseline_null if name == "baseline_null" else baseline_full) for name in conditions},
                        })
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
        _store_progress(prepared.metadata, started, identity_delta, full_delta, null_delta, "mechanical_failure")
        prepared.metadata["error"] = {"type": type(exc).__name__, "message": str(exc)}
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise
    _store_progress(prepared.metadata, started, identity_delta, full_delta, null_delta, "running")
    _write_json(run_dir / "metadata.json", prepared.metadata)
    return _finalize(
        run_dir,
        prepared,
        target_records,
        conditions,
        identity_delta=identity_delta,
        full_delta=full_delta,
        null_delta=null_delta,
    )


def _load_next_wave_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    observed = sha256_file(path)
    if observed != NEXT_WAVE_MANIFEST_SHA256:
        raise ValueError("next-wave manifest differs from frozen digest")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("status") != "frozen_before_next_wave_outcomes":
        raise ValueError("next-wave manifest freeze attestation is inadmissible")
    if value.get("scope", {}).get("dataset_version") != "full_confirm_preceding40k_v11":
        raise ValueError("next-wave dataset binding drift")
    value["_sha256"] = observed
    return value


def _store_progress(
    metadata: dict[str, Any],
    started: float,
    identity_delta: float,
    full_delta: float,
    null_delta: float,
    status: str,
) -> None:
    metadata.update(
        {
            "status": status,
            "resumable": status != "running",
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0)) + time.monotonic() - started,
            "maximum_identity_delta": float(identity_delta),
            "maximum_full_baseline_delta": float(full_delta),
            "maximum_null_baseline_delta": float(null_delta),
        }
    )


def _finalize(
    run_dir: Path,
    prepared: Any,
    records: Sequence[Any],
    conditions: Sequence[str],
    *,
    identity_delta: float,
    full_delta: float,
    null_delta: float,
) -> dict[str, Any]:
    if identity_delta > 1.0e-5 or full_delta > 1.0e-5 or null_delta > 1.0e-5:
        message = (
            "component-composition identity/baseline gate failed: "
            f"identity={identity_delta}, full={full_delta}, null={null_delta}"
        )
        prepared.metadata["error"] = {"type": "ValueError", "message": message}
        _store_progress(prepared.metadata, 0.0, identity_delta, full_delta, null_delta, "mechanical_failure")
        _write_json(run_dir / "metadata.json", prepared.metadata)
        raise ValueError(message)
    result = finalize_scalar_bundle(
        run_dir,
        prepared,
        records,
        conditions,
        maximum_identity_delta=identity_delta,
    )
    result.update(
        {
            "maximum_full_baseline_delta": float(full_delta),
            "maximum_null_baseline_delta": float(null_delta),
        }
    )
    _write_json(run_dir / "metadata.json", result)
    return result
