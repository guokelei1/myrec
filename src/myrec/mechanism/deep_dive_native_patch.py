"""Qrels-blind Q3 all-native-position block-state patch gate.

Q3's score contains four log-probability terms at three distinct hidden
states.  The first-round patch changed only the shared prompt state.  This
runtime captures and patches both target paths so the full native score is
covered without treating one prompt token as the whole readout.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord, sanitize_record_for_model
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _answer_target_tokens,
    _checkpoint_identity,
    _git_revision,
    _load_model_and_tokenizer,
    _runtime_metadata,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.representation_probe import (
    MechanicalPositionError,
    instrument_pointwise_prompt,
)
from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


DEEP_DIVE_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_deep_dive_manifest.yaml"
)
Q3_METHOD_ID = "q3_tallrec_generalqwen"
GATE_BLOCKS = (13, 27)
NATIVE_TERMS = (
    "prompt_predict_yes",
    "yes_predict_terminator",
    "prompt_predict_no",
    "no_predict_terminator",
)
SCORE_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "full_to_full_identity",
    "null_to_null_identity",
    "same_all_native_positions",
    "same_first_position_only",
)
MAX_WALL_SECONDS = 13_500.0


def q3_native_patch_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    block: int,
    device: str,
) -> dict[str, Any]:
    """Return six native-score conditions and their four exact terms."""

    import torch

    block = int(block)
    if block not in GATE_BLOCKS:
        raise ValueError("Q3 native gate block must be 13 or 27")
    if not candidates:
        raise ValueError("Q3 native gate candidate chunk is empty")
    spec = NodeSpec("block_output_residual", block)
    full = _q3_context(tokenizer, record, candidates, record.history, config, device)
    null = _q3_context(tokenizer, record, candidates, [], config, device)
    with QwenNodeCapture(model, [spec]) as capture:
        full_capture = _capture_context(model, capture, spec, full)
        null_capture = _capture_context(model, capture, spec, null)
    shared_delta_full = float(
        torch.max(
            torch.abs(
                full_capture["yes_states"][:, 0]
                - full_capture["no_states"][:, 0]
            )
        ).item()
    )
    shared_delta_null = float(
        torch.max(
            torch.abs(
                null_capture["yes_states"][:, 0]
                - null_capture["no_states"][:, 0]
            )
        ).item()
    )
    if shared_delta_full != 0.0 or shared_delta_null != 0.0:
        raise RuntimeError(
            "Q3 shared prompt state differs across equal-length Yes/No paths: "
            f"full={shared_delta_full}, null={shared_delta_null}"
        )
    with QwenNodePatch(model, spec) as patcher:
        full_identity = _patch_context(
            model,
            patcher,
            full,
            full_capture,
            scope="all_native_positions",
        )
        null_identity = _patch_context(
            model,
            patcher,
            null,
            null_capture,
            scope="all_native_positions",
        )
        same_all = _patch_context(
            model,
            patcher,
            null,
            full_capture,
            scope="all_native_positions",
        )
        same_first = _patch_context(
            model,
            patcher,
            null,
            full_capture,
            scope="first_position_only",
        )
    results = {
        "baseline_full": full_capture,
        "baseline_null": null_capture,
        "full_to_full_identity": full_identity,
        "null_to_null_identity": null_identity,
        "same_all_native_positions": same_all,
        "same_first_position_only": same_first,
    }
    return {
        "conditions": {
            name: {
                "score": value["score"],
                "terms": value["terms"],
            }
            for name, value in results.items()
        },
        "shared_prompt_path_max_abs_delta": {
            "full": shared_delta_full,
            "null": shared_delta_null,
        },
    }


def write_q3_native_patch_gate(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score one full-population Q3 native-position gate block."""

    _validate_run_id(run_id)
    block = int(block)
    if block not in GATE_BLOCKS:
        raise ValueError("Q3 native gate block must be 13 or 27")
    if not str(device).strip():
        raise ValueError("an explicit native-gate device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("native-gate max_wall_seconds must be in (0, 13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("max_requests must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    manifest = _load_manifest(manifest_path)
    records_path = standardized_dir / "records_dev.jsonl"
    input_paths = {
        records_path: manifest["frozen_inputs"]["records_dev_sha256"],
        standardized_dir / "manifest.json": manifest["frozen_inputs"][
            "dataset_manifest_sha256"
        ],
        standardized_dir / "request_manifest.json": manifest["frozen_inputs"][
            "request_manifest_sha256"
        ],
        standardized_dir / "candidate_manifest.json": manifest["frozen_inputs"][
            "candidate_manifest_sha256"
        ],
    }
    for path, expected in input_paths.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"native-gate frozen input hash mismatch: {path}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("Q3 native gate requires all 8000 internal-dev requests")

    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q3_METHOD_ID:
        raise ValueError("native-position gate is Q3-only")
    frozen_model = manifest["frozen_inputs"]["models"][Q3_METHOD_ID]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("native-gate config differs from frozen manifest")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, Q3_METHOD_ID
    )
    if checkpoint_id != frozen_model["checkpoint_id"]:
        raise ValueError("native-gate checkpoint differs from frozen manifest")
    if training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("native-gate checkpoint changed after training metadata")

    target_records = records if max_requests is None else records[: int(max_requests)]
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = native_patch_implementation_identity()
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q3_METHOD_ID,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "deep_dive_manifest_sha256": manifest["sha256"],
        "block_zero_based": block,
        "native_terms": list(NATIVE_TERMS),
        "score_conditions": list(SCORE_CONDITIONS),
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d2_q3_native_position_gate",
        "run_id": run_id,
        "method_id": Q3_METHOD_ID,
        "block_zero_based": block,
        "patch_node": "block_output_residual",
        "native_positions": [
            "shared_prompt_last",
            "teacher_forced_yes_last",
            "teacher_forced_no_last",
        ],
        "native_terms": list(NATIVE_TERMS),
        "score_conditions": list(SCORE_CONDITIONS),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "candidate_manifest_sha256": sha256_file(
            standardized_dir / "candidate_manifest.json"
        ),
        "request_manifest_sha256": sha256_file(
            standardized_dir / "request_manifest.json"
        ),
        "dataset_manifest_sha256": sha256_file(standardized_dir / "manifest.json"),
        "deep_dive_manifest_path": str(manifest["path"]),
        "deep_dive_manifest_sha256": manifest["sha256"],
        "implementation_identity": implementation,
        "identity_tolerance": 1.0e-5,
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
        "evidence_mode": evidence_mode,
        "result_eligible": max_requests is None,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    metadata, progress, partial_hasher = _prepare_state(
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        resume=resume,
    )
    completed = int(progress["completed_requests"])
    if completed >= len(target_records):
        return _finalize(run_dir, metadata, progress, target_records)

    started = time.monotonic()
    max_identity_delta = {
        "full_score": float(metadata.get("max_identity_delta", {}).get("full_score", 0.0)),
        "null_score": float(metadata.get("max_identity_delta", {}).get("null_score", 0.0)),
        "full_terms": float(metadata.get("max_identity_delta", {}).get("full_terms", 0.0)),
        "null_terms": float(metadata.get("max_identity_delta", {}).get("null_terms", 0.0)),
    }
    shared_path_delta = float(metadata.get("shared_prompt_path_max_abs_delta", 0.0))
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
        yes_target = _answer_target_tokens(tokenizer, "Yes")
        no_target = _answer_target_tokens(tokenizer, "No")
        expected_targets = manifest["registered_design"]["q3_targets"]
        # PyYAML 1.1 resolves the frozen unquoted keys ``yes``/``no`` to
        # booleans.  Accept only that exact frozen parse; do not rewrite the
        # immutable manifest after outcomes exist.
        expected_yes = expected_targets.get("yes", expected_targets.get(True))
        expected_no = expected_targets.get("no", expected_targets.get(False))
        if yes_target != expected_yes or no_target != expected_no:
            raise MechanicalPositionError(
                "q3_target_identity_mismatch",
                "runtime Q3 targets differ from frozen deep-dive target IDs",
            )
        metadata.update(_runtime_metadata(Q3_METHOD_ID, torch, transformers))
        metadata["status"] = "running"
        _write_json(run_dir / "metadata.json", metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode():
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    metadata.update(
                        {
                            "status": "wall_time_exhausted",
                            "resumable": True,
                            "completed_requests": ordinal,
                            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
                            + time.monotonic()
                            - started,
                            "max_identity_delta": max_identity_delta,
                            "shared_prompt_path_max_abs_delta": shared_path_delta,
                        }
                    )
                    _write_json(run_dir / "metadata.json", metadata)
                    return metadata
                record = target_records[ordinal]
                score_rows: list[dict[str, Any]] = []
                for start in range(0, len(record.candidates), batch_size):
                    chunk = list(record.candidates[start : start + batch_size])
                    result = q3_native_patch_chunk(
                        model,
                        tokenizer,
                        record,
                        chunk,
                        config,
                        block=block,
                        device=str(device),
                    )
                    shared_path_delta = max(
                        shared_path_delta,
                        *result["shared_prompt_path_max_abs_delta"].values(),
                    )
                    conditions = result["conditions"]
                    for local, candidate in enumerate(chunk):
                        row_conditions: dict[str, Any] = {}
                        for name in SCORE_CONDITIONS:
                            score = float(conditions[name]["score"][local])
                            terms = [
                                float(value)
                                for value in conditions[name]["terms"][local]
                            ]
                            if not math.isfinite(score) or not all(
                                math.isfinite(value) for value in terms
                            ):
                                raise FloatingPointError(
                                    "Q3 native gate produced a non-finite score term"
                                )
                            row_conditions[name] = {"score": score, "terms": terms}
                        max_identity_delta["full_score"] = max(
                            max_identity_delta["full_score"],
                            abs(
                                row_conditions["baseline_full"]["score"]
                                - row_conditions["full_to_full_identity"]["score"]
                            ),
                        )
                        max_identity_delta["null_score"] = max(
                            max_identity_delta["null_score"],
                            abs(
                                row_conditions["baseline_null"]["score"]
                                - row_conditions["null_to_null_identity"]["score"]
                            ),
                        )
                        max_identity_delta["full_terms"] = max(
                            max_identity_delta["full_terms"],
                            max(
                                abs(a - b)
                                for a, b in zip(
                                    row_conditions["baseline_full"]["terms"],
                                    row_conditions["full_to_full_identity"]["terms"],
                                )
                            ),
                        )
                        max_identity_delta["null_terms"] = max(
                            max_identity_delta["null_terms"],
                            max(
                                abs(a - b)
                                for a, b in zip(
                                    row_conditions["baseline_null"]["terms"],
                                    row_conditions["null_to_null_identity"]["terms"],
                                )
                            ),
                        )
                        score_rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "candidate_ordinal": start + local,
                                "conditions": row_conditions,
                            }
                        )
                block_row = {
                    "ordinal": ordinal,
                    "request_id": record.request_id,
                    "block_zero_based": block,
                    "rows": score_rows,
                    "rows_sha256": _canonical_sha256(score_rows),
                }
                line = _canonical_json(block_row) + "\n"
                _append_sync(run_dir / "scores.partial.jsonl", line)
                partial_hasher.update(line.encode("utf-8"))
                progress.update(
                    {
                        "completed_requests": ordinal + 1,
                        "completed_score_rows": int(progress["completed_score_rows"])
                        + len(score_rows),
                        "last_request_id": record.request_id,
                        "partial_sha256": partial_hasher.hexdigest(),
                        "status": "running",
                        "updated_at": _utc_now(),
                    }
                )
                _write_json(run_dir / "progress.json", progress)
    except Exception as exc:
        metadata.update(
            {
                "status": "mechanical_failure"
                if isinstance(exc, (MechanicalPositionError, RuntimeError))
                else "failed",
                "resumable": not isinstance(exc, MechanicalPositionError),
                "qrels_read": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
                "max_identity_delta": max_identity_delta,
                "shared_prompt_path_max_abs_delta": shared_path_delta,
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise
    metadata.update(
        {
            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
            + time.monotonic()
            - started,
            "max_identity_delta": max_identity_delta,
            "shared_prompt_path_max_abs_delta": shared_path_delta,
        }
    )
    return _finalize(run_dir, metadata, progress, target_records)


def _q3_context(
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: str,
) -> dict[str, Any]:
    yes_target = _answer_target_tokens(tokenizer, "Yes")
    no_target = _answer_target_tokens(tokenizer, "No")
    if len(yes_target) != 2 or len(no_target) != 2 or len(yes_target) != len(no_target):
        raise MechanicalPositionError(
            "q3_native_target_length",
            "Q3 native gate requires frozen two-token equal-length targets",
        )
    prompts = [
        instrument_pointwise_prompt(
            tokenizer,
            Q3_METHOD_ID,
            record,
            candidate,
            history=history,
            history_budget=int(config["training"]["history_budget"]),
            max_length=int(config["training"]["max_length"]) - 2,
        )
        for candidate in candidates
    ]
    paths: dict[str, Any] = {}
    for name, target in (("yes", yes_target), ("no", no_target)):
        sequences = [list(prompt.token_ids) + list(target) for prompt in prompts]
        ids, mask, padding = _left_pad_sequences(
            sequences, tokenizer.pad_token_id, device
        )
        positions = __import__("torch").tensor(
            [
                [left + prompt.candidate_readout, left + prompt.candidate_readout + 1]
                for left, prompt in zip(padding, prompts)
            ],
            dtype=__import__("torch").long,
            device=device,
        )
        paths[name] = {
            "ids": ids,
            "mask": mask,
            "positions": positions,
            "target": list(target),
        }
    return {"paths": paths, "candidates": len(candidates)}


def _capture_context(
    model: Any, capture: QwenNodeCapture, spec: NodeSpec, context: Mapping[str, Any]
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    terms: list[Any] = []
    for name in ("yes", "no"):
        path = context["paths"][name]
        output, captured = capture.capture_forward(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            positions=path["positions"],
            model_kwargs={"logits_to_keep": 3},
        )
        values[f"{name}_states"] = captured[spec.key]
        terms.append(_path_terms(output, path))
    term_matrix, score = _combine_terms(terms[0], terms[1])
    values.update({"terms": term_matrix, "score": score})
    return values


def _patch_context(
    model: Any,
    patcher: QwenNodePatch,
    context: Mapping[str, Any],
    donor: Mapping[str, Any],
    *,
    scope: str,
) -> dict[str, Any]:
    if scope not in {"all_native_positions", "first_position_only"}:
        raise ValueError("unsupported Q3 native patch scope")
    terms: list[Any] = []
    count = 2 if scope == "all_native_positions" else 1
    for name in ("yes", "no"):
        path = context["paths"][name]
        positions = path["positions"][:, :count]
        vectors = donor[f"{name}_states"][:, :count]
        patcher.arm(positions, vectors, sequence_length=int(path["ids"].shape[1]))
        output = model(
            input_ids=path["ids"],
            attention_mask=path["mask"],
            use_cache=False,
            logits_to_keep=3,
        )
        patcher.disarm()
        terms.append(_path_terms(output, path))
    term_matrix, score = _combine_terms(terms[0], terms[1])
    return {"terms": term_matrix, "score": score}


def _path_terms(output: Any, path: Mapping[str, Any]) -> Any:
    import torch
    from torch.nn import functional as F

    logits = output.logits[:, -3:-1].float()
    target = torch.tensor(path["target"], dtype=torch.long, device=logits.device)
    target = target[None, :, None].expand(logits.shape[0], -1, -1)
    return F.log_softmax(logits, dim=-1).gather(2, target).squeeze(2)


def _combine_terms(yes_terms: Any, no_terms: Any) -> tuple[Any, Any]:
    import torch

    matrix = torch.stack(
        (yes_terms[:, 0], yes_terms[:, 1], no_terms[:, 0], no_terms[:, 1]),
        dim=1,
    )
    score = yes_terms.mean(dim=1) - no_terms.mean(dim=1)
    return matrix.float().cpu().numpy(), score.float().cpu().numpy()


def native_patch_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "src/myrec/mechanism/deep_dive_native_patch.py",
        root / "src/myrec/mechanism/transformer_instrumentation.py",
        root / "src/myrec/mechanism/representation_probe.py",
        root / "scripts/score_deep_dive_q3_native_gate.py",
    )
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _prepare_state(
    run_dir: Path,
    *,
    metadata: dict[str, Any],
    contract_sha256: str,
    records: Sequence[ModelRecord],
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    partial_path = run_dir / "scores.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"native-gate run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial_path.touch(exist_ok=False)
        metadata.update({"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []})
        progress = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "completed_requests": 0,
            "completed_score_rows": 0,
            "last_request_id": None,
            "partial_sha256": sha256_file(partial_path),
            "status": "initializing",
            "updated_at": _utc_now(),
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress, hashlib.sha256()
    metadata = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if metadata.get("run_contract_sha256") != contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != contract_sha256:
        raise ValueError("native-gate resume contract drift")
    observed = _audit_partial(partial_path, records, metadata)
    for key in (
        "completed_requests",
        "completed_score_rows",
        "last_request_id",
        "partial_sha256",
    ):
        if progress.get(key) != observed[key]:
            raise ValueError(f"native-gate resume progress mismatch: {key}")
    lineage = list(metadata.get("resume_lineage", []))
    lineage.append(
        {
            "resumed_at": _utc_now(),
            "from_status": metadata.get("status"),
            "completed_requests": observed["completed_requests"],
            "partial_sha256": observed["partial_sha256"],
        }
    )
    metadata.update({"status": "initializing", "resume_lineage": lineage})
    _write_json(run_dir / "metadata.json", metadata)
    return metadata, progress, hashlib.sha256(partial_path.read_bytes())


def _audit_partial(
    path: Path, records: Sequence[ModelRecord], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError("native-gate partial file has an incomplete line")
    rows_count = 0
    last_request = None
    lines = content.splitlines()
    if len(lines) > len(records):
        raise ValueError("native-gate partial exceeds request population")
    for ordinal, encoded in enumerate(lines):
        row = json.loads(encoded)
        record = records[ordinal]
        if row.get("ordinal") != ordinal or row.get("request_id") != record.request_id:
            raise ValueError("native-gate partial request identity/order mismatch")
        if row.get("block_zero_based") != metadata.get("block_zero_based"):
            raise ValueError("native-gate partial block mismatch")
        score_rows = row.get("rows")
        if row.get("rows_sha256") != _canonical_sha256(score_rows):
            raise ValueError("native-gate partial row hash mismatch")
        expected_ids = [str(value["item_id"]) for value in record.candidates]
        if [value.get("candidate_item_id") for value in score_rows] != expected_ids:
            raise ValueError("native-gate candidate identity/order mismatch")
        for value in score_rows:
            conditions = value.get("conditions", {})
            if len(conditions) != len(SCORE_CONDITIONS) or set(conditions) != set(
                SCORE_CONDITIONS
            ):
                raise ValueError("native-gate score condition set mismatch")
            for name in SCORE_CONDITIONS:
                condition = conditions[name]
                if not math.isfinite(float(condition.get("score"))):
                    raise ValueError("native-gate partial score is non-finite")
                terms = condition.get("terms")
                if len(terms) != 4 or not all(math.isfinite(float(x)) for x in terms):
                    raise ValueError("native-gate partial term is non-finite")
        rows_count += len(score_rows)
        last_request = record.request_id
    return {
        "completed_requests": len(lines),
        "completed_score_rows": rows_count,
        "last_request_id": last_request,
        "partial_sha256": hashlib.sha256(content).hexdigest(),
    }


def _finalize(
    run_dir: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    records: Sequence[ModelRecord],
) -> dict[str, Any]:
    observed = _audit_partial(run_dir / "scores.partial.jsonl", records, metadata)
    expected_rows = sum(len(row.candidates) for row in records)
    if observed["completed_requests"] != len(records) or observed[
        "completed_score_rows"
    ] != expected_rows:
        raise ValueError("cannot finalize incomplete native-gate coverage")
    maximum = max(float(value) for value in metadata["max_identity_delta"].values())
    identity_passed = maximum <= float(metadata["identity_tolerance"])
    scores_path = run_dir / "scores.jsonl"
    temporary = run_dir / ".scores.writing.jsonl"
    with (run_dir / "scores.partial.jsonl").open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            block = json.loads(line)
            for row in block["rows"]:
                target.write(_canonical_json(row) + "\n")
    temporary.replace(scores_path)
    metadata.update(
        {
            "status": "completed" if identity_passed else "mechanical_failure",
            "resumable": False,
            "request_count": len(records),
            "score_rows": expected_rows,
            "scores_path": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "complete_finite_score_coverage": True,
            "identity_passed": identity_passed,
            "maximum_identity_delta": maximum,
            "qrels_read": False,
            "completed_at": _utc_now(),
        }
    )
    _write_json(run_dir / "metadata.json", metadata)
    progress.update(
        {
            **observed,
            "scores_sha256": metadata["scores_sha256"],
            "status": metadata["status"],
            "updated_at": _utc_now(),
        }
    )
    _write_json(run_dir / "progress.json", progress)
    return metadata


def _load_manifest(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != (
        "frozen_before_transformer_deep_dive_outcomes"
    ):
        raise ValueError("deep-dive manifest is not frozen")
    if payload.get("registered_design", {}).get("q3_native_positions") != [
        "shared_prompt_last",
        "teacher_forced_yes_last",
        "teacher_forced_no_last",
    ]:
        raise ValueError("deep-dive Q3 native-position contract drifted")
    return {**payload, "path": path, "sha256": sha256_file(path)}


def _append_sync(path: Path, line: str) -> None:
    with path.open("ab") as handle:
        handle.write(line.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return sha256_text(_canonical_json(value))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + f".writing.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
