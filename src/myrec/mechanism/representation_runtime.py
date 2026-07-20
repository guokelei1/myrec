"""Resumable, qrels-blind M2 activation extraction and patch scoring runtime."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import (
    FORBIDDEN_MODEL_INPUT_FIELDS,
    HISTORY_INPUT_FIELDS,
    ModelRecord,
    sanitize_record_for_model,
)
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
from myrec.mechanism.representation_probe import (
    CANDIDATE_POSITIONS,
    M2_ANCHORS,
    M2_CONDITIONS,
    M2_HIDDEN_STATE_INDICES,
    M2_TRAIN_REQUESTS,
    PREFERENCE_POSITION,
    REQUEST_POSITIONS,
    MechanicalPositionError,
    InstrumentedPrompt,
    instrument_pointwise_prompt,
    load_m2_probe_manifest,
    normalize_query,
    select_train_probe_records,
    write_activation_shard,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


MAX_WALL_SECONDS = 13_500.0
DEFAULT_REQUESTS_PER_SHARD = 32
METADATA_FILENAME = "metadata.json"
INDEX_FILENAME = "index.json"
PARTIAL_INDEX_FILENAME = "index.partial.json"
PROGRESS_FILENAME = "progress.json"
_ASSIGNMENT_FIELDS = frozenset({"request_id", "condition_id", "history"})


@dataclass(frozen=True)
class RecordActivations:
    request: np.ndarray
    candidate: np.ndarray
    candidate_ids: tuple[str, ...]
    prompt_boundary_rows: int


class SelectedStateCapture:
    """Capture state 0 and four block outputs only at requested token positions."""

    def __init__(self, model: Any) -> None:
        import torch

        self.model = model
        self.torch = torch
        self.layers = resolve_transformer_layers(model)
        self.embedding = model.get_input_embeddings()
        if self.embedding is None:
            raise TypeError("M2 model has no input embedding module")
        self._positions: Any = None
        self._captured: dict[int, Any] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "SelectedStateCapture":
        self._handles.append(self.embedding.register_forward_hook(self._hook(0)))
        for state_index in M2_HIDDEN_STATE_INDICES[1:]:
            block = state_index - 1
            self._handles.append(
                self.layers[block].register_forward_hook(self._hook(state_index))
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._positions = None
        self._captured.clear()

    def capture(
        self,
        input_ids: Any,
        attention_mask: Any,
        positions: Any,
    ) -> np.ndarray:
        torch = self.torch
        if positions.ndim != 2 or positions.shape[0] != input_ids.shape[0]:
            raise ValueError("capture positions must have shape [batch, positions]")
        if positions.device != input_ids.device:
            positions = positions.to(input_ids.device)
        if int(positions.min()) < 0 or int(positions.max()) >= input_ids.shape[1]:
            raise ValueError("activation capture position is outside padded input")
        self._positions = positions
        self._captured = {}
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        missing = set(M2_HIDDEN_STATE_INDICES) - set(self._captured)
        if missing:
            raise RuntimeError(f"activation hooks did not fire for states={sorted(missing)}")
        result = torch.stack(
            [self._captured[state] for state in M2_HIDDEN_STATE_INDICES], dim=2
        )
        # [batch, positions, states, hidden]
        values = result.float().cpu().numpy()
        if values.ndim != 4 or not np.isfinite(values).all():
            raise FloatingPointError("captured activation tensor is invalid")
        self._positions = None
        self._captured = {}
        return values

    def _hook(self, state_index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if self._positions is None:
                raise RuntimeError("activation hook fired outside capture call")
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3:
                raise ValueError(f"hidden state {state_index} is not rank three")
            rows = self.torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            selected = hidden[rows, self._positions]
            self._captured[state_index] = selected.detach()

        return capture


def resolve_transformer_layers(model: Any) -> Any:
    """Resolve the unique 28-block Qwen ModuleList through PEFT wrappers."""

    import torch

    candidates: list[Any] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        if (
            isinstance(module, torch.nn.ModuleList)
            and len(module) == 28
            and (name == "layers" or name.endswith(".layers"))
            and id(module) not in seen
        ):
            seen.add(id(module))
            candidates.append(module)
    if len(candidates) != 1:
        raise TypeError(
            f"expected one 28-block Qwen ModuleList, observed {len(candidates)}"
        )
    return candidates[0]


def extract_m2_activations(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    role: str,
    condition_id: str,
    device: str,
    runs_dir: str | Path = "runs",
    assignment_path: str | Path | None = None,
    assignment_manifest_path: str | Path | None = None,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    requests_per_shard: int = DEFAULT_REQUESTS_PER_SHARD,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Extract a train or exhaustive internal-dev activation bundle.

    ``max_requests`` permanently marks the bundle as ``smoke_non_result``.
    Wall-time exits commit only complete independently hashed shards.
    """

    _validate_run_id(run_id)
    if role not in {"train_probe", "dev_representation"}:
        raise ValueError("role must be train_probe or dev_representation")
    if role == "train_probe" and condition_id != "full":
        raise ValueError("train probe activations are frozen to condition=full")
    if role == "dev_representation" and condition_id not in M2_CONDITIONS:
        raise ValueError(f"unsupported M2 representation condition={condition_id}")
    if not str(device).strip():
        raise ValueError("an explicit extraction device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("M2 max_wall_seconds must be in (0, 13500]")
    if requests_per_shard <= 0:
        raise ValueError("requests_per_shard must be positive")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("max_requests must be positive")

    probe_manifest = load_m2_probe_manifest()
    implementation_identity = representation_runtime_implementation_identity()
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    split = "train" if role == "train_probe" else "dev"
    records_path = standardized_dir / f"records_{split}.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    for path in (
        records_path,
        dataset_manifest_path,
        candidate_manifest_path,
        request_manifest_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    if method_id not in M2_ANCHORS:
        raise ValueError("M2 deep activation extraction admits only Q2/Q3")
    frozen_model = probe_manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("M2 anchor config hash differs from frozen probe manifest")
    checkpoint_dir = checkpoint_root / CHECKPOINT_DIRNAME
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(
        training_metadata,
        config,
        allow_smoke=max_requests is not None,
    )
    checkpoint_id, checkpoint_weight_files = _checkpoint_identity(
        checkpoint_dir / "model", method_id
    )
    if checkpoint_id != frozen_model["checkpoint_id"]:
        raise ValueError("M2 anchor checkpoint differs from frozen probe manifest")
    if checkpoint_id != training_metadata.get("checkpoint_id"):
        raise ValueError("M2 checkpoint weights changed after training metadata")

    frozen_inputs = probe_manifest["frozen_inputs"]
    hashes = {
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "records_sha256": sha256_file(records_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
    }
    expected_records_hash = frozen_inputs[f"records_{split}_sha256"]
    expected = {
        "dataset_manifest_sha256": frozen_inputs["dataset_manifest_sha256"],
        "candidate_manifest_sha256": frozen_inputs["candidate_manifest_sha256"],
        "request_manifest_sha256": frozen_inputs["request_manifest_sha256"],
        "records_sha256": expected_records_hash,
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise ValueError(f"frozen M2 input hash mismatch: {key}")

    raw_records = list(iter_jsonl(records_path))
    if role == "train_probe":
        records, selection_audit = select_train_probe_records(
            raw_records, limit=M2_TRAIN_REQUESTS
        )
    else:
        records = [sanitize_record_for_model(row) for row in raw_records]
        selection_audit = {
            "selection": "full_internal_dev_in_frozen_record_order",
            "selection_label_free": True,
            "selected_requests": len(records),
        }
        if len(records) != 8000:
            raise ValueError("M2 dev activation population must contain 8000 requests")

    assignments, assignment_identity = _condition_histories(
        condition_id=condition_id,
        role=role,
        records=records,
        all_dev_records=(
            [sanitize_record_for_model(row) for row in raw_records]
            if role == "dev_representation"
            else records
        ),
        assignment_path=assignment_path,
        assignment_manifest_path=assignment_manifest_path,
    )
    evidence_mode = "smoke_non_result" if max_requests is not None else "mechanism_diagnostic"
    target_records = records[: int(max_requests)] if max_requests is not None else records
    result_eligible = max_requests is None
    run_contract = {
        "schema_version": 1,
        "run_id": run_id,
        "role": role,
        "condition_id": condition_id,
        "method_id": method_id,
        "device": str(device),
        "config_sha256": config["_config_sha256"],
        "checkpoint_id": checkpoint_id,
        "records_sha256": hashes["records_sha256"],
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "mechanism_probe_manifest_sha256": probe_manifest["sha256"],
        "assignment_sha256": assignment_identity.get("assignment_sha256"),
        "assignment_manifest_sha256": assignment_identity.get(
            "assignment_manifest_sha256"
        ),
        "target_requests": len(target_records),
        "requests_per_shard": int(requests_per_shard),
        "evidence_mode": evidence_mode,
        "implementation_digest": implementation_identity["digest"],
    }
    contract_sha256 = _canonical_sha256(run_contract)
    base_metadata = {
        "schema_version": 1,
        "analysis_stage": "m2_representation",
        "bundle_role": role,
        "condition_id": condition_id,
        "method_id": method_id,
        "run_id": run_id,
        "split": split,
        "dataset_id": "kuaisearch",
        "dataset_version": "full_confirm_preceding40k_v11",
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_weight_files,
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": hashes["training_metadata_sha256"],
        "records_path": str(records_path),
        **hashes,
        "mechanism_probe_manifest": {
            key: probe_manifest[key]
            for key in ("path", "sha256", "expected_sha256", "verified", "manifest_id")
        },
        "implementation_identity": implementation_identity,
        "selection_audit": selection_audit,
        "assignment": assignment_identity,
        "request_positions": list(REQUEST_POSITIONS),
        "candidate_positions": (
            list(CANDIDATE_POSITIONS) if role == "dev_representation" else []
        ),
        "hidden_state_indices": list(M2_HIDDEN_STATE_INDICES),
        "transformer_blocks_zero_based": [6, 13, 20, 27],
        "preference_classifier_position": PREFERENCE_POSITION,
        "candidate_text_visible_to_preference_classifier": False,
        "position_contract": {
            "query_end": (
                "last tokenizer-offset subtoken covering the exact query text-span "
                "terminal character"
            ),
            "history_summary_end": (
                "last tokenizer-offset subtoken covering the exact serialized-history "
                "terminal character; null uses the [NO_HISTORY] marker terminal character"
            ),
            "candidate_readout": "last frozen prompt token used to emit first answer token",
            "offset_failure_or_history_truncation": "mechanical_failure_no_fallback",
            "request_activation_batching": (
                "single_first_candidate_prompt_for_train_dev_equivalence; "
                "position remains strictly before candidate text"
            ),
            "candidate_activation_batching": (
                "frozen scoring.batch_size and frozen candidate order; Q3 uses "
                "the equal-length frozen Yes target suffix to match scoring-kernel context"
            ),
        },
        "activation_passes": {
            "positions_share_same_forward": False,
            "request_level_query_history": {
                "context": "prompt_only",
                "batching": "single_first_candidate_prompt",
                "causal_before_candidate_text": True,
                "positions": list(REQUEST_POSITIONS),
            },
            "candidate_readout_donor": {
                "context": (
                    "prompt_plus_fixed_yes_target_scoring_kernel"
                    if method_id == "q3_tallrec_generalqwen"
                    else "prompt_only_frozen_scoring_kernel"
                ),
                "batching": "frozen_scoring_batch_size_and_candidate_order",
                "position": CANDIDATE_POSITIONS[0],
                "q3_yes_no_target_length_equal": (
                    None if method_id != "q3_tallrec_generalqwen" else "pending_runtime_check"
                ),
                "q3_target_length_tokens": None,
            },
        },
        "qrels_read": False,
        "source_test_opened": False,
        "evidence_mode": evidence_mode,
        "result_eligible": result_eligible,
        "run_contract": run_contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    state = _prepare_extraction_state(
        run_dir,
        base_metadata=base_metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        resume=resume,
    )
    metadata = state["metadata"]
    partial_index = state["partial_index"]
    completed = int(partial_index["request_count"])
    if completed >= len(target_records):
        return _finalize_activation_bundle(run_dir, metadata, partial_index, target_records)

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_model_and_tokenizer(
            config,
            device=str(device),
            training=False,
            checkpoint_model_dir=checkpoint_dir / "model",
        )
        model.eval()
        metadata.update(_runtime_metadata(method_id, torch, transformers))
        if method_id == "q3_tallrec_generalqwen":
            yes_length = len(_answer_target_tokens(tokenizer, "Yes"))
            no_length = len(_answer_target_tokens(tokenizer, "No"))
            if yes_length != no_length:
                raise MechanicalPositionError(
                    "q3_target_length_mismatch",
                    "frozen Yes/No target lengths differ at runtime",
                )
            metadata["activation_passes"]["candidate_readout_donor"].update(
                {
                    "q3_yes_no_target_length_equal": True,
                    "q3_target_length_tokens": yes_length,
                }
            )
        metadata["status"] = "running"
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode(), SelectedStateCapture(model) as capture:
            while completed < len(target_records):
                if time.monotonic() - started >= max_wall_seconds:
                    metadata.update(
                        {
                            "status": "wall_time_exhausted",
                            "resumable": True,
                            "completed_requests": completed,
                            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
                            + (time.monotonic() - started),
                            "qrels_read": False,
                        }
                    )
                    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
                    return metadata
                stop = min(completed + requests_per_shard, len(target_records))
                shard_records = target_records[completed:stop]
                request_ids: list[str] = []
                normalized_queries: list[str] = []
                request_values: list[np.ndarray] = []
                candidate_values: list[np.ndarray] = []
                candidate_ids: list[str] = []
                offsets = [0]
                boundary_rows = 0
                for record in shard_records:
                    try:
                        value = extract_record_activations(
                            model,
                            tokenizer,
                            capture,
                            record,
                            assignments[record.request_id],
                            config,
                            device=str(device),
                            batch_size=batch_size,
                            include_candidate_readout=role == "dev_representation",
                        )
                    except MechanicalPositionError as exc:
                        metadata.update(
                            {
                                "status": "mechanical_failure",
                                "resumable": False,
                                "qrels_read": False,
                                "mechanical_failure": {
                                    "request_id": record.request_id,
                                    "code": exc.code,
                                    "detail": exc.detail,
                                },
                            }
                        )
                        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
                        raise
                    request_ids.append(record.request_id)
                    normalized_queries.append(normalize_query(record.query))
                    request_values.append(value.request)
                    candidate_values.append(value.candidate)
                    candidate_ids.extend(value.candidate_ids)
                    offsets.append(offsets[-1] + len(value.candidate_ids))
                    boundary_rows += value.prompt_boundary_rows
                hidden_size = int(request_values[0].shape[-1])
                candidate_matrix = (
                    np.concatenate(candidate_values, axis=0)
                    if candidate_values and candidate_values[0].shape[0]
                    else np.empty(
                        (0, len(M2_HIDDEN_STATE_INDICES), hidden_size),
                        dtype=np.float32,
                    )
                )
                shard_name = f"shard_{len(partial_index['shards']):05d}.npz"
                shard = write_activation_shard(
                    run_dir / "shards" / shard_name,
                    request_ids=request_ids,
                    normalized_queries=normalized_queries,
                    request_activations=np.stack(request_values),
                    candidate_offsets=offsets,
                    candidate_ids=candidate_ids,
                    candidate_activations=candidate_matrix,
                )
                shard["start_request_ordinal"] = completed
                shard["prompt_at_max_boundary_rows"] = boundary_rows
                partial_index["shards"].append(shard)
                partial_index["request_count"] += len(shard_records)
                partial_index["candidate_count"] += len(candidate_ids)
                partial_index["prompt_at_max_boundary_rows"] += boundary_rows
                partial_index["rolling_shards_sha256"] = _rolling_digest(
                    partial_index["rolling_shards_sha256"], shard["sha256"]
                )
                completed = stop
                _write_json_atomic(run_dir / PARTIAL_INDEX_FILENAME, partial_index)
                _write_json_atomic(
                    run_dir / PROGRESS_FILENAME,
                    {
                        "schema_version": 1,
                        "run_contract_sha256": contract_sha256,
                        "completed_requests": completed,
                        "completed_candidates": partial_index["candidate_count"],
                        "last_request_id": request_ids[-1],
                        "partial_index_sha256": sha256_file(
                            run_dir / PARTIAL_INDEX_FILENAME
                        ),
                        "status": "running",
                        "updated_at": _utc_now(),
                    },
                )
    except Exception as exc:
        if metadata.get("status") != "mechanical_failure":
            metadata.update(
                {
                    "status": "failed",
                    "resumable": True,
                    "qrels_read": False,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            )
            _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        raise
    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        time.monotonic() - started
    )
    return _finalize_activation_bundle(run_dir, metadata, partial_index, target_records)


def representation_runtime_implementation_identity() -> dict[str, Any]:
    """Hash extractor, shared representation contract, and production CLI."""

    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "src/myrec/mechanism/representation_probe.py",
        root / "src/myrec/mechanism/representation_runtime.py",
        root / "scripts/extract_m2_activations.py",
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


def extract_record_activations(
    model: Any,
    tokenizer: Any,
    capture: SelectedStateCapture,
    record: ModelRecord,
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
    batch_size: int,
    include_candidate_readout: bool,
    instrument_prompt: Any | None = None,
) -> RecordActivations:
    """Extract standardized request states and frozen-batch candidate readouts.

    Request positions always use a dedicated unpadded first-candidate prompt in
    both train and dev.  This removes left-padding batch-composition numerics
    while remaining causally before candidate text.  Candidate readouts use
    the exact frozen scoring batch size/order needed by patch scoring.
    """

    import torch

    if batch_size <= 0:
        raise ValueError("activation batch size must be positive")
    method_id = str(config["method_id"])
    prompt_instrument = instrument_prompt or instrument_pointwise_prompt
    training = config["training"]
    target_reserve = 0
    q3_scoring_suffix: list[int] = []
    if method_id == "q3_tallrec_generalqwen":
        yes_target = _answer_target_tokens(tokenizer, "Yes")
        no_target = _answer_target_tokens(tokenizer, "No")
        if len(yes_target) != len(no_target):
            raise MechanicalPositionError(
                "q3_target_length_mismatch",
                "one shared readout donor requires equal frozen Yes/No target lengths",
            )
        target_reserve = len(yes_target)
        q3_scoring_suffix = yes_target
    max_prompt_length = int(training["max_length"]) - target_reserve
    candidates = list(record.candidates)
    # Standardized request activation: one unpadded first-candidate prompt for
    # both train and dev, with no candidate readout retained from this pass.
    request_prompt = prompt_instrument(
        tokenizer,
        method_id,
        record,
        candidates[0],
        history=history,
        history_budget=int(training["history_budget"]),
        max_length=max_prompt_length,
    )
    if not (
        request_prompt.query_end < request_prompt.candidate_start
        and request_prompt.history_summary_end < request_prompt.candidate_start
    ):
        raise MechanicalPositionError(
            "candidate_leakage_position",
            "request positions are not strictly before candidate text",
        )
    request_ids, request_mask, request_padding = _left_pad_prompts(
        [request_prompt.token_ids], tokenizer.pad_token_id, device
    )
    request_positions = torch.tensor(
        [
            [
                request_padding[0] + request_prompt.query_end,
                request_padding[0] + request_prompt.history_summary_end,
            ]
        ],
        dtype=torch.long,
        device=device,
    )
    request_reference = capture.capture(
        request_ids, request_mask, request_positions
    )[0]
    candidate_values: list[np.ndarray] = []
    candidate_ids: list[str] = []
    boundary_rows = int(request_prompt.prompt_at_max_boundary)
    if not include_candidate_readout:
        hidden_size = int(request_reference.shape[-1])
        return RecordActivations(
            request=request_reference,
            candidate=np.empty(
                (0, len(M2_HIDDEN_STATE_INDICES), hidden_size), dtype=np.float32
            ),
            candidate_ids=(),
            prompt_boundary_rows=boundary_rows,
        )

    # Candidate donor/readout activations follow the frozen scoring chunks.
    boundary_rows = 0
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start : start + batch_size]
        prompts: list[InstrumentedPrompt] = []
        for candidate in chunk:
            prompt = prompt_instrument(
                tokenizer,
                method_id,
                record,
                candidate,
                history=history,
                history_budget=int(training["history_budget"]),
                max_length=max_prompt_length,
            )
            if not (
                prompt.query_end < prompt.candidate_start
                and prompt.history_summary_end < prompt.candidate_start
            ):
                raise MechanicalPositionError(
                    "candidate_leakage_position",
                    "request positions are not strictly before candidate text",
                )
            prompts.append(prompt)
            boundary_rows += int(prompt.prompt_at_max_boundary)
        input_ids, attention_mask, left_padding = _left_pad_prompts(
            [list(value.token_ids) + q3_scoring_suffix for value in prompts],
            tokenizer.pad_token_id,
            device,
        )
        position_rows = []
        for prompt, padding in zip(prompts, left_padding):
            position_rows.append([padding + prompt.candidate_readout])
        positions = torch.tensor(position_rows, dtype=torch.long, device=device)
        values = capture.capture(input_ids, attention_mask, positions)
        for row, candidate in enumerate(chunk):
            candidate_values.append(values[row, 0])
            candidate_ids.append(str(candidate["item_id"]))
    hidden_size = int(request_reference.shape[-1])
    candidate_matrix = (
        np.stack(candidate_values)
        if candidate_values
        else np.empty((0, len(M2_HIDDEN_STATE_INDICES), hidden_size), dtype=np.float32)
    )
    if include_candidate_readout:
        expected_ids = [str(row["item_id"]) for row in record.candidates]
        if candidate_ids != expected_ids:
            raise ValueError("candidate activation identity/order coverage failed")
    return RecordActivations(
        request=request_reference,
        candidate=candidate_matrix,
        candidate_ids=tuple(candidate_ids),
        prompt_boundary_rows=boundary_rows,
    )


def _left_pad_prompts(
    prompts: Sequence[Sequence[int]], pad_token_id: int, device: str
) -> tuple[Any, Any, list[int]]:
    import torch

    if not prompts or any(not value for value in prompts):
        raise ValueError("cannot pad an empty prompt batch")
    maximum = max(len(value) for value in prompts)
    input_ids = torch.full(
        (len(prompts), maximum), int(pad_token_id), dtype=torch.long, device=device
    )
    attention = torch.zeros_like(input_ids)
    padding: list[int] = []
    for row, values in enumerate(prompts):
        left = maximum - len(values)
        padding.append(left)
        input_ids[row, left:] = torch.tensor(values, dtype=torch.long, device=device)
        attention[row, left:] = 1
    return input_ids, attention, padding


def _condition_histories(
    *,
    condition_id: str,
    role: str,
    records: Sequence[ModelRecord],
    all_dev_records: Sequence[ModelRecord],
    assignment_path: str | Path | None,
    assignment_manifest_path: str | Path | None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if condition_id == "full":
        if assignment_path is not None or assignment_manifest_path is not None:
            raise ValueError("full condition does not accept an assignment file")
        return {
            row.request_id: [dict(value) for value in row.history] for row in records
        }, {
            "source": "frozen_record_history",
            "history_is_only_intervention": True,
            "qrels_read": False,
        }
    if condition_id == "null":
        if role != "dev_representation":
            raise ValueError("null activation condition is dev-only")
        if assignment_path is not None or assignment_manifest_path is not None:
            raise ValueError("null condition does not accept an assignment file")
        return {row.request_id: [] for row in records}, {
            "source": "empty_history_serialized_as_[NO_HISTORY]",
            "history_is_only_intervention": True,
            "qrels_read": False,
        }
    if role != "dev_representation" or condition_id not in {"relevant_6", "irrelevant_6"}:
        raise ValueError(f"assignment-backed condition is unsupported: {condition_id}")
    if assignment_path is None or assignment_manifest_path is None:
        raise ValueError(f"condition={condition_id} requires assignment and manifest paths")
    assignment_path = Path(assignment_path)
    assignment_manifest_path = Path(assignment_manifest_path)
    manifest = _read_json(assignment_manifest_path)
    if manifest.get("qrels_read") is not False or manifest.get("source_test_opened") is not False:
        raise ValueError("M1 assignment manifest crossed a label/test boundary")
    if int(manifest.get("candidate_leakage_count", -1)) != 0:
        raise ValueError("M1 assignment manifest reports candidate leakage")
    if int(manifest.get("causality_violation_count", -1)) != 0:
        raise ValueError("M1 assignment manifest reports causality violations")
    condition = manifest.get("conditions", {}).get(condition_id)
    if not isinstance(condition, dict):
        raise ValueError("M1 assignment manifest lacks requested condition")
    observed_assignment_hash = sha256_file(assignment_path)
    if condition.get("sha256") != observed_assignment_hash:
        raise ValueError("M1 assignment bytes differ from their manifest")
    full_by_id = {row.request_id: row for row in all_dev_records}
    result: dict[str, list[dict[str, Any]]] = {}
    for row in iter_jsonl(assignment_path):
        if set(row) != _ASSIGNMENT_FIELDS:
            raise ValueError("M2 assignment rows have unexpected fields")
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in result or request_id not in full_by_id:
            raise ValueError(f"invalid M2 assignment request_id={request_id!r}")
        if row.get("condition_id") != condition_id:
            raise ValueError("M2 assignment condition mismatch")
        history = row.get("history")
        if not isinstance(history, list):
            raise ValueError("M2 assignment history must be a list")
        for event in history:
            if not isinstance(event, dict) or set(event) - set(HISTORY_INPUT_FIELDS):
                raise ValueError("M2 assignment history escaped the input whitelist")
            if _contains_forbidden(event):
                raise ValueError("M2 assignment history contains a forbidden label field")
        projected = sanitize_record_for_model(
            {
                "request_id": request_id,
                "query": "m2-assignment-audit",
                "history": history,
                "candidates": [{"item_id": "a"}, {"item_id": "b"}],
            }
        )
        result[request_id] = [dict(value) for value in projected.history]
    expected_ids = [row.request_id for row in all_dev_records]
    if list(result) != expected_ids:
        raise ValueError("M2 assignment request identity/order coverage mismatch")
    selected = {row.request_id: result[row.request_id] for row in records}
    return selected, {
        "source": "audited_m1_assignment",
        "condition_id": condition_id,
        "assignment_path": str(assignment_path),
        "assignment_sha256": observed_assignment_hash,
        "assignment_manifest_path": str(assignment_manifest_path),
        "assignment_manifest_sha256": sha256_file(assignment_manifest_path),
        "history_is_only_intervention": True,
        "candidate_leakage_audit_passed": True,
        "causality_audit_passed": True,
        "qrels_read": False,
    }


def _prepare_extraction_state(
    run_dir: Path,
    *,
    base_metadata: dict[str, Any],
    contract_sha256: str,
    records: Sequence[ModelRecord],
    resume: bool,
) -> dict[str, Any]:
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"M2 run directory is not empty: {run_dir}")
        (run_dir / "shards").mkdir(parents=True, exist_ok=True)
        metadata = dict(base_metadata)
        metadata.update(
            {"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []}
        )
        partial = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "request_count": 0,
            "candidate_count": 0,
            "prompt_at_max_boundary_rows": 0,
            "rolling_shards_sha256": hashlib.sha256(b"").hexdigest(),
            "shards": [],
        }
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        _write_json_atomic(run_dir / PARTIAL_INDEX_FILENAME, partial)
        _write_json_atomic(
            run_dir / PROGRESS_FILENAME,
            {
                "schema_version": 1,
                "run_contract_sha256": contract_sha256,
                "completed_requests": 0,
                "completed_candidates": 0,
                "last_request_id": None,
                "partial_index_sha256": sha256_file(run_dir / PARTIAL_INDEX_FILENAME),
                "status": "initializing",
                "updated_at": _utc_now(),
            },
        )
        return {"metadata": metadata, "partial_index": partial}
    metadata = _read_json(run_dir / METADATA_FILENAME)
    partial = _read_json(run_dir / PARTIAL_INDEX_FILENAME)
    progress = _read_json(run_dir / PROGRESS_FILENAME)
    if metadata.get("run_contract_sha256") != contract_sha256:
        raise ValueError("M2 resume metadata contract drift")
    if partial.get("run_contract_sha256") != contract_sha256:
        raise ValueError("M2 resume partial-index contract drift")
    if progress.get("run_contract_sha256") != contract_sha256:
        raise ValueError("M2 resume progress contract drift")
    if metadata.get("status") not in {"initializing", "running", "failed", "wall_time_exhausted"}:
        raise ValueError(f"M2 activation run is not resumable: {metadata.get('status')}")
    _audit_partial_index(run_dir, partial, records)
    if int(progress.get("completed_requests", -1)) != int(partial["request_count"]):
        raise ValueError("M2 resume progress/request count mismatch")
    if progress.get("partial_index_sha256") != sha256_file(
        run_dir / PARTIAL_INDEX_FILENAME
    ):
        raise ValueError("M2 resume progress/partial-index hash mismatch")
    lineage = metadata.get("resume_lineage", [])
    if not isinstance(lineage, list):
        raise ValueError("M2 activation resume lineage is invalid")
    lineage.append(
        {
            "resumed_at": _utc_now(),
            "from_status": metadata.get("status"),
            "completed_requests": int(partial["request_count"]),
            "prior_error": metadata.get("error"),
            "partial_index_sha256": sha256_file(
                run_dir / PARTIAL_INDEX_FILENAME
            ),
        }
    )
    metadata.update(
        {"status": "initializing", "error": None, "resume_lineage": lineage}
    )
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    return {"metadata": metadata, "partial_index": partial}


def _audit_partial_index(
    run_dir: Path, partial: Mapping[str, Any], records: Sequence[ModelRecord]
) -> None:
    observed_ids: list[str] = []
    candidates = 0
    rolling = hashlib.sha256(b"").hexdigest()
    expected_start = 0
    for shard in partial.get("shards", []):
        if int(shard.get("start_request_ordinal", -1)) != expected_start:
            raise ValueError("M2 partial shard ordinal is not contiguous")
        path = run_dir / "shards" / str(shard.get("path"))
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError("M2 partial shard hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            request_ids = [str(value) for value in payload["request_ids"].tolist()]
            candidate_ids = [str(value) for value in payload["candidate_ids"].tolist()]
        if request_ids != [
            row.request_id
            for row in records[expected_start : expected_start + len(request_ids)]
        ]:
            raise ValueError("M2 partial shard request identity/order mismatch")
        if int(shard.get("request_count", -1)) != len(request_ids):
            raise ValueError("M2 partial shard request count mismatch")
        if int(shard.get("candidate_count", -1)) != len(candidate_ids):
            raise ValueError("M2 partial shard candidate count mismatch")
        expected_start += len(request_ids)
        candidates += len(candidate_ids)
        observed_ids.extend(request_ids)
        rolling = _rolling_digest(rolling, str(shard["sha256"]))
    if int(partial.get("request_count", -1)) != len(observed_ids):
        raise ValueError("M2 partial index request count mismatch")
    if int(partial.get("candidate_count", -1)) != candidates:
        raise ValueError("M2 partial index candidate count mismatch")
    if partial.get("rolling_shards_sha256") != rolling:
        raise ValueError("M2 partial index rolling digest mismatch")


def _finalize_activation_bundle(
    run_dir: Path,
    metadata: dict[str, Any],
    partial_index: dict[str, Any],
    records: Sequence[ModelRecord],
) -> dict[str, Any]:
    _audit_partial_index(run_dir, partial_index, records)
    if int(partial_index["request_count"]) != len(records):
        raise ValueError("cannot finalize incomplete M2 activation coverage")
    role = str(metadata["bundle_role"])
    expected_candidates = (
        sum(len(row.candidates) for row in records)
        if role == "dev_representation"
        else 0
    )
    if int(partial_index["candidate_count"]) != expected_candidates:
        raise ValueError("cannot finalize incomplete M2 candidate activation coverage")
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "qrels_read": False,
            "request_count": len(records),
            "candidate_count": expected_candidates,
            "prompt_at_max_boundary_rows": partial_index[
                "prompt_at_max_boundary_rows"
            ],
            "complete_finite_activation_coverage": True,
            "mechanical_failure": None,
            "completed_at": _utc_now(),
        }
    )
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    final_index = dict(partial_index)
    final_index.update(
        {
            "metadata_sha256": sha256_file(run_dir / METADATA_FILENAME),
            "status": "completed",
        }
    )
    final_index.pop("run_contract_sha256", None)
    _write_json_atomic(run_dir / INDEX_FILENAME, final_index)
    _write_json_atomic(
        run_dir / PROGRESS_FILENAME,
        {
            "schema_version": 1,
            "run_contract_sha256": metadata["run_contract_sha256"],
            "completed_requests": len(records),
            "completed_candidates": expected_candidates,
            "last_request_id": records[-1].request_id,
            "index_sha256": sha256_file(run_dir / INDEX_FILENAME),
            "status": "completed",
            "updated_at": _utc_now(),
        },
    )
    return metadata


def _contains_forbidden(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            key in FORBIDDEN_MODEL_INPUT_FIELDS or _contains_forbidden(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden(child) for child in value)
    return False


def _rolling_digest(previous: str, current: str) -> str:
    return hashlib.sha256(f"{previous}\n{current}".encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
