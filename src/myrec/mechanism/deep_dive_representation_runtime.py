"""Resumable qrels-blind extraction of all 29 Qwen residual states.

This runtime is deliberately separate from the frozen five-state M2 extractor.
It reuses the frozen prompt builder and scoring-context conventions, but gives
the Transformer deep-dive its own manifest binding, shard schema, provenance,
and resume contract.
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
from myrec.mechanism.representation_probe import (
    M2_TRAIN_REQUESTS,
    normalize_query,
    select_train_probe_records,
)
from myrec.mechanism.representation_runtime import (
    extract_record_activations,
    resolve_transformer_layers,
)
from myrec.mechanism.q0_representation_prompt import instrument_q0_pointwise_prompt
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


DEEP_DIVE_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_deep_dive_manifest.yaml"
)
ALL_HIDDEN_STATE_INDICES = tuple(range(29))
REQUEST_POSITIONS = ("query_end", "history_summary_end")
MAX_WALL_SECONDS = 13_500.0
DEFAULT_REQUESTS_PER_SHARD = 32
SCHEMA_VERSION = 1


class AllResidualStateCapture:
    """Capture embedding state 0 and every one of 28 block outputs.

    Hooks gather only registered token positions on-device. Full sequence
    hidden tensors are neither retained nor copied to CPU.
    """

    def __init__(self, model: Any) -> None:
        import torch

        self.model = model
        self.torch = torch
        self.layers = resolve_transformer_layers(model)
        self.embedding = model.get_input_embeddings()
        if self.embedding is None:
            raise TypeError("deep-dive model has no input embedding module")
        self._positions: Any = None
        self._captured: dict[int, Any] = {}
        self._fire_counts: dict[int, int] = {}
        self._handles: list[Any] = []
        self.capture_calls = 0

    def __enter__(self) -> "AllResidualStateCapture":
        self._handles.append(self.embedding.register_forward_hook(self._hook(0)))
        for state_index in ALL_HIDDEN_STATE_INDICES[1:]:
            self._handles.append(
                self.layers[state_index - 1].register_forward_hook(
                    self._hook(state_index)
                )
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._positions = None
        self._captured.clear()
        self._fire_counts.clear()

    def capture(self, input_ids: Any, attention_mask: Any, positions: Any) -> np.ndarray:
        torch = self.torch
        if positions.ndim != 2 or positions.shape[0] != input_ids.shape[0]:
            raise ValueError("capture positions must have shape [batch, positions]")
        if positions.device != input_ids.device:
            positions = positions.to(input_ids.device)
        if int(positions.min()) < 0 or int(positions.max()) >= input_ids.shape[1]:
            raise ValueError("activation capture position is outside padded input")
        self._positions = positions
        self._captured = {}
        self._fire_counts = {}
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        observed = set(self._captured)
        expected = set(ALL_HIDDEN_STATE_INDICES)
        if observed != expected:
            raise RuntimeError(
                f"all-state hooks differ: missing={sorted(expected-observed)}, "
                f"extra={sorted(observed-expected)}"
            )
        bad_counts = {
            index: count for index, count in self._fire_counts.items() if count != 1
        }
        if bad_counts:
            raise RuntimeError(f"all-state hook fire counts are not one: {bad_counts}")
        result = torch.stack(
            [self._captured[index] for index in ALL_HIDDEN_STATE_INDICES], dim=2
        )
        values = result.float().cpu().numpy()
        if values.ndim != 4 or not np.isfinite(values).all():
            raise FloatingPointError("all-state activation tensor is invalid")
        self.capture_calls += 1
        self._positions = None
        self._captured = {}
        self._fire_counts = {}
        return values

    def _hook(self, state_index: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if self._positions is None:
                raise RuntimeError("all-state hook fired outside an armed capture")
            self._fire_counts[state_index] = self._fire_counts.get(state_index, 0) + 1
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3:
                raise ValueError(f"hidden state {state_index} is not rank three")
            rows = self.torch.arange(hidden.shape[0], device=hidden.device)[:, None]
            self._captured[state_index] = hidden[rows, self._positions].detach()

        return capture


def write_all_state_shard(
    path: str | Path,
    *,
    request_ids: Sequence[str],
    normalized_queries: Sequence[str],
    request_activations: np.ndarray,
    candidate_offsets: Sequence[int],
    candidate_ids: Sequence[str],
    candidate_activations: np.ndarray,
) -> dict[str, Any]:
    """Atomically write one finite, identity-complete all-state shard."""

    path = Path(path)
    request_values = np.asarray(request_activations)
    candidate_values = np.asarray(candidate_activations)
    offsets = np.asarray(candidate_offsets, dtype=np.int64)
    expected_request_shape = (len(REQUEST_POSITIONS), len(ALL_HIDDEN_STATE_INDICES))
    if request_values.ndim != 4 or request_values.shape[1:3] != expected_request_shape:
        raise ValueError("all-state request activation shard has invalid shape")
    if candidate_values.ndim != 3 or candidate_values.shape[1] != len(
        ALL_HIDDEN_STATE_INDICES
    ):
        raise ValueError("all-state candidate activation shard has invalid shape")
    if request_values.shape[0] != len(request_ids):
        raise ValueError("request activation/identity count mismatch")
    if len(normalized_queries) != len(request_ids):
        raise ValueError("normalized-query/request count mismatch")
    if offsets.shape != (len(request_ids) + 1,):
        raise ValueError("candidate offsets have invalid shape")
    if int(offsets[0]) != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError("candidate offsets must be monotone from zero")
    if int(offsets[-1]) != len(candidate_ids) or len(candidate_ids) != len(
        candidate_values
    ):
        raise ValueError("candidate offsets/identity/activation count mismatch")
    if request_values.shape[-1] != candidate_values.shape[-1]:
        raise ValueError("request/candidate hidden sizes differ")
    if len(set(map(str, request_ids))) != len(request_ids):
        raise ValueError("all-state shard contains duplicate request IDs")
    if not np.isfinite(request_values).all() or not np.isfinite(candidate_values).all():
        raise FloatingPointError("all-state shard contains non-finite values")

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing.npz")
    if temporary.exists():
        temporary.unlink()
    np.savez(
        temporary,
        request_ids=np.asarray(request_ids, dtype=np.str_),
        normalized_queries=np.asarray(normalized_queries, dtype=np.str_),
        request_activations=request_values.astype(np.float16, copy=False),
        candidate_offsets=offsets,
        candidate_ids=np.asarray(candidate_ids, dtype=np.str_),
        candidate_activations=candidate_values.astype(np.float16, copy=False),
        hidden_state_indices=np.asarray(ALL_HIDDEN_STATE_INDICES, dtype=np.int64),
        request_positions=np.asarray(REQUEST_POSITIONS, dtype=np.str_),
    )
    temporary.replace(path)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "request_count": len(request_ids),
        "candidate_count": len(candidate_ids),
        "first_request_id": str(request_ids[0]) if request_ids else None,
        "last_request_id": str(request_ids[-1]) if request_ids else None,
    }


def extract_deep_dive_representations(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    run_id: str,
    *,
    role: str,
    condition_id: str,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = DEEP_DIVE_MANIFEST_PATH,
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    requests_per_shard: int = DEFAULT_REQUESTS_PER_SHARD,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Extract train/full or internal-dev/full|null all-layer bundles."""

    _validate_run_id(run_id)
    if role not in {"train_probe", "dev_representation"}:
        raise ValueError("role must be train_probe or dev_representation")
    if role == "train_probe" and condition_id != "full":
        raise ValueError("deep-dive train activations are full-only")
    if role == "dev_representation" and condition_id not in {"full", "null"}:
        raise ValueError("deep-dive dev activations admit only full or null")
    if not str(device).strip():
        raise ValueError("an explicit extraction device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("max_wall_seconds must be in (0, 13500]")
    if int(requests_per_shard) <= 0:
        raise ValueError("requests_per_shard must be positive")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("max_requests must be positive")

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    manifest = _load_manifest(manifest_path)
    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    supported = {
        "q0_qwen3_reranker_06b",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
    }
    if method_id not in supported:
        raise ValueError("all-layer extraction admits Q0 plus frozen Q2/Q3 anchors")
    if method_id == "q0_qwen3_reranker_06b" and role != "dev_representation":
        raise ValueError("Q0 breadth extraction is dev-only and does not fit a new probe")
    frozen_model = manifest["frozen_inputs"]["models"][method_id]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("all-layer config differs from frozen deep-dive manifest")

    split = "train" if role == "train_probe" else "dev"
    records_path = standardized_dir / f"records_{split}.jsonl"
    dataset_manifest_path = standardized_dir / "manifest.json"
    request_manifest_path = standardized_dir / "request_manifest.json"
    candidate_manifest_path = standardized_dir / "candidate_manifest.json"
    paths_and_hashes = {
        records_path: manifest["frozen_inputs"][f"records_{split}_sha256"],
        dataset_manifest_path: manifest["frozen_inputs"]["dataset_manifest_sha256"],
        request_manifest_path: manifest["frozen_inputs"]["request_manifest_sha256"],
        candidate_manifest_path: manifest["frozen_inputs"]["candidate_manifest_sha256"],
    }
    for path, expected in paths_and_hashes.items():
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen deep-dive input hash mismatch: {path}")

    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(checkpoint_model_dir, method_id)
    if checkpoint_id != frozen_model["checkpoint_id"]:
        raise ValueError("all-layer checkpoint differs from frozen manifest")
    if training_metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("all-layer checkpoint changed after training metadata")

    raw_records = list(iter_jsonl(records_path))
    if role == "train_probe":
        records, selection = select_train_probe_records(
            raw_records, limit=M2_TRAIN_REQUESTS
        )
    else:
        records = [sanitize_record_for_model(row) for row in raw_records]
        if len(records) != 8000:
            raise ValueError("all-layer dev population must contain 8000 requests")
        selection = {
            "selection": "full_internal_dev_in_frozen_record_order",
            "selection_label_free": True,
            "selected_requests": len(records),
            "selected_request_ids_sha256": _identity_digest(
                [row.request_id for row in records]
            ),
        }
    target_records = records if max_requests is None else records[: int(max_requests)]
    histories = {
        row.request_id: ([dict(value) for value in row.history] if condition_id == "full" else [])
        for row in target_records
    }
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests is None
        else "mechanical_smoke_non_result"
    )
    implementation = deep_dive_representation_implementation_identity()
    run_contract = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "role": role,
        "condition_id": condition_id,
        "method_id": method_id,
        "device": str(device),
        "config_sha256": config["_config_sha256"],
        "checkpoint_id": checkpoint_id,
        "records_sha256": sha256_file(records_path),
        "deep_dive_manifest_sha256": manifest["sha256"],
        "hidden_state_indices": list(ALL_HIDDEN_STATE_INDICES),
        "target_request_ids_sha256": _identity_digest(
            [row.request_id for row in target_records]
        ),
        "target_requests": len(target_records),
        "requests_per_shard": int(requests_per_shard),
        "evidence_mode": evidence_mode,
        "implementation_digest": implementation["digest"],
    }
    contract_sha256 = _canonical_sha256(run_contract)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "analysis_stage": "transformer_deep_dive_d1_all_layer_representation",
        "run_id": run_id,
        "method_id": method_id,
        "bundle_role": role,
        "condition_id": condition_id,
        "split": split,
        "dataset_id": "kuaisearch",
        "dataset_version": "full_confirm_preceding40k_v11",
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "records_path": str(records_path),
        "records_sha256": sha256_file(records_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "deep_dive_manifest_path": str(manifest["path"]),
        "deep_dive_manifest_sha256": manifest["sha256"],
        "implementation_identity": implementation,
        "selection_audit": selection,
        "request_positions": list(REQUEST_POSITIONS),
        # D1 decodes all three registered positions. Train qrels are opened
        # only after this label-free bundle passes integrity checks, so every
        # train candidate state is retained and the target is selected later.
        "candidate_positions": ["candidate_readout"],
        "hidden_state_indices": list(ALL_HIDDEN_STATE_INDICES),
        "candidate_text_visible_to_request_positions": False,
        "activation_storage_dtype": "float16",
        "full_sequence_hidden_states_materialized": False,
        "qrels_read": False,
        "source_test_opened": False,
        "result_eligible": max_requests is None,
        "evidence_mode": evidence_mode,
        "run_contract": run_contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    metadata, partial = _prepare_state(
        run_dir,
        metadata=metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        resume=resume,
    )
    completed = int(partial["request_count"])
    if completed >= len(target_records):
        return _finalize(run_dir, metadata, partial, target_records)

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
        metadata.update(_runtime_metadata(method_id, torch, transformers))
        metadata["status"] = "running"
        _write_json_atomic(run_dir / "metadata.json", metadata)
        batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        with torch.inference_mode(), AllResidualStateCapture(model) as capture:
            while completed < len(target_records):
                if time.monotonic() - started >= max_wall_seconds:
                    metadata.update(
                        {
                            "status": "wall_time_exhausted",
                            "resumable": True,
                            "completed_requests": completed,
                            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
                            + time.monotonic()
                            - started,
                            "capture_calls": capture.capture_calls,
                        }
                    )
                    _write_json_atomic(run_dir / "metadata.json", metadata)
                    return metadata
                stop = min(completed + int(requests_per_shard), len(target_records))
                shard_records = target_records[completed:stop]
                request_ids: list[str] = []
                queries: list[str] = []
                request_values: list[np.ndarray] = []
                candidate_values: list[np.ndarray] = []
                candidate_ids: list[str] = []
                offsets = [0]
                boundary_rows = 0
                for record in shard_records:
                    value = extract_record_activations(
                        model,
                        tokenizer,
                        capture,
                        record,
                        histories[record.request_id],
                        config,
                        device=str(device),
                        batch_size=batch_size,
                        include_candidate_readout=True,
                        instrument_prompt=(
                            instrument_q0_pointwise_prompt
                            if method_id == "q0_qwen3_reranker_06b"
                            else None
                        ),
                    )
                    request_ids.append(record.request_id)
                    queries.append(normalize_query(record.query))
                    request_values.append(value.request)
                    candidate_values.append(value.candidate)
                    candidate_ids.extend(value.candidate_ids)
                    offsets.append(offsets[-1] + len(value.candidate_ids))
                    boundary_rows += value.prompt_boundary_rows
                hidden_size = int(request_values[0].shape[-1])
                candidate_matrix = (
                    np.concatenate(candidate_values, axis=0)
                    if candidate_values
                    else np.empty(
                        (0, len(ALL_HIDDEN_STATE_INDICES), hidden_size),
                        dtype=np.float32,
                    )
                )
                shard_name = f"shard_{len(partial['shards']):05d}.npz"
                shard = write_all_state_shard(
                    run_dir / "shards" / shard_name,
                    request_ids=request_ids,
                    normalized_queries=queries,
                    request_activations=np.stack(request_values),
                    candidate_offsets=offsets,
                    candidate_ids=candidate_ids,
                    candidate_activations=candidate_matrix,
                )
                shard["start_request_ordinal"] = completed
                shard["prompt_at_max_boundary_rows"] = boundary_rows
                partial["shards"].append(shard)
                partial["request_count"] += len(shard_records)
                partial["candidate_count"] += len(candidate_ids)
                partial["prompt_at_max_boundary_rows"] += boundary_rows
                partial["rolling_shards_sha256"] = _rolling_digest(
                    partial["rolling_shards_sha256"], shard["sha256"]
                )
                completed = stop
                _write_json_atomic(run_dir / "index.partial.json", partial)
                _write_json_atomic(
                    run_dir / "progress.json",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_contract_sha256": contract_sha256,
                        "completed_requests": completed,
                        "completed_candidates": partial["candidate_count"],
                        "last_request_id": request_ids[-1],
                        "partial_index_sha256": sha256_file(
                            run_dir / "index.partial.json"
                        ),
                        "status": "running",
                        "updated_at": _utc_now(),
                    },
                )
        metadata["capture_calls"] = capture.capture_calls
    except Exception as exc:
        metadata.update(
            {
                "status": "failed",
                "resumable": True,
                "qrels_read": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json_atomic(run_dir / "metadata.json", metadata)
        raise
    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        time.monotonic() - started
    )
    return _finalize(run_dir, metadata, partial, target_records)


def deep_dive_representation_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "src/myrec/mechanism/deep_dive_representation_runtime.py",
        root / "src/myrec/mechanism/representation_probe.py",
        root / "src/myrec/mechanism/representation_runtime.py",
        root / "src/myrec/mechanism/q0_representation_prompt.py",
        root / "scripts/extract_deep_dive_representations.py",
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
    records: Sequence[Any],
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"all-layer run directory is not empty: {run_dir}")
        (run_dir / "shards").mkdir(parents=True, exist_ok=True)
        partial = {
            "schema_version": SCHEMA_VERSION,
            "run_contract_sha256": contract_sha256,
            "request_count": 0,
            "candidate_count": 0,
            "prompt_at_max_boundary_rows": 0,
            "rolling_shards_sha256": hashlib.sha256(b"").hexdigest(),
            "shards": [],
        }
        _write_json_atomic(run_dir / "metadata.json", metadata)
        _write_json_atomic(run_dir / "index.partial.json", partial)
        _write_json_atomic(
            run_dir / "progress.json",
            {
                "schema_version": SCHEMA_VERSION,
                "run_contract_sha256": contract_sha256,
                "completed_requests": 0,
                "completed_candidates": 0,
                "last_request_id": None,
                "partial_index_sha256": sha256_file(run_dir / "index.partial.json"),
                "status": "initializing",
                "updated_at": _utc_now(),
            },
        )
        return metadata, partial
    old_metadata = _read_json(run_dir / "metadata.json")
    partial = _read_json(run_dir / "index.partial.json")
    progress = _read_json(run_dir / "progress.json")
    for payload in (old_metadata, partial, progress):
        if payload.get("run_contract_sha256") != contract_sha256:
            raise ValueError("all-layer resume contract drift")
    if old_metadata.get("status") not in {
        "initializing",
        "running",
        "failed",
        "wall_time_exhausted",
    }:
        raise ValueError("all-layer run is not resumable")
    _audit_partial(run_dir, partial, records)
    if progress.get("partial_index_sha256") != sha256_file(
        run_dir / "index.partial.json"
    ):
        raise ValueError("all-layer resume progress hash mismatch")
    lineage = list(old_metadata.get("resume_lineage", []))
    lineage.append(
        {
            "resumed_at": _utc_now(),
            "from_status": old_metadata.get("status"),
            "completed_requests": int(partial["request_count"]),
            "partial_index_sha256": sha256_file(run_dir / "index.partial.json"),
        }
    )
    old_metadata.update({"status": "initializing", "resume_lineage": lineage})
    _write_json_atomic(run_dir / "metadata.json", old_metadata)
    return old_metadata, partial


def _audit_partial(run_dir: Path, partial: Mapping[str, Any], records: Sequence[Any]) -> None:
    expected_start = 0
    candidate_count = 0
    rolling = hashlib.sha256(b"").hexdigest()
    for shard in partial.get("shards", []):
        if int(shard.get("start_request_ordinal", -1)) != expected_start:
            raise ValueError("all-layer shard ordinal is not contiguous")
        path = run_dir / "shards" / str(shard["path"])
        if sha256_file(path) != shard.get("sha256"):
            raise ValueError("all-layer partial shard hash mismatch")
        with np.load(path, allow_pickle=False) as payload:
            request_ids = [str(value) for value in payload["request_ids"].tolist()]
            candidate_ids = [str(value) for value in payload["candidate_ids"].tolist()]
            states = payload["hidden_state_indices"].tolist()
        if states != list(ALL_HIDDEN_STATE_INDICES):
            raise ValueError("all-layer shard state indices drifted")
        expected_ids = [
            row.request_id
            for row in records[expected_start : expected_start + len(request_ids)]
        ]
        if request_ids != expected_ids:
            raise ValueError("all-layer partial request identity/order mismatch")
        if int(shard.get("request_count", -1)) != len(request_ids):
            raise ValueError("all-layer partial request count mismatch")
        if int(shard.get("candidate_count", -1)) != len(candidate_ids):
            raise ValueError("all-layer partial candidate count mismatch")
        expected_start += len(request_ids)
        candidate_count += len(candidate_ids)
        rolling = _rolling_digest(rolling, str(shard["sha256"]))
    if int(partial.get("request_count", -1)) != expected_start:
        raise ValueError("all-layer partial request total mismatch")
    if int(partial.get("candidate_count", -1)) != candidate_count:
        raise ValueError("all-layer partial candidate total mismatch")
    if partial.get("rolling_shards_sha256") != rolling:
        raise ValueError("all-layer partial rolling digest mismatch")


def _finalize(
    run_dir: Path,
    metadata: dict[str, Any],
    partial: dict[str, Any],
    records: Sequence[Any],
) -> dict[str, Any]:
    _audit_partial(run_dir, partial, records)
    if int(partial["request_count"]) != len(records):
        raise ValueError("cannot finalize incomplete all-layer request coverage")
    expected_candidates = sum(len(row.candidates) for row in records)
    if int(partial["candidate_count"]) != expected_candidates:
        raise ValueError("cannot finalize incomplete all-layer candidate coverage")
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "request_count": len(records),
            "candidate_count": expected_candidates,
            "prompt_at_max_boundary_rows": partial["prompt_at_max_boundary_rows"],
            "complete_finite_activation_coverage": True,
            "qrels_read": False,
            "completed_at": _utc_now(),
        }
    )
    _write_json_atomic(run_dir / "metadata.json", metadata)
    final_index = dict(partial)
    final_index.pop("run_contract_sha256", None)
    final_index.update(
        {
            "status": "completed",
            "metadata_sha256": sha256_file(run_dir / "metadata.json"),
        }
    )
    _write_json_atomic(run_dir / "index.json", final_index)
    _write_json_atomic(
        run_dir / "progress.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_contract_sha256": metadata["run_contract_sha256"],
            "completed_requests": len(records),
            "completed_candidates": expected_candidates,
            "last_request_id": records[-1].request_id,
            "index_sha256": sha256_file(run_dir / "index.json"),
            "status": "completed",
            "updated_at": _utc_now(),
        },
    )
    return metadata


def _load_manifest(path: str | Path) -> dict[str, Any]:
    import yaml

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("deep-dive manifest is not a mapping")
    if payload.get("status") != "frozen_before_transformer_deep_dive_outcomes":
        raise ValueError("deep-dive manifest is not frozen")
    if payload.get("registered_design", {}).get("hidden_state_indices") != list(
        ALL_HIDDEN_STATE_INDICES
    ):
        raise ValueError("deep-dive manifest all-layer state contract drifted")
    return {**payload, "path": path, "sha256": sha256_file(path)}


def _identity_digest(values: Sequence[str]) -> str:
    return sha256_text(
        json.dumps(list(values), ensure_ascii=False, separators=(",", ":"))
    )


def _rolling_digest(previous: str, current: str) -> str:
    return hashlib.sha256(f"{previous}\0{current}".encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".writing.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
