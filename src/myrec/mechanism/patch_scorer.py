"""Qrels-blind, resumable M2 full-to-null activation patch scorer."""

from __future__ import annotations

import hashlib
import json
import math
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
    _single_token_id,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    load_v12_ranker_config,
)
from myrec.mechanism.representation_probe import (
    M2_HIDDEN_STATE_INDICES,
    M2_PATCH_BLOCKS,
    MechanicalPositionError,
    audit_activation_bundle,
    instrument_pointwise_prompt,
    load_m2_probe_manifest,
)
from myrec.mechanism.representation_runtime import resolve_transformer_layers
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl


MAX_WALL_SECONDS = 13_500.0
PATCH_KINDS = (
    "same_request_full_to_null",
    "full_to_full_identity",
    "cross_request_same_layer",
)
PARTIAL_FILENAME = "scores.partial.jsonl"
SCORES_FILENAME = "scores.jsonl"
METADATA_FILENAME = "metadata.json"
PROGRESS_FILENAME = "progress.json"


class FullActivationDonor:
    """Sequential/cached access to audited full candidate readout activations."""

    def __init__(self, bundle: Any, records: Sequence[ModelRecord]) -> None:
        self.bundle = bundle
        self.records = records
        self._request_locations: dict[str, tuple[int, int]] = {}
        ordinal = 0
        for shard_ordinal, shard in enumerate(bundle.index["shards"]):
            with np.load(
                bundle.root / "shards" / shard["path"], allow_pickle=False
            ) as payload:
                request_ids = [str(value) for value in payload["request_ids"].tolist()]
            for local, request_id in enumerate(request_ids):
                if request_id in self._request_locations:
                    raise ValueError("duplicate donor activation request identity")
                if records[ordinal].request_id != request_id:
                    raise ValueError("donor activation request order differs from records")
                self._request_locations[request_id] = (shard_ordinal, local)
                ordinal += 1
        if ordinal != len(records):
            raise ValueError("donor activation request coverage is incomplete")
        self._cached_ordinal: int | None = None
        self._cached: dict[str, Any] | None = None

    def candidate_vector(
        self, request_id: str, candidate_ordinal: int, hidden_state_index: int
    ) -> np.ndarray:
        if hidden_state_index not in M2_HIDDEN_STATE_INDICES:
            raise ValueError("donor hidden-state index is not preregistered")
        shard_ordinal, local = self._request_locations[request_id]
        if self._cached_ordinal != shard_ordinal:
            shard = self.bundle.index["shards"][shard_ordinal]
            with np.load(
                self.bundle.root / "shards" / shard["path"], allow_pickle=False
            ) as payload:
                self._cached = {key: np.asarray(payload[key]) for key in payload.files}
            self._cached_ordinal = shard_ordinal
        assert self._cached is not None
        offsets = self._cached["candidate_offsets"]
        start = int(offsets[local])
        stop = int(offsets[local + 1])
        if not 0 <= candidate_ordinal < stop - start:
            raise IndexError("donor candidate ordinal is outside request slate")
        state_ordinal = M2_HIDDEN_STATE_INDICES.index(hidden_state_index)
        value = np.asarray(
            self._cached["candidate_activations"][start + candidate_ordinal, state_ordinal],
            dtype=np.float32,
        )
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError("donor activation vector is invalid")
        return value


class ReadoutActivationPatch:
    """Replace one block's post-block state at exact per-row readout positions."""

    def __init__(self, model: Any, block: int) -> None:
        import torch

        if block not in M2_PATCH_BLOCKS:
            raise ValueError("patch block is outside the preregistered set")
        self.torch = torch
        self.layer = resolve_transformer_layers(model)[block]
        self.positions: Any = None
        self.vectors: Any = None
        self.fired = 0
        self.handle: Any = None

    def __enter__(self) -> "ReadoutActivationPatch":
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.positions = None
        self.vectors = None

    def arm(self, positions: Any, vectors: Any) -> None:
        if self.positions is not None:
            raise RuntimeError("activation patch is already armed")
        if positions.ndim != 1 or vectors.ndim != 2 or vectors.shape[0] != positions.shape[0]:
            raise ValueError("activation patch arrays have invalid shape")
        self.positions = positions
        self.vectors = vectors
        self.fired = 0

    def disarm(self) -> None:
        if self.fired != 1:
            raise RuntimeError(f"activation patch hook fired {self.fired} times")
        self.positions = None
        self.vectors = None

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        if self.positions is None or self.vectors is None:
            raise RuntimeError("patch hook fired while unarmed")
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != self.positions.shape[0]:
            raise ValueError("patch block output shape mismatch")
        if int(self.positions.min()) < 0 or int(self.positions.max()) >= hidden.shape[1]:
            raise ValueError("patch readout position is outside the sequence")
        replacement = self.vectors.to(device=hidden.device, dtype=hidden.dtype)
        if replacement.shape != (hidden.shape[0], hidden.shape[2]):
            raise ValueError("patch donor hidden size mismatch")
        modified = hidden.clone()
        rows = self.torch.arange(hidden.shape[0], device=hidden.device)
        modified[rows, self.positions.to(hidden.device)] = replacement
        self.fired += 1
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified


def write_m2_patch_scores(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    full_activation_bundle_dir: str | Path,
    run_id: str,
    *,
    patch_kind: str,
    block: int,
    device: str,
    runs_dir: str | Path = "runs",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Score one of two registered patches or the identity negative control."""

    _validate_run_id(run_id)
    if patch_kind not in PATCH_KINDS:
        raise ValueError(f"unsupported M2 patch_kind={patch_kind}")
    block = int(block)
    if block not in M2_PATCH_BLOCKS:
        raise ValueError("M2 patch block must be 13 or 27")
    if not str(device).strip():
        raise ValueError("an explicit patch scoring device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("M2 patch max_wall_seconds must be in (0, 13500]")
    if max_requests is not None and int(max_requests) <= 0:
        raise ValueError("max_requests must be positive")

    probe_manifest = load_m2_probe_manifest()
    implementation_identity = patch_scorer_implementation_identity()
    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    runs_dir = Path(runs_dir)
    run_dir = runs_dir / run_id
    records_path = standardized_dir / "records_dev.jsonl"
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
    frozen = probe_manifest["frozen_inputs"]
    hashes = {
        "records_sha256": sha256_file(records_path),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "candidate_manifest_sha256": sha256_file(candidate_manifest_path),
        "request_manifest_sha256": sha256_file(request_manifest_path),
    }
    expected = {
        "records_sha256": frozen["records_dev_sha256"],
        "dataset_manifest_sha256": frozen["dataset_manifest_sha256"],
        "candidate_manifest_sha256": frozen["candidate_manifest_sha256"],
        "request_manifest_sha256": frozen["request_manifest_sha256"],
    }
    for key, value in expected.items():
        if hashes[key] != value:
            raise ValueError(f"frozen M2 patch input hash mismatch: {key}")
    records = [sanitize_record_for_model(row) for row in iter_jsonl(records_path)]
    if len(records) != 8000:
        raise ValueError("M2 patch scorer requires all 8000 internal-dev requests")

    config = load_v12_ranker_config(config_path)
    method_id = str(config["method_id"])
    frozen_model = frozen["models"].get(method_id)
    if frozen_model is None:
        raise ValueError("M2 patch scorer admits only Q2/Q3")
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("M2 patch config hash differs from frozen manifest")
    checkpoint_dir = checkpoint_root / CHECKPOINT_DIRNAME
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(
        training_metadata, config, allow_smoke=max_requests is not None
    )
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_dir / "model", method_id
    )
    if checkpoint_id != frozen_model["checkpoint_id"]:
        raise ValueError("M2 patch checkpoint differs from frozen manifest")
    if checkpoint_id != training_metadata.get("checkpoint_id"):
        raise ValueError("M2 patch checkpoint changed after training metadata")

    full_bundle = audit_activation_bundle(
        full_activation_bundle_dir,
        expected_records=records,
        expected_role="dev_representation",
        expected_condition="full",
        require_result_eligible=True,
    )
    if full_bundle.metadata.get("method_id") != method_id:
        raise ValueError("M2 patch donor/model method mismatch")
    if full_bundle.metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("M2 patch donor/model checkpoint mismatch")
    donor = FullActivationDonor(full_bundle, records)
    cross_mapping = _cross_request_mapping(records)
    records_by_id = {row.request_id: row for row in records}
    mapping_sha256 = _canonical_sha256(cross_mapping)
    evidence_mode = "smoke_non_result" if max_requests is not None else "mechanism_diagnostic"
    target_records = records[: int(max_requests)] if max_requests is not None else records
    recipient_condition = (
        "full" if patch_kind == "full_to_full_identity" else "null"
    )
    run_contract = {
        "schema_version": 1,
        "run_id": run_id,
        "patch_kind": patch_kind,
        "patch_block_zero_based": block,
        "donor_hidden_state_index": block + 1,
        "recipient_condition": recipient_condition,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "records_sha256": hashes["records_sha256"],
        "candidate_manifest_sha256": hashes["candidate_manifest_sha256"],
        "request_manifest_sha256": hashes["request_manifest_sha256"],
        "full_activation_index_sha256": sha256_file(full_bundle.root / "index.json"),
        "cross_request_mapping_sha256": mapping_sha256,
        "mechanism_probe_manifest_sha256": probe_manifest["sha256"],
        "target_requests": len(target_records),
        "evidence_mode": evidence_mode,
        "device": str(device),
        "implementation_digest": implementation_identity["digest"],
    }
    contract_sha256 = _canonical_sha256(run_contract)
    base_metadata = {
        "schema_version": 1,
        "analysis_stage": "m2_mediation_patch",
        "run_id": run_id,
        "patch_kind": patch_kind,
        "patch_block_zero_based": block,
        "donor_hidden_state_index": block + 1,
        "patch_position": "candidate_readout",
        "patch_direction": (
            "full_readout_position_to_null_same_request_same_candidate"
            if patch_kind == "same_request_full_to_null"
            else patch_kind
        ),
        "recipient_condition": recipient_condition,
        "method_id": method_id,
        "checkpoint_id": checkpoint_id,
        "checkpoint_weight_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "dataset_id": "kuaisearch",
        "dataset_version": "full_confirm_preceding40k_v11",
        "split": "dev",
        **hashes,
        "full_activation_bundle_path": str(full_bundle.root),
        "full_activation_metadata_sha256": sha256_file(
            full_bundle.root / "metadata.json"
        ),
        "full_activation_index_sha256": sha256_file(full_bundle.root / "index.json"),
        "cross_request_mapping_rule": (
            "sha256(namespace, request_id) ring successor; candidate ordinal modulo donor slate"
        ),
        "cross_request_mapping_sha256": mapping_sha256,
        "mechanism_probe_manifest": {
            key: probe_manifest[key]
            for key in ("path", "sha256", "expected_sha256", "verified", "manifest_id")
        },
        "implementation_identity": implementation_identity,
        "donor_activation_pass_contract": full_bundle.metadata[
            "activation_passes"
        ],
        "qrels_read": False,
        "source_test_opened": False,
        "complete_finite_score_coverage_required": True,
        "identity_control_acceptance": {
            "comparison": "patched_identity_scores_vs_frozen_full_scores",
            "max_abs_score_delta_tolerance": 1.0e-5,
            "evaluated_by": "independent_m2_patch_evaluator_before_qrels",
        },
        "evidence_mode": evidence_mode,
        "result_eligible": max_requests is None,
        "run_contract": run_contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "initializing",
    }
    state = _prepare_score_state(
        run_dir,
        base_metadata=base_metadata,
        contract_sha256=contract_sha256,
        records=target_records,
        resume=resume,
    )
    metadata = state["metadata"]
    progress = state["progress"]
    partial_hasher = state["partial_hasher"]
    completed = int(progress["completed_requests"])
    if completed >= len(target_records):
        return _finalize_patch_scores(run_dir, metadata, progress, target_records)

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
        metadata["status"] = "running"
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        score_batch_size = int(config.get("scoring", {}).get("batch_size", 8))
        if score_batch_size <= 0:
            raise ValueError("frozen patch scoring batch size must be positive")
        with torch.inference_mode(), ReadoutActivationPatch(model, block) as patcher:
            for ordinal in range(completed, len(target_records)):
                if time.monotonic() - started >= max_wall_seconds:
                    metadata.update(
                        {
                            "status": "wall_time_exhausted",
                            "resumable": True,
                            "completed_requests": ordinal,
                            "qrels_read": False,
                            "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
                            + (time.monotonic() - started),
                        }
                    )
                    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
                    return metadata
                record = target_records[ordinal]
                recipient_history = list(record.history) if recipient_condition == "full" else []
                rows: list[dict[str, Any]] = []
                for start in range(0, len(record.candidates), score_batch_size):
                    chunk = list(record.candidates[start : start + score_batch_size])
                    vectors: list[np.ndarray] = []
                    for offset, _candidate in enumerate(chunk):
                        candidate_ordinal = start + offset
                        if patch_kind == "cross_request_same_layer":
                            donor_request_id = cross_mapping[record.request_id]
                            donor_record = records_by_id[donor_request_id]
                            donor_ordinal = candidate_ordinal % len(donor_record.candidates)
                        else:
                            donor_request_id = record.request_id
                            donor_ordinal = candidate_ordinal
                        vectors.append(
                            donor.candidate_vector(
                                donor_request_id, donor_ordinal, block + 1
                            )
                        )
                    scores = score_candidates_with_patch(
                        model,
                        tokenizer,
                        patcher,
                        record,
                        chunk,
                        recipient_history,
                        vectors,
                        config,
                        device=str(device),
                    )
                    for candidate, score in zip(chunk, scores):
                        if not math.isfinite(score):
                            raise FloatingPointError("M2 patch score is non-finite")
                        rows.append(
                            {
                                "request_id": record.request_id,
                                "candidate_item_id": str(candidate["item_id"]),
                                "method_id": method_id,
                                "score": float(score),
                            }
                        )
                block_row = {
                    "ordinal": ordinal,
                    "request_id": record.request_id,
                    "patch_kind": patch_kind,
                    "patch_block_zero_based": block,
                    "rows": rows,
                    "rows_sha256": _canonical_sha256(rows),
                }
                line = _canonical_json(block_row) + "\n"
                _append_sync(run_dir / PARTIAL_FILENAME, line)
                partial_hasher.update(line.encode("utf-8"))
                progress.update(
                    {
                        "completed_requests": ordinal + 1,
                        "completed_score_rows": int(progress["completed_score_rows"])
                        + len(rows),
                        "last_request_id": record.request_id,
                        "partial_sha256": partial_hasher.hexdigest(),
                        "status": "running",
                        "updated_at": _utc_now(),
                    }
                )
                _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    except Exception as exc:
        metadata.update(
            {
                "status": (
                    "mechanical_failure"
                    if isinstance(exc, MechanicalPositionError)
                    else "failed"
                ),
                "resumable": not isinstance(exc, MechanicalPositionError),
                "qrels_read": False,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        raise
    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        time.monotonic() - started
    )
    return _finalize_patch_scores(run_dir, metadata, progress, target_records)


def patch_scorer_implementation_identity() -> dict[str, Any]:
    """Hash every project-owned component that defines patch scores."""

    root = Path(__file__).resolve().parents[3]
    paths = (
        root / "src/myrec/mechanism/representation_probe.py",
        root / "src/myrec/mechanism/representation_runtime.py",
        root / "src/myrec/mechanism/patch_scorer.py",
        root / "scripts/score_m2_activation_patch.py",
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


def score_candidate_with_patch(
    model: Any,
    tokenizer: Any,
    patcher: ReadoutActivationPatch,
    record: ModelRecord,
    candidate: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    donor_vector: np.ndarray,
    config: Mapping[str, Any],
    *,
    device: str,
) -> float:
    """Single-row wrapper used for mechanical equivalence tests."""

    return score_candidates_with_patch(
        model,
        tokenizer,
        patcher,
        record,
        [candidate],
        history,
        [donor_vector],
        config,
        device=device,
    )[0]


def score_candidates_with_patch(
    model: Any,
    tokenizer: Any,
    patcher: ReadoutActivationPatch,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    donor_vectors: Sequence[np.ndarray],
    config: Mapping[str, Any],
    *,
    device: str,
) -> list[float]:
    """Vectorized Q2/Q3 scoring with one donor vector/position per batch row.

    Q2 performs one model call.  Q3 performs exactly two calls (Yes and No)
    for the whole chunk, matching the frozen per-token mean log-likelihood.
    """

    import torch
    from torch.nn import functional as F

    if not candidates or len(candidates) != len(donor_vectors):
        raise ValueError("patch candidate/donor batch is empty or misaligned")
    method_id = str(config["method_id"])
    training = config["training"]
    yes_target = (
        _answer_target_tokens(tokenizer, "Yes")
        if method_id == "q3_tallrec_generalqwen"
        else []
    )
    no_target = (
        _answer_target_tokens(tokenizer, "No")
        if method_id == "q3_tallrec_generalqwen"
        else []
    )
    if method_id == "q3_tallrec_generalqwen" and len(yes_target) != len(no_target):
        raise MechanicalPositionError(
            "q3_target_length_mismatch",
            "Q3 shared readout donor requires equal frozen Yes/No target lengths",
        )
    reserve = max(len(yes_target), len(no_target)) if yes_target else 0
    prompts = [
        instrument_pointwise_prompt(
            tokenizer,
            method_id,
            record,
            candidate,
            history=history,
            history_budget=int(training["history_budget"]),
            max_length=int(training["max_length"]) - reserve,
        )
        for candidate in candidates
    ]
    donor_array = np.stack([np.asarray(value, dtype=np.float32) for value in donor_vectors])
    if donor_array.ndim != 2 or not np.isfinite(donor_array).all():
        raise ValueError("patch donor batch is invalid")
    donor = torch.tensor(donor_array, dtype=torch.float32, device=device)
    if method_id == "q2_recranker_generalqwen":
        ids, mask, padding = _left_pad_sequences(
            [prompt.token_ids for prompt in prompts], tokenizer.pad_token_id, device
        )
        positions = torch.tensor(
            [left + prompt.candidate_readout for left, prompt in zip(padding, prompts)],
            dtype=torch.long,
            device=device,
        )
        patcher.arm(positions, donor)
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=1,
        )
        patcher.disarm()
        logits = output.logits[:, -1]
        yes_id = _single_token_id(tokenizer, "yes")
        no_id = _single_token_id(tokenizer, "no")
        return [
            float(value)
            for value in (logits[:, yes_id] - logits[:, no_id]).float().cpu().tolist()
        ]
    values: list[np.ndarray] = []
    for target in (yes_target, no_target):
        sequences = [list(prompt.token_ids) + list(target) for prompt in prompts]
        ids, mask, padding = _left_pad_sequences(
            sequences, tokenizer.pad_token_id, device
        )
        positions = torch.tensor(
            [left + prompt.candidate_readout for left, prompt in zip(padding, prompts)],
            dtype=torch.long,
            device=device,
        )
        patcher.arm(positions, donor)
        output = model(
            input_ids=ids,
            attention_mask=mask,
            use_cache=False,
            logits_to_keep=len(target) + 1,
        )
        patcher.disarm()
        logits = output.logits[:, -(len(target) + 1) : -1].float()
        target_tensor = torch.tensor(target, dtype=torch.long, device=device)
        target_tensor = target_tensor[None, :, None].expand(len(prompts), -1, -1)
        log_likelihood = F.log_softmax(logits, dim=-1).gather(
            2, target_tensor
        ).squeeze(2).mean(dim=1)
        values.append(log_likelihood.cpu().numpy())
    return [float(value) for value in (values[0] - values[1]).tolist()]


def _left_pad_sequences(
    sequences: Sequence[Sequence[int]], pad_token_id: int, device: str
) -> tuple[Any, Any, list[int]]:
    import torch

    if not sequences or any(not value for value in sequences):
        raise ValueError("cannot left-pad an empty sequence batch")
    maximum = max(len(value) for value in sequences)
    ids = torch.full(
        (len(sequences), maximum),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    mask = torch.zeros_like(ids)
    padding: list[int] = []
    for row, values in enumerate(sequences):
        left = maximum - len(values)
        padding.append(left)
        ids[row, left:] = torch.tensor(values, dtype=torch.long, device=device)
        mask[row, left:] = 1
    return ids, mask, padding


def _cross_request_mapping(records: Sequence[ModelRecord]) -> dict[str, str]:
    if len(records) < 2:
        raise ValueError("cross-request patch requires at least two requests")
    ordered = sorted(
        (row.request_id for row in records),
        key=lambda value: (
            hashlib.sha256(f"m2_cross_patch\x1f{value}".encode("utf-8")).hexdigest(),
            value,
        ),
    )
    result = {
        request_id: ordered[(index + 1) % len(ordered)]
        for index, request_id in enumerate(ordered)
    }
    if any(key == value for key, value in result.items()):
        raise AssertionError("cross-request donor mapping contains an identity edge")
    return result


def _prepare_score_state(
    run_dir: Path,
    *,
    base_metadata: dict[str, Any],
    contract_sha256: str,
    records: Sequence[ModelRecord],
    resume: bool,
) -> dict[str, Any]:
    partial_path = run_dir / PARTIAL_FILENAME
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"M2 patch run directory is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial_path.touch(exist_ok=False)
        metadata = dict(base_metadata)
        metadata.update(
            {"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []}
        )
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
        _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
        _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
        return {
            "metadata": metadata,
            "progress": progress,
            "partial_hasher": hashlib.sha256(),
        }
    metadata = _read_json(run_dir / METADATA_FILENAME)
    progress = _read_json(run_dir / PROGRESS_FILENAME)
    if metadata.get("run_contract_sha256") != contract_sha256:
        raise ValueError("M2 patch resume metadata contract drift")
    if progress.get("run_contract_sha256") != contract_sha256:
        raise ValueError("M2 patch resume progress contract drift")
    if metadata.get("status") not in {"initializing", "running", "failed", "wall_time_exhausted"}:
        raise ValueError("M2 patch run is not resumable")
    observed = _audit_partial_scores(partial_path, records, metadata)
    for key in (
        "completed_requests",
        "completed_score_rows",
        "last_request_id",
        "partial_sha256",
    ):
        if progress.get(key) != observed[key]:
            raise ValueError(f"M2 patch resume progress mismatch: {key}")
    lineage = metadata.get("resume_lineage", [])
    if not isinstance(lineage, list):
        raise ValueError("M2 patch resume lineage is invalid")
    lineage.append(
        {
            "resumed_at": _utc_now(),
            "from_status": metadata.get("status"),
            "completed_requests": observed["completed_requests"],
            "completed_score_rows": observed["completed_score_rows"],
            "prior_error": metadata.get("error"),
            "partial_sha256": observed["partial_sha256"],
        }
    )
    metadata.update(
        {"status": "initializing", "error": None, "resume_lineage": lineage}
    )
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    return {
        "metadata": metadata,
        "progress": progress,
        "partial_hasher": hashlib.sha256((run_dir / PARTIAL_FILENAME).read_bytes()),
    }


def _audit_partial_scores(
    path: Path, records: Sequence[ModelRecord], metadata: Mapping[str, Any]
) -> dict[str, Any]:
    content = path.read_bytes()
    if content and not content.endswith(b"\n"):
        raise ValueError("M2 patch partial score file has an incomplete line")
    rows_count = 0
    last_request_id = None
    lines = content.splitlines()
    if len(lines) > len(records):
        raise ValueError("M2 patch partial score exceeds request population")
    for ordinal, encoded in enumerate(lines):
        row = json.loads(encoded)
        if (_canonical_json(row) + "\n").encode("utf-8") != encoded + b"\n":
            raise ValueError("M2 patch partial score block is not canonical")
        record = records[ordinal]
        if row.get("ordinal") != ordinal or row.get("request_id") != record.request_id:
            raise ValueError("M2 patch partial request identity/order mismatch")
        if row.get("patch_kind") != metadata.get("patch_kind"):
            raise ValueError("M2 patch partial patch_kind mismatch")
        if row.get("patch_block_zero_based") != metadata.get("patch_block_zero_based"):
            raise ValueError("M2 patch partial block mismatch")
        score_rows = row.get("rows")
        if row.get("rows_sha256") != _canonical_sha256(score_rows):
            raise ValueError("M2 patch partial score block hash mismatch")
        expected_ids = [str(value["item_id"]) for value in record.candidates]
        if [value.get("candidate_item_id") for value in score_rows] != expected_ids:
            raise ValueError("M2 patch partial candidate identity/order mismatch")
        for value in score_rows:
            if value.get("request_id") != record.request_id:
                raise ValueError("M2 patch partial request identity mismatch")
            if value.get("method_id") != metadata.get("method_id"):
                raise ValueError("M2 patch partial method identity mismatch")
            if not math.isfinite(float(value.get("score"))):
                raise ValueError("M2 patch partial score is non-finite")
        rows_count += len(score_rows)
        last_request_id = record.request_id
    return {
        "completed_requests": len(lines),
        "completed_score_rows": rows_count,
        "last_request_id": last_request_id,
        "partial_sha256": hashlib.sha256(content).hexdigest(),
    }


def _finalize_patch_scores(
    run_dir: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    records: Sequence[ModelRecord],
) -> dict[str, Any]:
    observed = _audit_partial_scores(run_dir / PARTIAL_FILENAME, records, metadata)
    if observed["completed_requests"] != len(records):
        raise ValueError("cannot finalize incomplete M2 patch score coverage")
    expected_rows = sum(len(row.candidates) for row in records)
    if observed["completed_score_rows"] != expected_rows:
        raise ValueError("cannot finalize incomplete M2 patch candidate coverage")
    scores_path = run_dir / SCORES_FILENAME
    temporary = run_dir / ".scores.writing.jsonl"
    with (run_dir / PARTIAL_FILENAME).open("r", encoding="utf-8") as source, temporary.open(
        "w", encoding="utf-8"
    ) as target:
        for line in source:
            block = json.loads(line)
            for row in block["rows"]:
                target.write(_canonical_json(row) + "\n")
    temporary.replace(scores_path)
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "qrels_read": False,
            "request_count": len(records),
            "score_rows": expected_rows,
            "scores_path": str(scores_path),
            "scores_sha256": sha256_file(scores_path),
            "complete_finite_score_coverage": True,
            "completed_at": _utc_now(),
        }
    )
    _write_json_atomic(run_dir / METADATA_FILENAME, metadata)
    progress.update(
        {
            **observed,
            "scores_sha256": metadata["scores_sha256"],
            "status": "completed",
            "updated_at": _utc_now(),
        }
    )
    _write_json_atomic(run_dir / PROGRESS_FILENAME, progress)
    return metadata


def _append_sync(path: Path, line: str) -> None:
    with path.open("ab") as handle:
        handle.write(line.encode("utf-8"))
        handle.flush()
        import os

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


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
