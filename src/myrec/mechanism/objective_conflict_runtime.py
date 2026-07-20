"""Exact train-only Q2 RankNet/ListNet full-model gradient conflict probe."""

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

from myrec.baselines.motivation_v12_contracts import load_training_groups
from myrec.baselines.motivation_v12_ranker import (
    CHECKPOINT_DIRNAME,
    TRAINING_METADATA,
    _assert_frozen_training_population,
    _checkpoint_identity,
    _git_revision,
    _runtime_metadata,
    _validate_run_id,
    _validate_scoring_checkpoint_provenance,
    _yes_no_group_scores,
    listwise_softmax_loss,
    load_v12_ranker_config,
    pairwise_ranknet_loss,
)
from myrec.mechanism.attention_edge_runtime import _load_manifest
from myrec.mechanism.gradient_diagnostic import (
    CONTROLS,
    MODEL_INITIALIZATION_SEED,
    REQUESTS_PER_SURFACE,
    SELECTION_SEED,
    SURFACES,
    _load_state_model,
    _load_train_gains,
    _seed_everything,
    deterministic_label_shuffle,
    select_surface_training_groups,
)
from myrec.mechanism.optimizer_replay_math import (
    gradient_pair_summary,
    parameter_order_digest,
    parameter_order_rows,
    q2_parameter_family,
)
from myrec.mechanism.representation_probe import normalize_query
from myrec.utils.hashing import sha256_file, sha256_text


Q2_METHOD_ID = "q2_recranker_generalqwen"
SUPPORTED_STATES = ("base_initialization", "frozen_final_checkpoint")
EXPECTED_PARAMETER_ORDER_DIGEST = (
    "f6f51ede754a5b360faba0b3acde525a3a03ba4e512d8e618ef321646dc9cf70"
)
MAX_WALL_SECONDS = 13_500.0


def run_q2_objective_conflict(
    standardized_dir: str | Path,
    config_path: str | Path,
    checkpoint_root: str | Path,
    state: str,
    run_id: str,
    *,
    device: str,
    runs_dir: str | Path = "runs",
    manifest_path: str | Path = "experiments/motivation/transformer_deep_dive_manifest.yaml",
    resume: bool = False,
    max_wall_seconds: float = MAX_WALL_SECONDS,
    max_requests_per_surface: int | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Compute per-request full-parameter objective-gradient cosines."""

    _validate_run_id(run_id)
    if state not in SUPPORTED_STATES:
        raise ValueError("Q2 objective conflict state is invalid")
    if not str(device).strip():
        raise ValueError("an explicit objective-conflict device is required")
    max_wall_seconds = float(max_wall_seconds)
    if not math.isfinite(max_wall_seconds) or not 0 < max_wall_seconds <= MAX_WALL_SECONDS:
        raise ValueError("objective-conflict max_wall_seconds must be in (0, 13500]")
    if max_requests_per_surface is not None and not 0 < int(
        max_requests_per_surface
    ) < REQUESTS_PER_SURFACE:
        raise ValueError("smoke request cap must be in [1,95]")
    target_per_surface = int(max_requests_per_surface or REQUESTS_PER_SURFACE)

    standardized_dir = Path(standardized_dir)
    config_path = Path(config_path)
    checkpoint_root = Path(checkpoint_root)
    run_dir = Path(runs_dir) / run_id
    manifest = _load_manifest(manifest_path)
    config = load_v12_ranker_config(config_path)
    if config["method_id"] != Q2_METHOD_ID:
        raise ValueError("objective conflict probe is Q2-only")
    frozen_model = manifest["frozen_inputs"]["models"][Q2_METHOD_ID]
    if config["_config_sha256"] != frozen_model["config_sha256"]:
        raise ValueError("objective conflict config differs from frozen manifest")
    _assert_frozen_training_population(standardized_dir, config)
    records_path = standardized_dir / "records_train.jsonl"
    qrels_path = standardized_dir / "qrels_train.jsonl"
    paths = {
        "records_train_sha256": records_path,
        "qrels_train_sha256": qrels_path,
        "dataset_manifest_sha256": standardized_dir / "manifest.json",
        "request_manifest_sha256": standardized_dir / "request_manifest.json",
        "candidate_manifest_sha256": standardized_dir / "candidate_manifest.json",
    }
    hashes = {key: sha256_file(path) for key, path in paths.items()}
    population = config["_protocol"]["data"]["development_population"]
    expected_hashes = {
        "records_train_sha256": population["records_train_sha256"],
        "qrels_train_sha256": population["qrels_train_sha256"],
        "dataset_manifest_sha256": population["manifest_sha256"],
        "request_manifest_sha256": population["request_manifest_sha256"],
        "candidate_manifest_sha256": population["candidate_manifest_sha256"],
    }
    if hashes != expected_hashes:
        raise ValueError("objective conflict frozen train hashes differ")
    training_metadata_path = checkpoint_root / TRAINING_METADATA
    training_metadata = _read_json(training_metadata_path)
    _validate_scoring_checkpoint_provenance(training_metadata, config, allow_smoke=False)
    checkpoint_model_dir = checkpoint_root / CHECKPOINT_DIRNAME / "model"
    checkpoint_id, checkpoint_files = _checkpoint_identity(
        checkpoint_model_dir, Q2_METHOD_ID
    )
    if checkpoint_id != frozen_model["checkpoint_id"] or training_metadata.get(
        "checkpoint_id"
    ) != checkpoint_id:
        raise ValueError("objective conflict checkpoint binding differs")

    gains = _load_train_gains(qrels_path)
    groups, group_stats = load_training_groups(
        records_path,
        qrels_path,
        seed=int(config["training"]["seed"]),
        negatives_per_positive=int(config["training"]["negatives_per_positive"]),
        max_group_size=int(config["training"].get("list_size", 8)),
    )
    selected, selection = select_surface_training_groups(
        groups,
        gains,
        requests_per_surface=target_per_surface,
        selection_seed=SELECTION_SEED,
    )
    tasks = []
    for control in CONTROLS:
        for surface in SURFACES:
            for ordinal, group in enumerate(selected[surface]):
                tasks.append((control, surface, ordinal, group))
    selection.update(
        {
            "method_id": Q2_METHOD_ID,
            "state": state,
            "group_construction": group_stats,
            "frozen_hashes": hashes,
            "deep_dive_manifest_sha256": manifest["_sha256"],
            "finalized_before_model_load_and_gradient": True,
        }
    )
    selection_sha256 = _canonical_sha256(selection)
    implementation = objective_conflict_implementation_identity()
    evidence_mode = (
        "registered_mechanism_diagnostic"
        if max_requests_per_surface is None
        else "mechanical_smoke_non_result"
    )
    contract = {
        "schema_version": 1,
        "run_id": run_id,
        "method_id": Q2_METHOD_ID,
        "state": state,
        "checkpoint_id": checkpoint_id,
        "config_sha256": config["_config_sha256"],
        "selection_sha256": selection_sha256,
        "tasks": len(tasks),
        "requests_per_surface": target_per_surface,
        "parameter_order_digest": EXPECTED_PARAMETER_ORDER_DIGEST,
        "deep_dive_manifest_sha256": manifest["_sha256"],
        "implementation_digest": implementation["digest"],
        "device": str(device),
        "evidence_mode": evidence_mode,
    }
    contract_sha256 = _canonical_sha256(contract)
    metadata = {
        "schema_version": 1,
        "analysis_stage": "transformer_deep_dive_d7_q2_objective_conflict",
        "run_id": run_id,
        "method_id": Q2_METHOD_ID,
        "state": state,
        "checkpoint_id": checkpoint_id,
        "checkpoint_files": checkpoint_files,
        "config_path": str(config_path),
        "config_sha256": config["_config_sha256"],
        "training_metadata_sha256": sha256_file(training_metadata_path),
        "frozen_hashes": hashes,
        "selection_sha256": selection_sha256,
        "requests_per_surface": target_per_surface,
        "controls": list(CONTROLS),
        "surfaces": list(SURFACES),
        "objectives": ["pairwise_ranknet", "listwise_softmax"],
        "parameter_boundary": "all_unique_q2_trainable_parameter_objects",
        "expected_parameter_order_digest": EXPECTED_PARAMETER_ORDER_DIGEST,
        "qrels_access": "train_only_before_model_load_for_frozen_selection",
        "dev_confirmation_test_qrels_read": False,
        "source_test_opened": False,
        "implementation_identity": implementation,
        "evidence_mode": evidence_mode,
        "result_eligible": max_requests_per_surface is None,
        "run_contract": contract,
        "run_contract_sha256": contract_sha256,
        "command": list(command or sys.argv),
        "code_revision": _git_revision(),
        "status": "selection_finalized",
    }
    metadata, progress = _prepare(
        run_dir,
        metadata,
        selection,
        contract_sha256,
        tasks,
        resume=resume,
    )
    completed = int(progress["completed_tasks"])
    if completed >= len(tasks):
        return _finalize(run_dir, metadata, progress, tasks)

    started = time.monotonic()
    try:
        import torch
        import transformers

        tokenizer, model = _load_state_model(
            config,
            state=state,
            device=str(device),
            checkpoint_model_dir=checkpoint_model_dir,
            torch_module=torch,
        )
        model.train(True)
        named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and name != "lm_head.weight"
        ]
        observed_digest = parameter_order_digest(named_parameters)
        if observed_digest != EXPECTED_PARAMETER_ORDER_DIGEST or len(
            named_parameters
        ) != 310:
            raise ValueError("Q2 full parameter order differs from frozen step-500 binding")
        names = [name for name, _ in named_parameters]
        parameters = [parameter for _, parameter in named_parameters]
        families = {name: q2_parameter_family(name) for name in names}
        metadata.update(
            {
                **_runtime_metadata(Q2_METHOD_ID, torch, transformers),
                "parameter_order_digest": observed_digest,
                "parameter_order_rows": parameter_order_rows(named_parameters),
                "status": "running",
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        dtype = str(config["training"].get("dtype", "bfloat16"))
        autocast_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        for task_index in range(completed, len(tasks)):
            if time.monotonic() - started >= max_wall_seconds:
                metadata.update(
                    {
                        "status": "wall_time_exhausted",
                        "resumable": True,
                        "elapsed_seconds": float(metadata.get("elapsed_seconds", 0.0))
                        + time.monotonic()
                        - started,
                    }
                )
                _write_json(run_dir / "metadata.json", metadata)
                return metadata
            control, surface, ordinal, original_group = tasks[task_index]
            group = original_group
            shuffle_audit = None
            if control == "within_request_label_shuffle":
                group, shuffle_audit = deterministic_label_shuffle(group)
            seed = _task_seed(state, control, surface, group.record.request_id)
            _seed_everything(torch, seed)
            model.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=autocast_dtype,
                enabled=str(device).startswith("cuda") and dtype != "float32",
            ):
                scores = _yes_no_group_scores(
                    model, tokenizer, [group], config, device=str(device)
                )[0]
                pair_loss = pairwise_ranknet_loss(scores, group.gains)
                list_loss = listwise_softmax_loss(scores, group.gains)
            pair_gradients = torch.autograd.grad(
                pair_loss,
                parameters,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
            )
            list_gradients = torch.autograd.grad(
                list_loss,
                parameters,
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
            summary = gradient_pair_summary(
                names,
                pair_gradients,
                list_gradients,
                family_by_name=families,
            )
            row = {
                "task_index": task_index,
                "control": control,
                "surface": surface,
                "ordinal_within_cell": ordinal,
                "request_id": group.record.request_id,
                "normalized_query": normalize_query(group.record.query),
                "gradient_rng_seed": seed,
                "label_shuffle": shuffle_audit,
                "pairwise_ranknet_loss": float(pair_loss.detach().float().cpu()),
                "listwise_softmax_loss": float(list_loss.detach().float().cpu()),
                **summary,
            }
            _append_sync(run_dir / "per_request.partial.jsonl", row)
            progress.update(
                {
                    "completed_tasks": task_index + 1,
                    "last_request_id": group.record.request_id,
                    "partial_sha256": sha256_file(
                        run_dir / "per_request.partial.jsonl"
                    ),
                    "status": "running",
                    "updated_at": _utc_now(),
                }
            )
            _write_json(run_dir / "progress.json", progress)
            del pair_gradients, list_gradients, scores, pair_loss, list_loss
            model.zero_grad(set_to_none=True)
    except Exception as exc:
        metadata.update(
            {
                "status": "mechanical_failure",
                "resumable": True,
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }
        )
        _write_json(run_dir / "metadata.json", metadata)
        raise
    metadata["elapsed_seconds"] = float(metadata.get("elapsed_seconds", 0.0)) + (
        time.monotonic() - started
    )
    return _finalize(run_dir, metadata, progress, tasks)


def objective_conflict_implementation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/objective_conflict_runtime.py",
        "src/myrec/mechanism/optimizer_replay_math.py",
        "src/myrec/mechanism/gradient_diagnostic.py",
        "scripts/run_deep_dive_objective_conflict.py",
    )
    files = [
        {
            "path": relative,
            "sha256": sha256_file(root / relative),
            "size_bytes": (root / relative).stat().st_size,
        }
        for relative in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _prepare(
    run_dir: Path,
    metadata: dict[str, Any],
    selection: dict[str, Any],
    contract_sha256: str,
    tasks: Sequence[Any],
    *,
    resume: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    partial_path = run_dir / "per_request.partial.jsonl"
    if not resume:
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(f"objective conflict run is not empty: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=True)
        partial_path.touch(exist_ok=False)
        _write_json(run_dir / "selection_manifest.json", selection)
        metadata.update({"elapsed_seconds": 0.0, "resumable": True, "resume_lineage": []})
        progress = {
            "schema_version": 1,
            "run_contract_sha256": contract_sha256,
            "completed_tasks": 0,
            "last_request_id": None,
            "partial_sha256": sha256_file(partial_path),
            "status": "selection_finalized",
            "updated_at": _utc_now(),
        }
        _write_json(run_dir / "metadata.json", metadata)
        _write_json(run_dir / "progress.json", progress)
        return metadata, progress
    stored = _read_json(run_dir / "metadata.json")
    progress = _read_json(run_dir / "progress.json")
    if stored.get("run_contract_sha256") != contract_sha256 or progress.get(
        "run_contract_sha256"
    ) != contract_sha256:
        raise ValueError("objective conflict resume contract drift")
    if _read_json(run_dir / "selection_manifest.json") != selection:
        raise ValueError("objective conflict resume selection drift")
    observed = _audit_partial(partial_path, tasks)
    if progress.get("completed_tasks") != observed["completed_tasks"] or progress.get(
        "partial_sha256"
    ) != observed["partial_sha256"]:
        raise ValueError("objective conflict progress differs from partial")
    lineage = list(stored.get("resume_lineage", []))
    lineage.append(
        {
            "resumed_at": _utc_now(),
            "completed_tasks": observed["completed_tasks"],
            "partial_sha256": observed["partial_sha256"],
        }
    )
    stored.update({"resume_lineage": lineage, "status": "selection_finalized"})
    _write_json(run_dir / "metadata.json", stored)
    return stored, progress


def _audit_partial(path: Path, tasks: Sequence[Any]) -> dict[str, Any]:
    count = 0
    for row in _iter_jsonl(path):
        if count >= len(tasks):
            raise ValueError("objective conflict partial has excess rows")
        control, surface, ordinal, group = tasks[count]
        if (
            row.get("task_index") != count
            or row.get("control") != control
            or row.get("surface") != surface
            or row.get("ordinal_within_cell") != ordinal
            or row.get("request_id") != group.record.request_id
        ):
            raise ValueError("objective conflict partial task identity drift")
        for key in ("cosine", "left_norm", "right_norm"):
            if not math.isfinite(float(row[key])):
                raise FloatingPointError("objective conflict partial is non-finite")
        count += 1
    return {"completed_tasks": count, "partial_sha256": sha256_file(path)}


def _finalize(
    run_dir: Path,
    metadata: dict[str, Any],
    progress: dict[str, Any],
    tasks: Sequence[Any],
) -> dict[str, Any]:
    observed = _audit_partial(run_dir / "per_request.partial.jsonl", tasks)
    if observed["completed_tasks"] != len(tasks):
        raise ValueError("cannot finalize incomplete objective conflict run")
    path = run_dir / "per_request.jsonl"
    os.replace(run_dir / "per_request.partial.jsonl", path)
    progress.update({"status": "completed", "updated_at": _utc_now()})
    metadata.update(
        {
            "status": "completed",
            "resumable": False,
            "completed_request_diagnostics": len(tasks),
            "per_request_path": str(path),
            "per_request_sha256": sha256_file(path),
        }
    )
    _write_json(run_dir / "progress.json", progress)
    _write_json(run_dir / "metadata.json", metadata)
    return metadata


def _task_seed(state: str, control: str, surface: str, request_id: str) -> int:
    payload = "|".join(
        map(
            str,
            (
                MODEL_INITIALIZATION_SEED,
                "d7-objective-conflict",
                state,
                control,
                surface,
                request_id,
            ),
        )
    )
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2**31)


def _append_sync(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _canonical_sha256(value: Any) -> str:
    return sha256_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
