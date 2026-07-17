"""Freeze Motivation V1.2 checkpoint selections before holdout creation.

This builder is deliberately label-opaque.  It byte-hashes the two permitted
development qrels files because they are frozen training/internal-dev inputs,
but it never deserializes them.  It also resolves only KuaiSearch source-train
files and never resolves or opens a source-test path.
"""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from myrec.data.history_assignments import (
    current_motivation_v12_assignment_implementation_identity,
    motivation_v12_assignment_implementation_paths,
    motivation_v12_assignment_recipe,
)
from myrec.data.kuaisearch_holdout import (
    V12_METHOD_IDS,
    V12_PILOT_SEED,
    _analysis_selection_implementation_paths,
    _current_analysis_selection_implementation_identity,
    _load_post_selection_lock,
)
from myrec.data.kuaisearch_scout import _resolve_source_path
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import write_json


Q_METHOD_IDS = frozenset(V12_METHOD_IDS - {"w0_copps_style_transfer_witness"})
W0_METHOD_ID = "w0_copps_style_transfer_witness"
LOCK_FILENAME = "post_selection_release_lock.json"
IDENTITY_SUFFIX = ".checkpoint_selection_identity.json"
Q_CHECKPOINT_MODEL_DIR = Path("checkpoint_latest") / "model"
Q_TRAINING_METADATA = "training_metadata.json"
W0_TRAINING_METADATA = "metadata.json"
_Q_INFERENCE_SUFFIXES = {".jinja", ".json", ".model", ".safetensors", ".txt"}

_DEVELOPMENT_INPUT_FILES = {
    "manifest": "manifest.json",
    "candidate_manifest": "candidate_manifest.json",
    "request_manifest": "request_manifest.json",
    "records_train": "records_train.jsonl",
    "records_dev": "records_dev.jsonl",
    "records_confirmation": "records_confirmation.jsonl",
    "qrels_train": "qrels_train.jsonl",
    "qrels_dev": "qrels_dev.jsonl",
}
_SCOUT_INPUT_FILES = {
    "manifest": "manifest.json",
    "candidate_manifest": "candidate_manifest.json",
    "request_manifest": "request_manifest.json",
    "records_train": "records_train.jsonl",
    "records_dev": "records_dev.jsonl",
}
_PROTOCOL_DEVELOPMENT_SHA_FIELDS = {
    "manifest": "manifest_sha256",
    "candidate_manifest": "candidate_manifest_sha256",
    "request_manifest": "request_manifest_sha256",
    "records_train": "records_train_sha256",
    "records_dev": "records_dev_sha256",
    "records_confirmation": "records_legacy_compatibility_sha256",
    "qrels_train": "qrels_train_sha256",
    "qrels_dev": "qrels_dev_sha256",
}


def build_motivation_v12_release_lock(
    *,
    protocol_path: str | Path,
    config_paths: Mapping[str, str | Path],
    q_checkpoint_roots: Mapping[str, str | Path],
    w0_checkpoint_dir: str | Path,
    raw_dir: str | Path,
    development_dir: str | Path,
    subsequent_scout_dir: str | Path,
    output_lock_dir: str | Path,
    lock_id: str = "motivation_v1_2_first_round_post_selection",
    command_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create five checkpoint identities and one atomic final release lock.

    The output directory must not already exist.  All files are assembled and
    schema-validated in a sibling staging directory, every external dependency
    is rehashed, and then the complete directory is published by one rename.
    """

    if not isinstance(lock_id, str) or not lock_id.strip():
        raise ValueError("lock_id must be non-empty")
    protocol_path = _existing_file(protocol_path, "protocol").resolve()
    raw_dir = Path(raw_dir).resolve()
    development_dir = Path(development_dir).resolve()
    subsequent_scout_dir = Path(subsequent_scout_dir).resolve()
    output_lock_dir = Path(output_lock_dir).resolve()
    if output_lock_dir.exists():
        raise FileExistsError(
            f"release-lock output directory already exists: {output_lock_dir}"
        )
    output_lock_dir.parent.mkdir(parents=True, exist_ok=True)

    config_paths = _normalize_exact_path_map(
        config_paths, V12_METHOD_IDS, "config_paths"
    )
    q_checkpoint_roots = _normalize_exact_path_map(
        q_checkpoint_roots, Q_METHOD_IDS, "q_checkpoint_roots", require_file=False
    )
    w0_checkpoint_dir = _existing_directory(
        w0_checkpoint_dir, "W0 checkpoint directory"
    ).resolve()

    protocol, protocol_sha256 = _load_protocol(protocol_path)
    selection_rule = _selection_rule(protocol)
    seed = _pilot_seed(protocol)
    configs, config_shas = _load_and_validate_configs(
        config_paths,
        protocol_path=protocol_path,
        protocol_sha256=protocol_sha256,
        seed=seed,
    )
    q_implementation, q_implementation_paths = _current_implementation_identity(
        "q"
    )
    w0_implementation, w0_implementation_paths = _current_implementation_identity(
        "w0"
    )
    analysis_selection_implementation = (
        _current_analysis_selection_implementation_identity()
    )
    analysis_selection_paths = _analysis_selection_implementation_paths()
    history_assignment_release = {
        "schema_version": 1,
        "recipe": motivation_v12_assignment_recipe(),
        "implementation": (
            current_motivation_v12_assignment_implementation_identity()
        ),
    }
    history_assignment_paths = motivation_v12_assignment_implementation_paths()
    input_paths = release_input_paths(
        raw_dir=raw_dir,
        development_dir=development_dir,
        subsequent_scout_dir=subsequent_scout_dir,
    )

    external_paths = {
        "protocol": protocol_path,
        **{f"config:{method_id}": path for method_id, path in config_paths.items()},
        **{f"input:{name}": path for name, path in input_paths.items()},
        **{
            f"implementation:q:{path.name}": path
            for path in q_implementation_paths
        },
        **{
            f"implementation:w0:{path.name}": path
            for path in w0_implementation_paths
        },
        **{
            f"analysis_selection:{relative_path}": path
            for relative_path, path in analysis_selection_paths.items()
        },
        **{
            f"history_assignment:{relative_path}": path
            for relative_path, path in history_assignment_paths.items()
        },
    }
    initial_snapshots = _snapshot_files(external_paths)
    if initial_snapshots["protocol"]["sha256"] != protocol_sha256:
        raise ValueError("protocol changed while opening release-lock inputs")
    for method_id, config_sha256 in config_shas.items():
        if initial_snapshots[f"config:{method_id}"]["sha256"] != config_sha256:
            raise ValueError(f"config changed while opening release lock: {method_id}")
    for role, implementation in (
        ("q", q_implementation),
        ("w0", w0_implementation),
    ):
        for source in implementation["files"]:
            snapshot = initial_snapshots[f"implementation:{role}:{Path(source['path']).name}"]
            if snapshot["sha256"] != source["sha256"]:
                raise ValueError(
                    f"{role} implementation changed while opening release lock"
                )
    for group_name in ("holdout_selection", "evaluator"):
        for source in analysis_selection_implementation[group_name]["files"]:
            snapshot = initial_snapshots[
                f"analysis_selection:{source['path']}"
            ]
            if snapshot["sha256"] != source["sha256"]:
                raise ValueError(
                    "analysis/selection implementation changed while opening "
                    f"release lock: {source['path']}"
                )
    for source in history_assignment_release["implementation"]["files"]:
        snapshot = initial_snapshots[f"history_assignment:{source['path']}"]
        if snapshot["sha256"] != source["sha256"]:
            raise ValueError(
                "history assignment implementation changed while opening "
                f"release lock: {source['path']}"
            )
    _assert_protocol_development_hashes(
        protocol,
        input_snapshots={
            name: initial_snapshots[f"input:{name}"] for name in input_paths
        },
    )

    identities: dict[str, dict[str, Any]] = {}
    checkpoint_external_paths: dict[str, Path] = {}
    for method_id in sorted(Q_METHOD_IDS):
        identity, paths = _build_q_identity(
            method_id=method_id,
            checkpoint_root=q_checkpoint_roots[method_id],
            config=configs[method_id],
            config_sha256=config_shas[method_id],
            protocol_sha256=protocol_sha256,
            seed=seed,
            selection_rule=selection_rule,
            expected_implementation_digest=q_implementation["digest"],
        )
        identities[method_id] = identity
        checkpoint_external_paths.update(
            {f"checkpoint:{method_id}:{name}": path for name, path in paths.items()}
        )
    w0_identity, w0_paths = _build_w0_identity(
        checkpoint_dir=w0_checkpoint_dir,
        config=configs[W0_METHOD_ID],
        config_sha256=config_shas[W0_METHOD_ID],
        protocol_sha256=protocol_sha256,
        seed=seed,
        selection_rule=selection_rule,
        expected_implementation_digest=w0_implementation["digest"],
    )
    identities[W0_METHOD_ID] = w0_identity
    checkpoint_external_paths.update(
        {f"checkpoint:{W0_METHOD_ID}:{name}": path for name, path in w0_paths.items()}
    )
    q_implementation_digests = {
        identities[method_id]["implementation_digest"] for method_id in Q_METHOD_IDS
    }
    if len(q_implementation_digests) != 1:
        raise ValueError(
            "Q0-Q3 completed checkpoints do not share one ranking-harness "
            "implementation digest"
        )
    initial_snapshots.update(_snapshot_files(checkpoint_external_paths))

    with tempfile.TemporaryDirectory(
        prefix=f".{output_lock_dir.name}.release-", dir=output_lock_dir.parent
    ) as temporary:
        temporary_root = Path(temporary)
        staging_dir = temporary_root / "assembled_release_lock"
        staging_dir.mkdir()
        identity_artifacts: dict[str, dict[str, str]] = {}
        for method_id, identity in sorted(identities.items()):
            identity_path = staging_dir / f"{method_id}{IDENTITY_SUFFIX}"
            write_json(identity_path, identity)
            identity_artifacts[method_id] = {
                "artifact_type": "checkpoint_selection_identity_manifest",
                "path": str(
                    output_lock_dir / f"{method_id}{IDENTITY_SUFFIX}"
                ),
                "sha256": sha256_file(identity_path),
            }

        payload = {
            "schema_version": 1,
            "lock_id": lock_id,
            "status": "post_selection_frozen",
            "protocol_sha256": protocol_sha256,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": list(command_argv) if command_argv is not None else None,
            "holdout_materialization": {
                "authorized": True,
                "methods_frozen": True,
                "witness_frozen": True,
                "configs_frozen": True,
                "checkpoint_selection_frozen": True,
                "analysis_and_selection_rules_frozen": True,
                "history_assignment_recipe_and_generator_frozen": True,
            },
            "analysis_selection_implementation": (
                analysis_selection_implementation
            ),
            "history_assignment_release": history_assignment_release,
            "frozen_configs": {
                method_id: {
                    "path": str(config_paths[method_id]),
                    "sha256": config_shas[method_id],
                }
                for method_id in sorted(V12_METHOD_IDS)
            },
            "frozen_checkpoints": identity_artifacts,
            "input_sha256": {
                name: initial_snapshots[f"input:{name}"]["sha256"]
                for name in sorted(input_paths)
            },
        }
        lock_path = staging_dir / LOCK_FILENAME
        write_json(lock_path, payload)

        # Exercise the materializer's exact schema before publication.  Only
        # the identity paths differ in this temporary validation copy.
        validation_payload = json.loads(json.dumps(payload))
        for method_id in V12_METHOD_IDS:
            validation_payload["frozen_checkpoints"][method_id]["path"] = str(
                staging_dir / f"{method_id}{IDENTITY_SUFFIX}"
            )
        validation_lock_path = temporary_root / "schema_validation_lock.json"
        write_json(validation_lock_path, validation_payload)
        _load_post_selection_lock(
            validation_lock_path,
            protocol_sha256=protocol_sha256,
            checkpoint_selection_rule=selection_rule,
            pilot_seed=seed,
        )

        _assert_snapshots_unchanged(initial_snapshots)
        for method_id, artifact in identity_artifacts.items():
            staged_identity = staging_dir / f"{method_id}{IDENTITY_SUFFIX}"
            if sha256_file(staged_identity) != artifact["sha256"]:
                raise ValueError(f"staged checkpoint identity changed: {method_id}")
        staged_lock_sha256 = sha256_file(lock_path)
        if output_lock_dir.exists():
            raise FileExistsError(
                "release-lock output appeared before atomic publication: "
                f"{output_lock_dir}"
            )
        staging_dir.rename(output_lock_dir)

        try:
            final_lock_path = output_lock_dir / LOCK_FILENAME
            verified = _load_post_selection_lock(
                final_lock_path,
                protocol_sha256=protocol_sha256,
                checkpoint_selection_rule=selection_rule,
                pilot_seed=seed,
            )
            if verified["sha256"] != staged_lock_sha256:
                raise ValueError("release lock changed during atomic publication")
            _assert_snapshots_unchanged(initial_snapshots)
        except Exception:
            # The directory was created exclusively by this invocation.  Move
            # it back under the disposable staging root so a failed final
            # verification never leaves an admitted-looking publication.
            rejected = temporary_root / "rejected_release_lock"
            if output_lock_dir.exists() and not rejected.exists():
                output_lock_dir.rename(rejected)
            raise

    return {
        "schema_version": 1,
        "lock_id": lock_id,
        "output_lock_dir": str(output_lock_dir),
        "lock_path": str(output_lock_dir / LOCK_FILENAME),
        "lock_sha256": staged_lock_sha256,
        "protocol_sha256": protocol_sha256,
        "selection_rule": selection_rule,
        "seed": seed,
        "identity_files": {
            method_id: {
                "path": str(output_lock_dir / f"{method_id}{IDENTITY_SUFFIX}"),
                "sha256": artifact["sha256"],
                "checkpoint_id": identities[method_id]["checkpoint_id"],
            }
            for method_id, artifact in sorted(identity_artifacts.items())
        },
        "input_sha256": dict(payload["input_sha256"]),
        "analysis_selection_implementation": {
            "schema_version": 1,
            "canonical_digest": analysis_selection_implementation[
                "canonical_digest"
            ],
            "holdout_selection_digest": analysis_selection_implementation[
                "holdout_selection"
            ]["digest"],
            "evaluator_digest": analysis_selection_implementation[
                "evaluator"
            ]["digest"],
        },
        "history_assignment_release": {
            "schema_version": 1,
            "conditions": list(
                history_assignment_release["recipe"]["conditions"]
            ),
            "seed": history_assignment_release["recipe"]["seed"],
            "global_donor_shortlist_size": history_assignment_release[
                "recipe"
            ]["global_donor_shortlist_size"],
            "implementation_digest": history_assignment_release[
                "implementation"
            ]["digest"],
        },
        "qrels_deserialized": False,
        "source_test_resolved_or_opened": False,
        "published_atomically": True,
        "passed": True,
    }


def release_input_paths(
    *,
    raw_dir: str | Path,
    development_dir: str | Path,
    subsequent_scout_dir: str | Path,
) -> dict[str, Path]:
    """Return exactly the input-SHA keys consumed by the holdout materializer."""

    raw_dir = Path(raw_dir).resolve()
    development_dir = Path(development_dir).resolve()
    subsequent_scout_dir = Path(subsequent_scout_dir).resolve()
    recall_path, recall_variant = _resolve_source_path(raw_dir, "recall")
    items_path, items_variant = _resolve_source_path(raw_dir, "items")
    if recall_variant != "full" or items_variant != "full":
        raise ValueError("Motivation V1.2 release lock requires KuaiSearch Full")
    paths = {
        "source_recall_train": recall_path.resolve(),
        "source_items_train": items_path.resolve(),
        **{
            f"development_{name}": (development_dir / filename).resolve()
            for name, filename in _DEVELOPMENT_INPUT_FILES.items()
        },
        **{
            f"subsequent_scout_{name}": (
                subsequent_scout_dir / filename
            ).resolve()
            for name, filename in _SCOUT_INPUT_FILES.items()
        },
    }
    for name, path in paths.items():
        _existing_file(path, f"release input {name}")
    return paths


def _load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"protocol is not UTF-8: {path}") from exc
    payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("protocol must contain a mapping")
    if payload.get("schema_version") != 1:
        raise ValueError("protocol schema_version must equal 1")
    status = str(payload.get("status") or "").casefold()
    if not (
        status in {"frozen", "locked"}
        or status.endswith(("_frozen", "_locked", "-frozen", "-locked"))
    ):
        raise ValueError("protocol status must be frozen or locked")
    return payload, sha256_text(text)


def _selection_rule(protocol: Mapping[str, Any]) -> str:
    common = protocol.get("common_training")
    rule = common.get("checkpoint_selection") if isinstance(common, dict) else None
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("protocol has no checkpoint selection rule")
    return rule


def _pilot_seed(protocol: Mapping[str, Any]) -> int:
    policy = protocol.get("seed_policy")
    seed = policy.get("pilot_seed") if isinstance(policy, dict) else None
    if seed != V12_PILOT_SEED:
        raise ValueError(f"protocol pilot seed must equal {V12_PILOT_SEED}")
    return seed


def _load_and_validate_configs(
    paths: Mapping[str, Path],
    *,
    protocol_path: Path,
    protocol_sha256: str,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    configs = {}
    shas = {}
    for method_id in sorted(V12_METHOD_IDS):
        path = paths[method_id]
        raw = path.read_bytes()
        try:
            config = yaml.safe_load(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError(f"config is not UTF-8: {path}") from exc
        if not isinstance(config, dict):
            raise ValueError(f"config must contain a mapping: {method_id}")
        if config.get("schema_version") != 1:
            raise ValueError(f"config schema_version must equal 1: {method_id}")
        if config.get("method_id") != method_id:
            raise ValueError(f"config method_id mismatch: {method_id}")
        protocol_ref = config.get("protocol")
        if not isinstance(protocol_ref, dict):
            raise ValueError(f"config has no protocol identity: {method_id}")
        if protocol_ref.get("sha256") != protocol_sha256:
            raise ValueError(f"config protocol SHA mismatch: {method_id}")
        referenced_path = protocol_ref.get("path")
        if not isinstance(referenced_path, str) or not referenced_path.strip():
            raise ValueError(f"config protocol path is empty: {method_id}")
        if Path(referenced_path).resolve() != protocol_path:
            raise ValueError(f"config protocol path mismatch: {method_id}")
        training = config.get("training")
        if not isinstance(training, dict) or training.get("seed") != seed:
            raise ValueError(f"config pilot seed mismatch: {method_id}")
        configs[method_id] = config
        shas[method_id] = sha256_file(path)
    return configs, shas


def _build_q_identity(
    *,
    method_id: str,
    checkpoint_root: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
    seed: int,
    selection_rule: str,
    expected_implementation_digest: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    checkpoint_root = _existing_directory(
        checkpoint_root, f"{method_id} checkpoint root"
    ).resolve()
    metadata_path = _existing_file(
        checkpoint_root / Q_TRAINING_METADATA,
        f"{method_id} training metadata",
    ).resolve()
    model_dir = _existing_directory(
        checkpoint_root / Q_CHECKPOINT_MODEL_DIR,
        f"{method_id} checkpoint model directory",
    ).resolve()
    metadata = _read_json(metadata_path, f"{method_id} training metadata")
    _validate_common_training_metadata(
        metadata,
        method_id=method_id,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        seed=seed,
    )
    if metadata.get("resume_state_complete") is not True:
        raise ValueError(f"{method_id} checkpoint resume state is incomplete")
    progress = metadata.get("progress")
    expected_epochs = config.get("training", {}).get("epochs")
    if not isinstance(progress, dict) or progress.get("epoch") != expected_epochs:
        raise ValueError(f"{method_id} completed checkpoint is not at final epoch")

    checkpoint_files, artifact_paths = _q_checkpoint_files(model_dir)
    checkpoint_sha256 = _checkpoint_files_digest(checkpoint_files)
    checkpoint_id = f"{method_id}@{checkpoint_sha256[:20]}"
    if metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError(f"{method_id} checkpoint_id does not match inference files")
    metadata_files = _canonical_checkpoint_entries(
        metadata.get("checkpoint_weight_files"),
        source=f"{method_id} training metadata",
    )
    if metadata_files != checkpoint_files:
        raise ValueError(
            f"{method_id} training metadata inference-file identity mismatch"
        )
    implementation_digest = _metadata_implementation_digest(metadata, method_id)
    if implementation_digest != expected_implementation_digest:
        raise ValueError(
            f"{method_id} training implementation differs from current frozen code"
        )
    identity = _identity_payload(
        method_id=method_id,
        selection_rule=selection_rule,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        seed=seed,
        implementation_digest=implementation_digest,
        checkpoint_reference=model_dir,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_files=[
            {**entry, "path": str(artifact_paths[entry["name"]])}
            for entry in checkpoint_files
        ],
        training_metadata_path=metadata_path,
    )
    paths = {"training_metadata.json": metadata_path, **artifact_paths}
    return identity, paths


def _build_w0_identity(
    *,
    checkpoint_dir: Path,
    config: Mapping[str, Any],
    config_sha256: str,
    protocol_sha256: str,
    seed: int,
    selection_rule: str,
    expected_implementation_digest: str,
) -> tuple[dict[str, Any], dict[str, Path]]:
    metadata_path = _existing_file(
        checkpoint_dir / W0_TRAINING_METADATA, "W0 training metadata"
    ).resolve()
    model_path = _existing_file(checkpoint_dir / "model.pt", "W0 model").resolve()
    metadata = _read_json(metadata_path, "W0 training metadata")
    _validate_common_training_metadata(
        metadata,
        method_id=W0_METHOD_ID,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        seed=seed,
    )
    expected_epochs = config.get("training", {}).get("epochs")
    training = metadata.get("training")
    if not isinstance(training, dict) or training.get("next_epoch") != expected_epochs:
        raise ValueError("W0 completed checkpoint is not at final epoch")
    model_sha256 = sha256_file(model_path)
    checkpoint_id = f"{W0_METHOD_ID}@{model_sha256[:20]}"
    if metadata.get("model_sha256") != model_sha256:
        raise ValueError("W0 metadata model SHA mismatch")
    if metadata.get("checkpoint_id") != checkpoint_id:
        raise ValueError("W0 checkpoint_id does not match model.pt")
    implementation_digest = _metadata_implementation_digest(metadata, W0_METHOD_ID)
    if implementation_digest != expected_implementation_digest:
        raise ValueError("W0 training implementation differs from current frozen code")
    identity = _identity_payload(
        method_id=W0_METHOD_ID,
        selection_rule=selection_rule,
        config_sha256=config_sha256,
        protocol_sha256=protocol_sha256,
        seed=seed,
        implementation_digest=implementation_digest,
        checkpoint_reference=checkpoint_dir,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=model_sha256,
        checkpoint_files=[
            {
                "name": "model.pt",
                "path": str(model_path),
                "sha256": model_sha256,
                "size_bytes": model_path.stat().st_size,
            }
        ],
        training_metadata_path=metadata_path,
    )
    return identity, {"metadata.json": metadata_path, "model.pt": model_path}


def _identity_payload(
    *,
    method_id: str,
    selection_rule: str,
    config_sha256: str,
    protocol_sha256: str,
    seed: int,
    implementation_digest: str,
    checkpoint_reference: Path,
    checkpoint_id: str,
    checkpoint_sha256: str,
    checkpoint_files: list[dict[str, Any]],
    training_metadata_path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "motivation_v1_2_checkpoint_selection_identity",
        "method_id": method_id,
        "selection_frozen": True,
        "selection_rule": selection_rule,
        "config_sha256": config_sha256,
        "protocol_sha256": protocol_sha256,
        "seed": seed,
        "status": "completed",
        "evidence_mode": "first_round_pilot",
        "implementation_digest": implementation_digest,
        "checkpoint_reference": str(checkpoint_reference),
        "checkpoint_id": checkpoint_id,
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_files": checkpoint_files,
        "training_metadata_path": str(training_metadata_path),
        "training_metadata_sha256": sha256_file(training_metadata_path),
    }


def _validate_common_training_metadata(
    metadata: Mapping[str, Any],
    *,
    method_id: str,
    config_sha256: str,
    protocol_sha256: str,
    seed: int,
) -> None:
    expected = {
        "method_id": method_id,
        "status": "completed",
        "evidence_mode": "first_round_pilot",
        "seed": seed,
        "protocol_sha256": protocol_sha256,
        "config_sha256": config_sha256,
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            raise ValueError(
                f"training metadata mismatch for {method_id}.{field}: "
                f"{metadata.get(field)!r} != {value!r}"
            )


def _metadata_implementation_digest(
    metadata: Mapping[str, Any], method_id: str
) -> str:
    value = metadata.get("implementation_digest")
    nested = metadata.get("implementation_identity")
    nested_value = nested.get("digest") if isinstance(nested, dict) else None
    if value is None:
        value = nested_value
    elif nested_value is not None and nested_value != value:
        raise ValueError(f"{method_id} implementation identities disagree")
    if value is None:
        value = metadata.get("implementation_sha256")
    return _require_sha256(value, f"{method_id} implementation digest")


def _q_checkpoint_files(
    model_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    paths = sorted(
        path
        for path in model_dir.rglob("*")
        if path.is_file() and path.suffix in _Q_INFERENCE_SUFFIXES
    )
    if not any(path.suffix == ".safetensors" for path in paths):
        raise FileNotFoundError(
            f"Q checkpoint contains no safetensors weights: {model_dir}"
        )
    entries = []
    by_name = {}
    for path in paths:
        name = str(path.relative_to(model_dir))
        entries.append(
            {
                "name": name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
        by_name[name] = path.resolve()
    return entries, by_name


def _current_implementation_identity(
    role: str,
) -> tuple[dict[str, Any], list[Path]]:
    """Mirror the implementation identities recorded by the two trainers."""

    baseline_dir = Path(__file__).resolve().parents[1] / "baselines"
    if role == "q":
        paths = sorted(
            [
                baseline_dir / "motivation_v12_ranker.py",
                baseline_dir / "motivation_v12_contracts.py",
            ]
        )
    elif role == "w0":
        paths = sorted(
            [
                baseline_dir / "copps_transfer_witness.py",
                baseline_dir / "frozen_text_features.py",
                baseline_dir / "representative_sequence_adapter.py",
            ]
        )
    else:
        raise ValueError(f"unknown implementation role: {role}")
    for path in paths:
        _existing_file(path, f"{role} implementation source")
    values = [{"path": str(path), "sha256": sha256_file(path)} for path in paths]
    return {
        "digest": sha256_text(
            json.dumps(values, sort_keys=True, separators=(",", ":"))
        ),
        "files": values,
    }, paths


def _canonical_checkpoint_entries(value: Any, *, source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source} has no checkpoint files")
    entries = []
    names = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"{source} checkpoint file {index} is not an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError(f"{source} checkpoint file {index} has invalid name")
        names.add(name)
        sha256 = _require_sha256(
            raw.get("sha256"), f"{source} checkpoint file {index}.sha256"
        )
        size_bytes = raw.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise ValueError(f"{source} checkpoint file {index} has invalid size")
        entries.append(
            {"name": name, "sha256": sha256, "size_bytes": size_bytes}
        )
    return sorted(entries, key=lambda entry: entry["name"])


def _checkpoint_files_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(
        json.dumps(list(entries), sort_keys=True, separators=(",", ":"))
    )


def _assert_protocol_development_hashes(
    protocol: Mapping[str, Any],
    *,
    input_snapshots: Mapping[str, Mapping[str, Any]],
) -> None:
    data = protocol.get("data")
    population = data.get("development_population") if isinstance(data, dict) else None
    if not isinstance(population, dict):
        raise ValueError("protocol has no frozen development population")
    for logical_name, protocol_field in _PROTOCOL_DEVELOPMENT_SHA_FIELDS.items():
        expected = _require_sha256(
            population.get(protocol_field),
            f"protocol development_population.{protocol_field}",
        )
        input_name = f"development_{logical_name}"
        if input_snapshots[input_name]["sha256"] != expected:
            raise ValueError(
                f"development input differs from protocol freeze: {input_name}"
            )


def _snapshot_files(paths: Mapping[str, Path]) -> dict[str, dict[str, Any]]:
    snapshots = {}
    for name, path in paths.items():
        path = _existing_file(path, name).resolve()
        snapshots[name] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return snapshots


def _assert_snapshots_unchanged(snapshots: Mapping[str, Mapping[str, Any]]) -> None:
    for name, snapshot in snapshots.items():
        path = _existing_file(snapshot["path"], name)
        if path.stat().st_size != snapshot["size_bytes"]:
            raise ValueError(f"frozen release dependency size changed: {name}")
        if sha256_file(path) != snapshot["sha256"]:
            raise ValueError(f"frozen release dependency SHA changed: {name}")


def _normalize_exact_path_map(
    value: Mapping[str, str | Path],
    expected_keys: set[str] | frozenset[str],
    name: str,
    *,
    require_file: bool = True,
) -> dict[str, Path]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise ValueError(f"{name} must cover exactly {sorted(expected_keys)}")
    normalized = {}
    for key, raw_path in value.items():
        path = Path(raw_path)
        if require_file:
            path = _existing_file(path, f"{name}.{key}")
        else:
            path = _existing_directory(path, f"{name}.{key}")
        normalized[key] = path.resolve()
    return normalized


def _read_json(path: Path, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValueError(f"{name} is not UTF-8: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload


def _existing_file(path: str | Path, name: str) -> Path:
    path = Path(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing non-empty {name}: {path}")
    return path


def _existing_directory(path: str | Path, name: str) -> Path:
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"missing {name}: {path}")
    return path


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value
