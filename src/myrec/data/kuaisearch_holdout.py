"""Materialize the frozen Motivation V1.2 KuaiSearch confirmation holdout.

The materializer is deliberately separate from model training and scoring.  It
is the only pre-evaluation component allowed to inspect source-train labels in
order to write separated qrels and an evaluator-only sealed aggregate audit.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from myrec.data.contracts import audit_standardized_file
from myrec.data.kuaisearch_scout import (
    SourceRequest,
    _collect_source_state,
    _load_item_map,
    _load_selected_requests,
    _request_id,
    _resolve_source_path,
    _select_latest_time_window,
    _write_scout,
)
from myrec.utils.hashing import sha256_file, sha256_text
from myrec.utils.jsonl import iter_jsonl, write_json


V12_END_BEFORE_TIME = 1_862_529
V12_SOURCE_WINDOW_REQUESTS = 20_000
V12_CONFIRMATION_FRACTION = 0.20
V12_CONFIRMATION_REQUESTS = 4_000
V12_MAX_HISTORY_LEN = 20
V12_MIN_CANDIDATE_COUNT = 2
V12_MAX_CANDIDATE_COUNT = 100
V12_PILOT_SEED = 20_260_714
V12_DATASET_VERSION = "full_confirm_preceding40k_newholdout4k_v12"
EVALUATOR_ONLY_LABEL_AUDIT_FILENAME = "evaluator_only_label_power_audit.json"
V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES = (
    "src/myrec/data/kuaisearch_holdout.py",
    "src/myrec/data/kuaisearch_scout.py",
    "src/myrec/data/motivation_v12_release_lock.py",
)
V12_EVALUATOR_IMPLEMENTATION_FILES = (
    "src/myrec/eval/motivation_v12_evidence.py",
    "src/myrec/eval/history_response_evaluator.py",
    "src/myrec/eval/target_aware_surfaces.py",
    "src/myrec/eval/controlled_composition.py",
    "src/myrec/eval/history_response.py",
)
_SEALED_ONLY_LABEL_DERIVED_KEYS = frozenset(
    {
        "materializer_only_power_audit",
        "retrospective_training_leakage_audit",
        "source_writer_label_derived_counts",
        "observed_positive",
        "no_observed_positive",
        "clicked_positive_requests",
        "purchased_positive_requests",
        "positive_candidate_instances",
        "candidate_overlap",
        "no_candidate_overlap_history_present",
        "no_history",
        "target_repeat",
        "target_nonrepeat_other_candidate_overlap",
        "target_nonrepeat_no_candidate_overlap",
        "target_nonrepeat_no_history",
        "holdout_users",
        "overlapping_users",
        "train_requests_with_same_user",
        "holdout_positive_event_instances",
        "matching_history_event_instances",
        "train_requests_with_holdout_event_in_history",
        "holdout_requests_with_event_in_train_history",
    }
)
V12_METHOD_IDS = frozenset(
    {
        "q0_qwen3_reranker_06b",
        "q1_instructrec_generalqwen",
        "q2_recranker_generalqwen",
        "q3_tallrec_generalqwen",
        "w0_copps_style_transfer_witness",
    }
)


@dataclass(frozen=True)
class PopulationState:
    """Identity and visible-candidate state for one standardized population."""

    name: str
    split: str
    path: Path
    request_ids: frozenset[str]
    session_ids: frozenset[str]
    records: Mapping[str, tuple[str, tuple[str, ...]]]
    time_min: int
    time_max: int


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _analysis_selection_implementation_paths() -> dict[str, Path]:
    root = _repository_root()
    return {
        relative_path: root / relative_path
        for relative_path in (
            *V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES,
            *V12_EVALUATOR_IMPLEMENTATION_FILES,
        )
    }


def _implementation_file_entries(
    relative_paths: Sequence[str],
) -> list[dict[str, str]]:
    paths = _analysis_selection_implementation_paths()
    entries = []
    for relative_path in sorted(relative_paths):
        path = paths[relative_path]
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"missing load-bearing implementation file: {path}"
            )
        entries.append(
            {"path": relative_path, "sha256": sha256_file(path)}
        )
    return entries


def _implementation_files_digest(
    entries: Sequence[Mapping[str, str]],
) -> str:
    return sha256_text(
        json.dumps(list(entries), sort_keys=True, separators=(",", ":"))
    )


def _current_analysis_selection_implementation_identity() -> dict[str, Any]:
    """Return the complete code identity frozen by the release gate."""

    holdout_files = _implementation_file_entries(
        V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES
    )
    evaluator_files = _implementation_file_entries(
        V12_EVALUATOR_IMPLEMENTATION_FILES
    )
    holdout_digest = _implementation_files_digest(holdout_files)
    evaluator_digest = _implementation_files_digest(evaluator_files)
    canonical_payload = {
        "evaluator": evaluator_digest,
        "holdout_selection": holdout_digest,
    }
    return {
        "schema_version": 1,
        "canonical_digest": sha256_text(
            json.dumps(
                canonical_payload, sort_keys=True, separators=(",", ":")
            )
        ),
        "holdout_selection": {
            "digest": holdout_digest,
            "files": holdout_files,
        },
        "evaluator": {
            "digest": evaluator_digest,
            "files": evaluator_files,
        },
    }


def _analysis_selection_implementation_summary(
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "canonical_digest": identity["canonical_digest"],
        "holdout_selection_digest": identity["holdout_selection"]["digest"],
        "evaluator_digest": identity["evaluator"]["digest"],
    }


def _validate_analysis_selection_implementation_identity(
    value: Any,
    *,
    verify_current_holdout_selection: bool,
) -> dict[str, Any]:
    """Validate the release identity without consuming evaluator code here."""

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(
            "analysis_selection_implementation schema_version must equal 1"
        )
    canonical_digest = _require_sha256(
        value.get("canonical_digest"),
        "analysis_selection_implementation.canonical_digest",
    )
    validated_groups: dict[str, dict[str, Any]] = {}
    expected_paths = {
        "holdout_selection": set(V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES),
        "evaluator": set(V12_EVALUATOR_IMPLEMENTATION_FILES),
    }
    for group_name, required_paths in expected_paths.items():
        group = value.get(group_name)
        if not isinstance(group, dict):
            raise ValueError(
                f"analysis_selection_implementation.{group_name} must be an object"
            )
        declared_digest = _require_sha256(
            group.get("digest"),
            f"analysis_selection_implementation.{group_name}.digest",
        )
        files = group.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(
                "analysis_selection_implementation."
                f"{group_name}.files must be non-empty"
            )
        canonical_files: list[dict[str, str]] = []
        seen_paths: set[str] = set()
        for index, entry in enumerate(files):
            if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
                raise ValueError(
                    "analysis_selection_implementation file entries must contain "
                    "exactly path and sha256"
                )
            relative_path = entry.get("path")
            if (
                not isinstance(relative_path, str)
                or not relative_path.strip()
                or Path(relative_path).is_absolute()
                or ".." in Path(relative_path).parts
                or relative_path in seen_paths
            ):
                raise ValueError(
                    "invalid/duplicate analysis-selection implementation path: "
                    f"{relative_path!r}"
                )
            seen_paths.add(relative_path)
            canonical_files.append(
                {
                    "path": relative_path,
                    "sha256": _require_sha256(
                        entry.get("sha256"),
                        "analysis_selection_implementation."
                        f"{group_name}.files[{index}].sha256",
                    ),
                }
            )
        canonical_files.sort(key=lambda entry: entry["path"])
        if seen_paths != required_paths:
            raise ValueError(
                "analysis-selection implementation file coverage mismatch for "
                f"{group_name}: missing={sorted(required_paths - seen_paths)}, "
                f"unexpected={sorted(seen_paths - required_paths)}"
            )
        if _implementation_files_digest(canonical_files) != declared_digest:
            raise ValueError(
                f"analysis-selection implementation digest mismatch: {group_name}"
            )
        validated_groups[group_name] = {
            "digest": declared_digest,
            "files": canonical_files,
        }

    expected_canonical_digest = sha256_text(
        json.dumps(
            {
                "evaluator": validated_groups["evaluator"]["digest"],
                "holdout_selection": validated_groups["holdout_selection"][
                    "digest"
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    if canonical_digest != expected_canonical_digest:
        raise ValueError(
            "analysis-selection canonical implementation digest mismatch"
        )
    if verify_current_holdout_selection:
        current_files = _implementation_file_entries(
            V12_HOLDOUT_SELECTION_IMPLEMENTATION_FILES
        )
        if current_files != validated_groups["holdout_selection"]["files"]:
            raise ValueError(
                "current holdout/selection implementation differs from release lock"
            )
    return {
        "schema_version": 1,
        "canonical_digest": canonical_digest,
        **validated_groups,
    }


def _validate_history_assignment_release(
    value: Any, *, pilot_seed: int
) -> dict[str, Any]:
    """Require the exact post-release assignment recipe and current code bytes."""

    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "recipe",
        "implementation",
    }:
        raise ValueError(
            "history_assignment_release must contain exactly schema_version, "
            "recipe, and implementation"
        )
    if value.get("schema_version") != 1:
        raise ValueError("history_assignment_release schema_version must equal 1")
    from myrec.data.history_assignments import (
        current_motivation_v12_assignment_implementation_identity,
        motivation_v12_assignment_recipe,
    )

    expected_recipe = motivation_v12_assignment_recipe()
    if expected_recipe.get("seed") != pilot_seed:
        raise ValueError(
            "history assignment recipe seed differs from the frozen pilot seed"
        )
    if value.get("recipe") != expected_recipe:
        raise ValueError(
            "history assignment recipe differs from the registered frozen recipe"
        )
    current_implementation = (
        current_motivation_v12_assignment_implementation_identity()
    )
    if value.get("implementation") != current_implementation:
        raise ValueError(
            "current history assignment implementation differs from release lock"
        )
    return {
        "schema_version": 1,
        "recipe": expected_recipe,
        "implementation": current_implementation,
    }


def materialize_motivation_v12_kuaisearch_holdout(
    *,
    raw_dir: str | Path,
    development_dir: str | Path,
    subsequent_scout_dir: str | Path,
    output_dir: str | Path,
    protocol_path: str | Path,
    recipe_checkpoint_lock_path: str | Path,
    dataset_version: str = V12_DATASET_VERSION,
    command_argv: Sequence[str] | None = None,
    enforce_registered_v12_recipe: bool = True,
) -> dict[str, Any]:
    """Reuse V1.1 train/dev and add one earlier, recipe-locked confirmation.

    A separate source-test path is never resolved or opened.  Logical non-train
    rows in KuaiSearch's physically mixed recall file are split-gated before
    JSON decoding, so their behavior fields are not exposed.  The confirmation
    is selected from the final fraction of a source-train-only window strictly
    before the development population.  Every known population is audited for
    pairwise request/session disjointness and strict time ordering before
    outputs are created.
    """

    raw_dir = Path(raw_dir)
    development_dir = Path(development_dir)
    subsequent_scout_dir = Path(subsequent_scout_dir)
    output_dir = Path(output_dir)
    protocol_path = Path(protocol_path)
    recipe_checkpoint_lock_path = Path(recipe_checkpoint_lock_path)
    if enforce_registered_v12_recipe and dataset_version != V12_DATASET_VERSION:
        raise ValueError(
            "registered holdout dataset_version must equal "
            f"{V12_DATASET_VERSION}"
        )
    if output_dir.exists() and not output_dir.is_dir():
        raise FileExistsError(f"holdout output path is not a directory: {output_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"holdout output directory is not empty: {output_dir}")
    if output_dir.exists():
        output_dir.rmdir()

    protocol = _load_holdout_protocol(
        protocol_path,
        enforce_registered_v12_recipe=enforce_registered_v12_recipe,
    )
    release_lock = _load_post_selection_lock(
        recipe_checkpoint_lock_path,
        protocol_sha256=protocol["sha256"],
        checkpoint_selection_rule=protocol["checkpoint_selection_rule"],
        pilot_seed=protocol["pilot_seed"],
    )
    recipe = protocol["recipe"]
    scope_warning = (
        "The new confirmation is a retrospective, earlier source-train "
        "population whose timestamps precede the reused 32k train split. "
        "It proves frozen recipe/request/session isolation, not forward "
        "temporal generalization. User, item, and query isolation are not "
        "claimed; later training histories can contain earlier holdout events."
    )
    protocol_file = _file_info(protocol_path)
    release_lock_file = _file_info(recipe_checkpoint_lock_path)
    if protocol_file["sha256"] != protocol["sha256"]:
        raise ValueError("protocol changed while opening the freeze gate")
    if release_lock_file["sha256"] != release_lock["sha256"]:
        raise ValueError("post-selection lock changed while opening the freeze gate")

    recall_path, recall_variant = _resolve_source_path(raw_dir, "recall")
    items_path, items_variant = _resolve_source_path(raw_dir, "items")
    if recall_variant != "full" or items_variant != "full":
        raise ValueError(
            "Motivation V1.2 holdout must use the KuaiSearch Full source-train files"
        )

    development_paths = {
        "manifest": development_dir / "manifest.json",
        "candidate_manifest": development_dir / "candidate_manifest.json",
        "request_manifest": development_dir / "request_manifest.json",
        "records_train": development_dir / "records_train.jsonl",
        "records_dev": development_dir / "records_dev.jsonl",
        "records_confirmation": development_dir / "records_confirmation.jsonl",
        "qrels_train": development_dir / "qrels_train.jsonl",
        "qrels_dev": development_dir / "qrels_dev.jsonl",
    }
    scout_paths = {
        "manifest": subsequent_scout_dir / "manifest.json",
        "candidate_manifest": subsequent_scout_dir / "candidate_manifest.json",
        "request_manifest": subsequent_scout_dir / "request_manifest.json",
        "records_train": subsequent_scout_dir / "records_train.jsonl",
        "records_dev": subsequent_scout_dir / "records_dev.jsonl",
    }
    _require_files(
        {
            **development_paths,
            **{f"scout_{key}": value for key, value in scout_paths.items()},
        }
    )
    frozen_input_paths = {
        "source_recall_train": recall_path,
        "source_items_train": items_path,
        **{
            f"development_{key}": value
            for key, value in development_paths.items()
        },
        **{
            f"subsequent_scout_{key}": value
            for key, value in scout_paths.items()
        },
    }
    frozen_input_files = _validate_release_input_shas(
        release_lock["payload"], frozen_input_paths
    )
    input_files = {
        "protocol": protocol_file,
        "post_selection_recipe_checkpoint_lock": release_lock_file,
        **frozen_input_files,
    }

    populations = [
        _load_population(
            "development_train", development_paths["records_train"], "train"
        ),
        _load_population(
            "development_internal_dev", development_paths["records_dev"], "dev"
        ),
        _load_population(
            "legacy_confirmation",
            development_paths["records_confirmation"],
            "confirmation",
        ),
        _load_population(
            "subsequent_scout_train", scout_paths["records_train"], "train"
        ),
        _load_population(
            "subsequent_scout_dev", scout_paths["records_dev"], "dev"
        ),
    ]
    _validate_development_lock(
        protocol["payload"],
        development_paths,
        populations[:3],
        require_frozen_shas=enforce_registered_v12_recipe,
    )
    base_candidate_manifest = _read_json(development_paths["candidate_manifest"])
    base_request_manifest = _read_json(development_paths["request_manifest"])
    scout_candidate_manifest = _read_json(scout_paths["candidate_manifest"])
    scout_request_manifest = _read_json(scout_paths["request_manifest"])
    _validate_identity_manifests(
        populations[:3], base_candidate_manifest, base_request_manifest
    )
    _validate_identity_manifests(
        populations[3:], scout_candidate_manifest, scout_request_manifest
    )

    source = _collect_source_state(
        recall_path,
        min_candidate_count=recipe["min_candidate_count"],
        max_candidate_count=recipe["max_candidate_count"],
    )
    selected_keys, split_by_key, split_info = _select_latest_time_window(
        source["eligible_keys"],
        max_requests=recipe["source_window_requests"],
        dev_fraction=recipe["confirmation_fraction"],
        evaluation_split="confirmation",
        end_before_time=recipe["end_before_time"],
    )
    if len(selected_keys) != recipe["source_window_requests"]:
        raise ValueError(
            "source window is smaller than the recipe lock: "
            f"{len(selected_keys)} != {recipe['source_window_requests']}"
        )
    confirmation_keys = {
        key for key, split in split_by_key.items() if split == "confirmation"
    }
    if len(confirmation_keys) != recipe["confirmation_requests"]:
        raise ValueError(
            "time-tie/session containment changed the fixed confirmation size: "
            f"{len(confirmation_keys)} != {recipe['confirmation_requests']}"
        )
    confirmation_requests = _load_selected_requests(
        recall_path,
        selected_keys=confirmation_keys,
        split_by_key={key: "confirmation" for key in confirmation_keys},
        events_by_user=source["events_by_user"],
        max_history_len=recipe["max_history_len"],
    )
    power_counts = _aggregate_power_counts(confirmation_requests)
    new_population = _population_from_source_requests(
        "new_confirmation",
        output_dir / "records_confirmation.jsonl",
        confirmation_requests,
    )
    all_populations = [*populations, new_population]
    overlap_audit = _cross_population_overlap_audit(all_populations)
    time_audit = _time_isolation_audit(
        all_populations,
        end_before_time=recipe["end_before_time"],
    )
    retrospective_leakage = _retrospective_training_leakage_audit(
        confirmation_requests,
        development_paths["records_train"],
    )

    needed_item_ids = {
        item_id
        for request in confirmation_requests
        for item_id in (
            *request.candidate_item_ids,
            *(event[1] for event in request.history),
        )
    }
    item_map = _load_item_map(items_path, needed_item_ids)
    missing_item_ids = needed_item_ids - set(item_map)
    if missing_item_ids:
        raise ValueError(
            "confirmation item metadata coverage is incomplete: "
            f"{len(missing_item_ids)} missing, examples={sorted(missing_item_ids)[:5]}"
        )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output_dir.name}.confirmation-", dir=output_dir.parent
    ) as temp:
        staging_dir = Path(temp) / "generated_confirmation"
        assembled_dir = Path(temp) / "assembled_output"
        staging_dir.mkdir()
        assembled_dir.mkdir()
        incomplete_marker = assembled_dir / ".materialization_incomplete"
        incomplete_marker.write_text(
            "Motivation V1.2 holdout publication has not completed.\n",
            encoding="utf-8",
        )
        generated = _write_scout(
            staging_dir,
            confirmation_requests,
            item_map=item_map,
            dataset_version=dataset_version,
            include_history_query=recipe["include_history_query"],
            evaluation_split="confirmation",
            output_splits=("confirmation",),
        )
        (
            public_source_writer_counts,
            source_writer_label_derived_counts,
        ) = _partition_source_writer_counts(generated["counts"])

        for split in ("train", "dev"):
            shutil.copy2(
                development_paths[f"records_{split}"],
                assembled_dir / f"records_{split}.jsonl",
            )
            shutil.copy2(
                development_paths[f"qrels_{split}"],
                assembled_dir / f"qrels_{split}.jsonl",
            )
        shutil.copy2(
            staging_dir / "records_confirmation.jsonl",
            assembled_dir / "records_confirmation.jsonl",
        )
        shutil.copy2(
            staging_dir / "qrels_confirmation.jsonl",
            assembled_dir / "qrels_confirmation.jsonl",
        )
        sealed_label_audit_path = (
            assembled_dir / EVALUATOR_ONLY_LABEL_AUDIT_FILENAME
        )
        write_json(
            sealed_label_audit_path,
            {
                "schema_version": 1,
                "kind": "motivation_v1_2_evaluator_only_label_power_audit",
                "dataset_id": "kuaisearch",
                "dataset_version": dataset_version,
                "protocol_sha256": protocol["sha256"],
                "post_selection_recipe_checkpoint_lock_sha256": release_lock[
                    "sha256"
                ],
                "access_policy": {
                    "evaluator_only": True,
                    "open_after_score_audit_only": True,
                    "model_and_scorer_access_forbidden": True,
                },
                "materializer_only_power_audit": {
                    "aggregate_only": True,
                    "definitions_match_evaluator_target_aware_partition": True,
                    **power_counts,
                },
                "retrospective_training_leakage_audit": (
                    retrospective_leakage
                ),
                "source_writer_label_derived_counts": (
                    source_writer_label_derived_counts
                ),
            },
        )

        new_candidate_manifest = _read_json(staging_dir / "candidate_manifest.json")
        new_request_manifest = _read_json(staging_dir / "request_manifest.json")
        combined_candidate_manifest = {
            "dataset_version": dataset_version,
            "entries": [
                *_entries_for_splits(
                    base_candidate_manifest, {"train", "dev"}
                ),
                *_entries_for_splits(new_candidate_manifest, {"confirmation"}),
            ],
        }
        combined_request_manifest = {
            "dataset_version": dataset_version,
            "entries": [
                *_entries_for_splits(base_request_manifest, {"train", "dev"}),
                *_entries_for_splits(new_request_manifest, {"confirmation"}),
            ],
        }
        candidate_manifest_path = assembled_dir / "candidate_manifest.json"
        request_manifest_path = assembled_dir / "request_manifest.json"
        write_json(candidate_manifest_path, combined_candidate_manifest)
        write_json(request_manifest_path, combined_request_manifest)

        assembled_populations = [
            _load_population(
                "assembled_train", assembled_dir / "records_train.jsonl", "train"
            ),
            _load_population(
                "assembled_dev", assembled_dir / "records_dev.jsonl", "dev"
            ),
            _load_population(
                "assembled_confirmation",
                assembled_dir / "records_confirmation.jsonl",
                "confirmation",
            ),
        ]
        _validate_identity_manifests(
            assembled_populations,
            combined_candidate_manifest,
            combined_request_manifest,
        )
        if assembled_populations[0].request_ids != populations[0].request_ids:
            raise ValueError("reused development train request population changed")
        if assembled_populations[1].request_ids != populations[1].request_ids:
            raise ValueError("reused internal-dev request population changed")
        output_filenames = {
            "records_train": "records_train.jsonl",
            "records_dev": "records_dev.jsonl",
            "records_confirmation": "records_confirmation.jsonl",
            "qrels_train": "qrels_train.jsonl",
            "qrels_dev": "qrels_dev.jsonl",
            "qrels_confirmation": "qrels_confirmation.jsonl",
            "candidate_manifest": "candidate_manifest.json",
            "request_manifest": "request_manifest.json",
            "evaluator_only_sealed_label_audit": (
                EVALUATOR_ONLY_LABEL_AUDIT_FILENAME
            ),
        }
        output_files = {
            name: _file_info(
                assembled_dir / filename,
                reported_path=output_dir / filename,
            )
            for name, filename in output_filenames.items()
        }
        output_files["evaluator_only_sealed_label_audit"][
            "open_after_score_audit_only"
        ] = True
        sealed_label_audit_reference = dict(
            output_files["evaluator_only_sealed_label_audit"]
        )
        structural_audits = {}
        for population in assembled_populations:
            audit = audit_standardized_file(population.path, population.split)
            audit["path"] = str(
                output_dir / f"records_{population.split}.jsonl"
            )
            structural_audits[population.split] = audit

        integrity_lock_path = assembled_dir / "confirmation_integrity_lock.json"
        integrity_lock_payload = {
            "schema_version": 1,
            "dataset_id": "kuaisearch",
            "dataset_version": dataset_version,
            "protocol_sha256": protocol["sha256"],
            "post_selection_recipe_checkpoint_lock_sha256": release_lock[
                "sha256"
            ],
            "scope_warning": scope_warning,
            "retrospective_time_direction": time_audit,
            "analysis_selection_implementation": (
                _analysis_selection_implementation_summary(
                    release_lock["analysis_selection_implementation"]
                )
            ),
            "history_assignment_release": release_lock[
                "history_assignment_release"
            ],
            "evaluator_only_sealed_label_audit": (
                sealed_label_audit_reference
            ),
            "files": {
                name: {
                    "sha256": info["sha256"],
                    "size_bytes": info["size_bytes"],
                }
                for name, info in output_files.items()
                if name != "evaluator_only_sealed_label_audit"
            },
            "qrels_open_policy": (
                "shared evaluator must verify this lock and score-bundle "
                "integrity before opening qrels_confirmation"
            ),
        }
        _assert_public_metadata_label_opaque(
            integrity_lock_payload, source="confirmation integrity lock"
        )
        write_json(integrity_lock_path, integrity_lock_payload)
        output_files["confirmation_integrity_lock"] = _file_info(
            integrity_lock_path,
            reported_path=output_dir / integrity_lock_path.name,
        )
        manifest = {
            "schema_version": 1,
            "dataset_id": "kuaisearch",
            "dataset_version": dataset_version,
            "evidence_mode": "recipe_locked_source_train_new_confirmation",
            "scope_warning": scope_warning,
            "analysis_selection_implementation": (
                _analysis_selection_implementation_summary(
                    release_lock["analysis_selection_implementation"]
                )
            ),
            "history_assignment_release": release_lock[
                "history_assignment_release"
            ],
            "evaluator_only_sealed_label_audit": (
                sealed_label_audit_reference
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "invocation": {
                "argv": list(command_argv) if command_argv is not None else None,
                "parameters": {
                    "raw_dir": str(raw_dir),
                    "development_dir": str(development_dir),
                    "subsequent_scout_dir": str(subsequent_scout_dir),
                    "output_dir": str(output_dir),
                    "protocol_path": str(protocol_path),
                    "recipe_checkpoint_lock_path": str(
                        recipe_checkpoint_lock_path
                    ),
                    "dataset_version": dataset_version,
                    "enforce_registered_v12_recipe": enforce_registered_v12_recipe,
                },
            },
            "freeze_gate": {
                "protocol": {
                    "path": str(protocol_path),
                    "sha256": protocol["sha256"],
                    "protocol_id": protocol["payload"].get("protocol_id"),
                    "status": protocol["payload"].get("status"),
                    "materialize_only_after_recipe_lock_verified": True,
                    "effective_holdout_recipe": recipe,
                    "checkpoint_selection_rule": protocol[
                        "checkpoint_selection_rule"
                    ],
                    "pilot_seed": protocol["pilot_seed"],
                },
                "post_selection_recipe_checkpoint_lock": {
                    "path": str(recipe_checkpoint_lock_path),
                    "sha256": release_lock["sha256"],
                    "lock_id": release_lock["payload"].get("lock_id"),
                    "status": release_lock["payload"].get("status"),
                    "protocol_sha256_verified": True,
                    "post_selection_release_verified": True,
                    "holdout_materialization_release": release_lock["release"],
                    "frozen_configs": release_lock["payload"]["frozen_configs"],
                    "frozen_checkpoints": release_lock["payload"][
                        "frozen_checkpoints"
                    ],
                    "analysis_selection_implementation": (
                        _analysis_selection_implementation_summary(
                            release_lock[
                                "analysis_selection_implementation"
                            ]
                        )
                    ),
                    "history_assignment_release": release_lock[
                        "history_assignment_release"
                    ],
                },
                "pre_publish_full_revalidation_completed": True,
            },
            "source": {
                "raw_dir": str(raw_dir),
                "variant": recall_variant,
                "source_splits_used": ["train"],
                "separate_source_test_path_resolved": False,
                "separate_source_test_path_opened": False,
                "mixed_physical_source_file_hashed_as_opaque_bytes": True,
                "mixed_file_non_train_rows_split_marker_scanned": True,
                "mixed_file_non_train_json_payloads_deserialized": False,
                "mixed_file_non_train_behavior_fields_accessed": False,
                "model_outputs_read": False,
                "excluded_non_train_rows_inside_source_train_file": dict(
                    source["excluded_split_counts"]
                ),
            },
            "selection": {
                "strategy": (
                    "latest fixed source-train window before exclusive cutoff; "
                    "confirmation is the final time-contained locked fraction"
                ),
                "source_window_requests": len(selected_keys),
                "discarded_earlier_buffer_requests": (
                    len(selected_keys) - len(confirmation_keys)
                ),
                "confirmation_requests": len(confirmation_keys),
                "end_before_time_exclusive": recipe["end_before_time"],
                "selection_uses_model_outputs": False,
                **split_info,
            },
            "population_isolation": {
                "overlap": overlap_audit,
                "time": time_audit,
                "retrospective_training_leakage_audit_sealed": True,
            },
            "label_isolation": {
                "train_records_have_supervision": True,
                "internal_dev_records_label_free": True,
                "confirmation_records_label_free": True,
                "confirmation_qrels_separate": True,
                "candidate_and_request_manifests_label_free": True,
                "confirmation_qrels_model_side_access": False,
                "confirmation_qrels_evaluator_access_only_after_score_integrity": True,
                "confirmation_integrity_lock_written": True,
                "source_labels_used_only_by_materializer": True,
                "per_request_label_aware_surface_artifacts_written": False,
                "evaluator_only_sealed_label_audit": (
                    sealed_label_audit_reference
                ),
            },
            "candidate_contract": {
                "combined_manifest_covers": ["train", "dev", "confirmation"],
                "base_train_dev_projection_reused": True,
                "record_manifest_candidate_identity_verified": True,
                "needed_confirmation_item_ids": len(needed_item_ids),
                "loaded_confirmation_item_ids": len(item_map),
                "missing_confirmation_item_ids": 0,
            },
            "inputs": input_files,
            "outputs": {
                "files": output_files,
                "structural_audits": structural_audits,
                "development_records_byte_identical": {
                    split: (
                        sha256_file(development_paths[f"records_{split}"])
                        == sha256_file(assembled_dir / f"records_{split}.jsonl")
                    )
                    for split in ("train", "dev")
                },
                "development_qrels_byte_identical": {
                    split: (
                        sha256_file(development_paths[f"qrels_{split}"])
                        == sha256_file(assembled_dir / f"qrels_{split}.jsonl")
                    )
                    for split in ("train", "dev")
                },
                "manifest": {
                    "path": str(output_dir / "manifest.json"),
                    "sha256_status": "self_reference_not_recorded",
                },
                "staging_marker_removed_before_publish": True,
                "published_atomically_by_single_directory_rename": True,
            },
            "source_writer_audit": {
                "label_opaque_counts": public_source_writer_counts,
                "candidate_count": generated["candidate_count"],
                "history_length": generated["history_length"],
                "repeated_query_requests": generated["repeated_query_requests"],
                "temporary_paths_not_promoted": True,
            },
            "admission_passed": True,
        }
        _assert_public_metadata_label_opaque(
            manifest, source="published holdout manifest"
        )
        write_json(assembled_dir / "manifest.json", manifest)
        incomplete_marker.unlink()

        _revalidate_frozen_state(
            protocol_path=protocol_path,
            protocol=protocol,
            recipe_checkpoint_lock_path=recipe_checkpoint_lock_path,
            release_lock=release_lock,
            frozen_input_paths=frozen_input_paths,
            enforce_registered_v12_recipe=enforce_registered_v12_recipe,
        )
        _atomic_publish_directory(assembled_dir, output_dir)

    return manifest


def verify_published_holdout(
    standardized_dir: str | Path,
    *,
    protocol_path: str | Path | None = None,
    recipe_checkpoint_lock_path: str | Path | None = None,
    open_qrels: bool = False,
    enforce_registered_v12_recipe: bool = True,
) -> dict[str, Any]:
    """Verify a published holdout; qrels remain unopened unless requested."""

    standardized_dir = Path(standardized_dir)
    marker = standardized_dir / ".materialization_incomplete"
    if marker.exists():
        raise ValueError(f"published holdout still has an incomplete marker: {marker}")
    manifest_path = standardized_dir / "manifest.json"
    integrity_lock_path = standardized_dir / "confirmation_integrity_lock.json"
    manifest = _read_json(manifest_path)
    integrity_lock = _read_json(integrity_lock_path)
    _assert_public_metadata_label_opaque(
        manifest, source="published holdout manifest"
    )
    _assert_public_metadata_label_opaque(
        integrity_lock, source="confirmation integrity lock"
    )
    if manifest.get("schema_version") != 1 or integrity_lock.get("schema_version") != 1:
        raise ValueError("published holdout manifest/lock schema_version must equal 1")
    if manifest.get("dataset_id") != "kuaisearch" or integrity_lock.get(
        "dataset_id"
    ) != "kuaisearch":
        raise ValueError("published holdout dataset_id must equal kuaisearch")
    if manifest.get("admission_passed") is not True:
        raise ValueError("published holdout manifest is not admitted")
    if manifest.get("dataset_version") != integrity_lock.get("dataset_version"):
        raise ValueError("published holdout manifest/lock dataset version mismatch")
    if enforce_registered_v12_recipe and manifest.get(
        "dataset_version"
    ) != V12_DATASET_VERSION:
        raise ValueError(
            "registered holdout dataset_version must equal "
            f"{V12_DATASET_VERSION}"
        )
    if manifest.get("scope_warning") != integrity_lock.get("scope_warning"):
        raise ValueError("published holdout scope warning differs from integrity lock")
    population_isolation = manifest.get("population_isolation")
    if not isinstance(population_isolation, dict):
        raise ValueError("published holdout has no population isolation audit")
    if population_isolation.get("time") != integrity_lock.get(
        "retrospective_time_direction"
    ):
        raise ValueError("published holdout time audit differs from integrity lock")
    if "retrospective_training_leakage_audit" in population_isolation or (
        "retrospective_training_leakage_audit" in integrity_lock
    ):
        raise ValueError(
            "label-derived retrospective leakage audit must remain sealed"
        )
    if "materializer_only_power_audit" in manifest or (
        "materializer_only_power_audit" in integrity_lock
    ):
        raise ValueError("materializer-only power audit must remain sealed")

    freeze_gate = manifest.get("freeze_gate")
    if not isinstance(freeze_gate, dict):
        raise ValueError("published holdout has no freeze gate")
    protocol_gate = freeze_gate.get("protocol")
    release_gate = freeze_gate.get("post_selection_recipe_checkpoint_lock")
    if not isinstance(protocol_gate, dict) or not isinstance(release_gate, dict):
        raise ValueError("published holdout freeze gate is incomplete")
    if protocol_path is None:
        protocol_path = protocol_gate.get("path")
    if recipe_checkpoint_lock_path is None:
        recipe_checkpoint_lock_path = release_gate.get("path")
    protocol_path = Path(str(protocol_path))
    recipe_checkpoint_lock_path = Path(str(recipe_checkpoint_lock_path))
    protocol = _load_holdout_protocol(
        protocol_path,
        enforce_registered_v12_recipe=enforce_registered_v12_recipe,
    )
    if protocol["sha256"] != protocol_gate.get("sha256"):
        raise ValueError("published holdout protocol SHA mismatch")
    if protocol["sha256"] != integrity_lock.get("protocol_sha256"):
        raise ValueError("published holdout integrity lock protocol SHA mismatch")
    release_lock = _load_post_selection_lock(
        recipe_checkpoint_lock_path,
        protocol_sha256=protocol["sha256"],
        checkpoint_selection_rule=protocol["checkpoint_selection_rule"],
        pilot_seed=protocol["pilot_seed"],
    )
    if release_lock["sha256"] != release_gate.get("sha256"):
        raise ValueError("published holdout release-lock SHA mismatch")
    if release_lock["sha256"] != integrity_lock.get(
        "post_selection_recipe_checkpoint_lock_sha256"
    ):
        raise ValueError("published integrity lock release-lock SHA mismatch")
    analysis_selection_summary = _analysis_selection_implementation_summary(
        release_lock["analysis_selection_implementation"]
    )
    if manifest.get(
        "analysis_selection_implementation"
    ) != analysis_selection_summary or integrity_lock.get(
        "analysis_selection_implementation"
    ) != analysis_selection_summary:
        raise ValueError(
            "published holdout analysis/selection implementation identity mismatch"
        )
    history_assignment_release = release_lock["history_assignment_release"]
    if (
        manifest.get("history_assignment_release")
        != history_assignment_release
        or integrity_lock.get("history_assignment_release")
        != history_assignment_release
        or release_gate.get("history_assignment_release")
        != history_assignment_release
    ):
        raise ValueError(
            "published holdout history assignment release identity mismatch"
        )
    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise ValueError("published holdout has no input identities")
    expected_input_shas = {
        "protocol": protocol["sha256"],
        "post_selection_recipe_checkpoint_lock": release_lock["sha256"],
    }
    for name, expected_sha256 in expected_input_shas.items():
        value = manifest_inputs.get(name)
        if not isinstance(value, dict) or value.get("sha256") != expected_sha256:
            raise ValueError(f"published holdout input identity mismatch: {name}")

    output_manifest = manifest.get("outputs")
    output_files = output_manifest.get("files") if isinstance(output_manifest, dict) else None
    locked_files = integrity_lock.get("files")
    if not isinstance(output_files, dict) or not isinstance(locked_files, dict):
        raise ValueError("published holdout output file identities are incomplete")
    sealed_label_audit_reference = _validate_sealed_label_audit_reference(
        manifest.get("evaluator_only_sealed_label_audit"),
        standardized_dir=standardized_dir,
    )
    if integrity_lock.get(
        "evaluator_only_sealed_label_audit"
    ) != sealed_label_audit_reference or output_files.get(
        "evaluator_only_sealed_label_audit"
    ) != sealed_label_audit_reference:
        raise ValueError(
            "published evaluator-only sealed label-audit identity mismatch"
        )
    required_files = {
        "records_train": "records_train.jsonl",
        "records_dev": "records_dev.jsonl",
        "records_confirmation": "records_confirmation.jsonl",
        "candidate_manifest": "candidate_manifest.json",
        "request_manifest": "request_manifest.json",
    }
    verified_files = {}
    for name, filename in required_files.items():
        verified_files[name] = _verify_published_file(
            standardized_dir / filename,
            name=name,
            integrity_info=locked_files.get(name),
            manifest_info=output_files.get(name),
        )
    integrity_lock_info = output_files.get("confirmation_integrity_lock")
    if not isinstance(integrity_lock_info, dict):
        raise ValueError("manifest does not identify confirmation_integrity_lock")
    actual_integrity_lock_sha256 = sha256_file(integrity_lock_path)
    actual_integrity_lock_size = integrity_lock_path.stat().st_size
    if actual_integrity_lock_sha256 != integrity_lock_info.get(
        "sha256"
    ) or actual_integrity_lock_size != integrity_lock_info.get("size_bytes"):
        raise ValueError("confirmation_integrity_lock SHA mismatch")
    verified_files["confirmation_integrity_lock"] = {
        "path": str(integrity_lock_path),
        "sha256": actual_integrity_lock_sha256,
        "size_bytes": actual_integrity_lock_size,
    }

    populations = [
        _load_population(
            "published_train", standardized_dir / "records_train.jsonl", "train"
        ),
        _load_population(
            "published_dev", standardized_dir / "records_dev.jsonl", "dev"
        ),
        _load_population(
            "published_confirmation",
            standardized_dir / "records_confirmation.jsonl",
            "confirmation",
        ),
    ]
    candidate_manifest = _read_json(standardized_dir / "candidate_manifest.json")
    request_manifest = _read_json(standardized_dir / "request_manifest.json")
    if candidate_manifest.get("dataset_version") != manifest.get(
        "dataset_version"
    ) or request_manifest.get("dataset_version") != manifest.get(
        "dataset_version"
    ):
        raise ValueError("published identity-manifest dataset version mismatch")
    _validate_identity_manifests(
        populations, candidate_manifest, request_manifest
    )

    qrels_audits = None
    sealed_label_audit = None
    if open_qrels:
        qrels_audits = {}
        for population in populations:
            name = f"qrels_{population.split}"
            path = standardized_dir / f"{name}.jsonl"
            verified_files[name] = _verify_published_file(
                path,
                name=name,
                integrity_info=locked_files.get(name),
                manifest_info=output_files.get(name),
            )
            qrels_audits[population.split] = _audit_qrels_for_population(
                path, population
            )
        sealed_path = standardized_dir / EVALUATOR_ONLY_LABEL_AUDIT_FILENAME
        sealed_file = _verify_published_file(
            sealed_path,
            name="evaluator_only_sealed_label_audit",
            integrity_info=sealed_label_audit_reference,
            manifest_info=sealed_label_audit_reference,
        )
        sealed_label_audit = _audit_opened_sealed_label_audit(
            sealed_path,
            expected_dataset_version=manifest["dataset_version"],
            expected_protocol_sha256=protocol["sha256"],
            expected_release_lock_sha256=release_lock["sha256"],
        )
        sealed_label_audit.update(
            {
                "file": sealed_file,
                "opened_only_after_qrels_integrity_verification": True,
            }
        )
        verified_files["evaluator_only_sealed_label_audit"] = sealed_file

    return {
        "schema_version": 1,
        "dataset_version": manifest["dataset_version"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "integrity_lock_sha256": actual_integrity_lock_sha256,
        "protocol_sha256": protocol["sha256"],
        "post_selection_recipe_checkpoint_lock_sha256": release_lock["sha256"],
        "post_selection_recipe_checkpoint_lock_path": str(
            recipe_checkpoint_lock_path
        ),
        "checkpoint_identities": release_lock["checkpoint_identities"],
        "analysis_selection_implementation": analysis_selection_summary,
        "history_assignment_release": history_assignment_release,
        "release_input_sha256": dict(
            release_lock["payload"]["input_sha256"]
        ),
        "verified_files": verified_files,
        "request_counts": {
            population.split: len(population.request_ids)
            for population in populations
        },
        "candidate_and_request_identity_verified": True,
        "qrels_opened": open_qrels,
        "qrels_audits": qrels_audits,
        "evaluator_only_sealed_label_audit_opened": open_qrels,
        "evaluator_only_sealed_label_audit": sealed_label_audit,
        "passed": True,
    }


def _assert_public_metadata_label_opaque(
    value: Any, *, source: str
) -> None:
    def visit(node: Any, location: str) -> None:
        if isinstance(node, Mapping):
            for raw_key, child in node.items():
                key = str(raw_key)
                if key in _SEALED_ONLY_LABEL_DERIVED_KEYS or key.endswith(
                    ("_with_click", "_with_purchase")
                ):
                    raise ValueError(
                        f"{source} exposes sealed label-derived field "
                        f"{location}.{key}"
                    )
                visit(child, f"{location}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                visit(child, f"{location}[{index}]")

    visit(value, source)


def _validate_sealed_label_audit_reference(
    value: Any, *, standardized_dir: Path
) -> dict[str, Any]:
    required_keys = {
        "path",
        "sha256",
        "size_bytes",
        "open_after_score_audit_only",
    }
    if not isinstance(value, dict) or set(value) != required_keys:
        raise ValueError(
            "evaluator-only sealed label-audit reference must contain exactly "
            "path, sha256, size_bytes, and open_after_score_audit_only"
        )
    path_value = value.get("path")
    if not isinstance(path_value, str) or not path_value.strip():
        raise ValueError("sealed label-audit reference path must be non-empty")
    declared_path = Path(path_value)
    expected_path = standardized_dir / EVALUATOR_ONLY_LABEL_AUDIT_FILENAME
    if declared_path.absolute() != expected_path.absolute():
        raise ValueError("sealed label-audit reference path is not canonical")
    size_bytes = value.get("size_bytes")
    if (
        isinstance(size_bytes, bool)
        or not isinstance(size_bytes, int)
        or size_bytes <= 0
    ):
        raise ValueError("sealed label-audit reference size_bytes is invalid")
    if value.get("open_after_score_audit_only") is not True:
        raise ValueError(
            "sealed label-audit must be marked open_after_score_audit_only"
        )
    return {
        "path": path_value,
        "sha256": _require_sha256(
            value.get("sha256"), "sealed label-audit reference sha256"
        ),
        "size_bytes": size_bytes,
        "open_after_score_audit_only": True,
    }


def _audit_opened_sealed_label_audit(
    path: Path,
    *,
    expected_dataset_version: str,
    expected_protocol_sha256: str,
    expected_release_lock_sha256: str,
) -> dict[str, Any]:
    payload = _read_json(path)
    expected_identity = {
        "schema_version": 1,
        "kind": "motivation_v1_2_evaluator_only_label_power_audit",
        "dataset_id": "kuaisearch",
        "dataset_version": expected_dataset_version,
        "protocol_sha256": expected_protocol_sha256,
        "post_selection_recipe_checkpoint_lock_sha256": (
            expected_release_lock_sha256
        ),
    }
    for field, expected in expected_identity.items():
        if payload.get(field) != expected:
            raise ValueError(
                "sealed label-audit identity mismatch for "
                f"{field}: {payload.get(field)!r} != {expected!r}"
            )
    expected_policy = {
        "evaluator_only": True,
        "open_after_score_audit_only": True,
        "model_and_scorer_access_forbidden": True,
    }
    if payload.get("access_policy") != expected_policy:
        raise ValueError("sealed label-audit access policy is invalid")
    required_audits = {
        "materializer_only_power_audit",
        "retrospective_training_leakage_audit",
        "source_writer_label_derived_counts",
    }
    for name in required_audits:
        if not isinstance(payload.get(name), dict):
            raise ValueError(f"sealed label-audit has no {name} object")
    return {
        "identity_verified": True,
        "access_policy_verified": True,
        "payload": payload,
    }


def _verify_published_file(
    path: Path,
    *,
    name: str,
    integrity_info: Any,
    manifest_info: Any,
) -> dict[str, Any]:
    if not isinstance(integrity_info, dict) or not isinstance(manifest_info, dict):
        raise ValueError(f"published holdout has no identity for {name}")
    if not path.is_file():
        raise FileNotFoundError(f"missing published holdout file: {path}")
    actual = {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
    for source, expected in (
        ("integrity lock", integrity_info),
        ("manifest", manifest_info),
    ):
        if expected.get("sha256") != actual["sha256"] or expected.get(
            "size_bytes"
        ) != actual["size_bytes"]:
            raise ValueError(f"{name} differs from published {source}")
    return {"path": str(path), **actual}


def _audit_qrels_for_population(
    path: Path, population: PopulationState
) -> dict[str, Any]:
    request_ids = set()
    positive_instances = 0
    for row in iter_jsonl(path):
        request_id = str(row.get("request_id") or "")
        if not request_id or request_id in request_ids:
            raise ValueError(f"invalid/duplicate qrels request_id in {path}")
        request_ids.add(request_id)
        visible = population.records.get(request_id)
        if visible is None:
            raise ValueError(f"qrels request is absent from records: {request_id}")
        candidate_ids = set(visible[1])
        clicked = {str(value) for value in row.get("clicked", [])}
        purchased = {str(value) for value in row.get("purchased", [])}
        relevance = row.get("relevance", {})
        if not isinstance(relevance, dict):
            raise ValueError(f"qrels relevance must be an object: {request_id}")
        relevance_ids = {str(value) for value in relevance}
        if not (clicked | purchased | relevance_ids) <= candidate_ids:
            raise ValueError(f"qrels contains an out-of-slate item: {request_id}")
        positive_instances += len(clicked | purchased | relevance_ids)
    if request_ids != set(population.request_ids):
        raise ValueError(f"qrels coverage differs from records: {path}")
    return {
        "path": str(path),
        "request_count": len(request_ids),
        "positive_candidate_instances": positive_instances,
        "coverage_verified": True,
    }


def _load_holdout_protocol(
    path: Path, *, enforce_registered_v12_recipe: bool
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing non-empty frozen protocol: {path}")
    protocol_text = path.read_bytes().decode("utf-8")
    payload = yaml.safe_load(protocol_text)
    if not isinstance(payload, dict):
        raise ValueError("protocol must contain a YAML/JSON object")
    status = str(payload.get("status") or "").casefold()
    frozen_status = status in {"frozen", "locked"} or status.endswith(
        ("_frozen", "_locked", "-frozen", "-locked")
    )
    if not frozen_status:
        raise ValueError(
            "protocol status must explicitly end in 'frozen' or 'locked'"
        )
    try:
        raw_rule = payload["data"]["new_holdout_rule"]
    except (KeyError, TypeError) as exc:
        raise ValueError("protocol has no data.new_holdout_rule mapping") from exc
    if not isinstance(raw_rule, dict):
        raise ValueError("data.new_holdout_rule must be a mapping")
    required_guards = {
        "materialize_only_after_recipe_lock": True,
        "source_split": "train",
        "source_test_opened": False,
        "exclude_all_prior_request_and_session_populations": True,
        "selection_uses_model_outputs": False,
    }
    for key, expected in required_guards.items():
        if raw_rule.get(key) != expected:
            raise ValueError(
                f"protocol holdout guard {key!r} must equal {expected!r}"
            )
    candidate_count = raw_rule.get("candidate_count")
    if (
        not isinstance(candidate_count, list)
        or len(candidate_count) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in candidate_count)
    ):
        raise ValueError("recipe candidate_count must be [min, max]")
    recipe = {
        "end_before_time": _positive_int(
            raw_rule.get("end_before_time_exclusive"),
            "end_before_time_exclusive",
        ),
        "source_window_requests": _positive_int(
            raw_rule.get("selected_source_window_requests"),
            "selected_source_window_requests",
        ),
        "confirmation_fraction": _fraction(
            raw_rule.get("confirmation_fraction"), "confirmation_fraction"
        ),
        "confirmation_requests": _positive_int(
            raw_rule.get("confirmation_requests"), "confirmation_requests"
        ),
        "max_history_len": _positive_int(
            raw_rule.get("max_history_len"), "max_history_len"
        ),
        "min_candidate_count": int(candidate_count[0]),
        "max_candidate_count": int(candidate_count[1]),
        "include_history_query": raw_rule.get("include_history_query") is True,
    }
    if raw_rule.get("include_history_query") is not True:
        raise ValueError("recipe include_history_query must be true")
    if recipe["min_candidate_count"] < 2 or (
        recipe["max_candidate_count"] < recipe["min_candidate_count"]
    ):
        raise ValueError("invalid recipe candidate-count boundary")
    expected_confirmation = (
        recipe["source_window_requests"] * recipe["confirmation_fraction"]
    )
    if not math.isclose(
        expected_confirmation,
        recipe["confirmation_requests"],
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "locked source window, fraction, and confirmation count disagree"
        )
    if enforce_registered_v12_recipe:
        expected_recipe = {
            "end_before_time": V12_END_BEFORE_TIME,
            "source_window_requests": V12_SOURCE_WINDOW_REQUESTS,
            "confirmation_fraction": V12_CONFIRMATION_FRACTION,
            "confirmation_requests": V12_CONFIRMATION_REQUESTS,
            "max_history_len": V12_MAX_HISTORY_LEN,
            "min_candidate_count": V12_MIN_CANDIDATE_COUNT,
            "max_candidate_count": V12_MAX_CANDIDATE_COUNT,
            "include_history_query": True,
        }
        if recipe != expected_recipe:
            raise ValueError(
                "protocol does not match the registered Motivation V1.2 "
                f"holdout rule: {recipe!r} != {expected_recipe!r}"
            )
    common_training = payload.get("common_training")
    if not isinstance(common_training, dict):
        raise ValueError("protocol common_training must be a mapping")
    checkpoint_selection_rule = common_training.get("checkpoint_selection")
    if (
        not isinstance(checkpoint_selection_rule, str)
        or not checkpoint_selection_rule.strip()
    ):
        raise ValueError(
            "protocol common_training.checkpoint_selection must be non-empty"
        )
    seed_policy = payload.get("seed_policy")
    if not isinstance(seed_policy, dict):
        raise ValueError("protocol seed_policy must be a mapping")
    pilot_seed = seed_policy.get("pilot_seed")
    if pilot_seed != V12_PILOT_SEED:
        raise ValueError(
            f"protocol pilot seed must equal the frozen value {V12_PILOT_SEED}"
        )
    return {
        "payload": payload,
        "recipe": recipe,
        "checkpoint_selection_rule": checkpoint_selection_rule,
        "pilot_seed": pilot_seed,
        "sha256": sha256_text(protocol_text),
    }


def _load_post_selection_lock(
    path: Path,
    *,
    protocol_sha256: str,
    checkpoint_selection_rule: str | None = None,
    pilot_seed: int = V12_PILOT_SEED,
) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"missing non-empty post-selection recipe/checkpoint lock: {path}"
        )
    if path.suffix.casefold() != ".json":
        raise ValueError("post-selection recipe/checkpoint lock must be JSON")
    lock_text = path.read_bytes().decode("utf-8")
    try:
        payload = json.loads(lock_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid post-selection lock JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("post-selection lock must contain a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("post-selection lock schema_version must equal 1")
    lock_id = payload.get("lock_id")
    if not isinstance(lock_id, str) or not lock_id.strip():
        raise ValueError("post-selection lock lock_id must be non-empty")
    status = str(payload.get("status") or "").casefold()
    final_status = status in {"frozen", "locked"} or status.endswith(
        ("_frozen", "_locked", "-frozen", "-locked")
    )
    if not final_status or "pre_pilot" in status or "pre-pilot" in status:
        raise ValueError(
            "post-selection lock status must be final and end in frozen/locked"
        )
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError("post-selection lock protocol SHA does not match protocol")

    release = payload.get("holdout_materialization")
    if not isinstance(release, dict):
        raise ValueError(
            "post-selection lock requires a holdout_materialization release"
        )
    required_release_guards = {
        "authorized": True,
        "methods_frozen": True,
        "witness_frozen": True,
        "configs_frozen": True,
        "checkpoint_selection_frozen": True,
        "analysis_and_selection_rules_frozen": True,
        "history_assignment_recipe_and_generator_frozen": True,
    }
    for key, expected in required_release_guards.items():
        if release.get(key) != expected:
            raise ValueError(
                f"holdout materialization release {key!r} must equal {expected!r}"
            )

    analysis_selection_implementation = (
        _validate_analysis_selection_implementation_identity(
            payload.get("analysis_selection_implementation"),
            verify_current_holdout_selection=True,
        )
    )
    history_assignment_release = _validate_history_assignment_release(
        payload.get("history_assignment_release"), pilot_seed=pilot_seed
    )

    config_shas = _validate_frozen_artifact_map(
        payload.get("frozen_configs"), "frozen_configs"
    )
    checkpoint_identities = _validate_checkpoint_selection_map(
        payload.get("frozen_checkpoints"),
        config_shas=config_shas,
        checkpoint_selection_rule=checkpoint_selection_rule,
        protocol_sha256=protocol_sha256,
        pilot_seed=pilot_seed,
    )
    return {
        "payload": payload,
        "release": dict(release),
        "checkpoint_identities": checkpoint_identities,
        "analysis_selection_implementation": (
            analysis_selection_implementation
        ),
        "history_assignment_release": history_assignment_release,
        "sha256": sha256_text(lock_text),
    }


def _validate_frozen_artifact_map(
    value: Any, name: str
) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != V12_METHOD_IDS:
        raise ValueError(
            f"{name} must cover exactly the fixed Q0-Q3 and W0 method IDs"
        )
    validated = {}
    for method_id, artifact in value.items():
        if not isinstance(artifact, dict):
            raise ValueError(f"{name}.{method_id} must be an object")
        path_value = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(f"{name}.{method_id}.path must be non-empty")
        _require_sha256(expected_sha, f"{name}.{method_id}.sha256")
        artifact_path = Path(path_value)
        if not artifact_path.is_file():
            raise FileNotFoundError(f"missing frozen artifact: {artifact_path}")
        if sha256_file(artifact_path) != expected_sha:
            raise ValueError(f"frozen artifact SHA mismatch: {name}.{method_id}")
        validated[method_id] = expected_sha
    return validated


def _validate_checkpoint_selection_map(
    value: Any,
    *,
    config_shas: Mapping[str, str],
    checkpoint_selection_rule: str | None,
    protocol_sha256: str,
    pilot_seed: int,
) -> dict[str, dict[str, Any]]:
    """Validate one frozen checkpoint-selection identity file per method."""

    if not isinstance(value, dict) or set(value) != V12_METHOD_IDS:
        raise ValueError(
            "frozen_checkpoints must cover exactly the fixed Q0-Q3 and W0 method IDs"
        )
    if (
        not isinstance(checkpoint_selection_rule, str)
        or not checkpoint_selection_rule.strip()
    ):
        raise ValueError(
            "checkpoint selection identities require the frozen protocol rule"
        )
    summaries = {}
    for method_id in sorted(value):
        artifact = value[method_id]
        if not isinstance(artifact, dict):
            raise ValueError(f"frozen_checkpoints.{method_id} must be an object")
        artifact_type = artifact.get("artifact_type")
        if artifact_type not in {None, "checkpoint_selection_identity_manifest"}:
            raise ValueError(
                f"frozen_checkpoints.{method_id}.artifact_type is invalid"
            )
        path_value = artifact.get("path")
        expected_sha = artifact.get("sha256")
        if not isinstance(path_value, str) or not path_value.strip():
            raise ValueError(
                f"frozen_checkpoints.{method_id}.path must be non-empty"
            )
        _require_sha256(
            expected_sha, f"frozen_checkpoints.{method_id}.sha256"
        )
        identity_path = Path(path_value)
        if identity_path.suffix.casefold() != ".json":
            raise ValueError(
                "frozen checkpoint selection identity must be a JSON file: "
                f"{method_id}"
            )
        if not identity_path.is_file():
            raise FileNotFoundError(
                f"missing frozen checkpoint selection identity: {identity_path}"
            )
        identity = _read_frozen_json(
            identity_path,
            expected_sha256=expected_sha,
            mismatch_message=(
                "frozen checkpoint selection identity SHA mismatch: "
                f"{method_id}"
            ),
        )
        expected_fields = {
            "schema_version": 1,
            "kind": "motivation_v1_2_checkpoint_selection_identity",
            "method_id": method_id,
            "selection_frozen": True,
            "selection_rule": checkpoint_selection_rule,
            "config_sha256": config_shas[method_id],
            "protocol_sha256": protocol_sha256,
            "seed": pilot_seed,
            "status": "completed",
            "evidence_mode": "first_round_pilot",
        }
        for field, expected in expected_fields.items():
            if identity.get(field) != expected:
                raise ValueError(
                    "checkpoint selection identity mismatch for "
                    f"{method_id}.{field}: {identity.get(field)!r} != {expected!r}"
                )
        checkpoint_reference = identity.get("checkpoint_reference")
        if (
            not isinstance(checkpoint_reference, str)
            or not checkpoint_reference.strip()
        ):
            raise ValueError(
                f"checkpoint selection identity {method_id} has no reference"
            )
        implementation_digest = _require_sha256(
            identity.get("implementation_digest"),
            f"checkpoint selection identity {method_id}.implementation_digest",
        )
        checkpoint_files, checkpoint_file_paths = _validate_checkpoint_files(
            method_id=method_id,
            identity_path=identity_path,
            value=identity.get("checkpoint_files"),
        )
        canonical_checkpoint_sha256 = _checkpoint_files_digest(checkpoint_files)
        if method_id == "w0_copps_style_transfer_witness":
            if len(checkpoint_files) != 1 or checkpoint_files[0]["name"] != "model.pt":
                raise ValueError(
                    "W0 checkpoint identity must contain only model.pt"
                )
            expected_checkpoint_sha256 = checkpoint_files[0]["sha256"]
        else:
            expected_checkpoint_sha256 = canonical_checkpoint_sha256
        declared_checkpoint_sha256 = _require_sha256(
            identity.get("checkpoint_sha256"),
            f"checkpoint selection identity {method_id}.checkpoint_sha256",
        )
        if declared_checkpoint_sha256 != expected_checkpoint_sha256:
            raise ValueError(
                "checkpoint selection identity actual artifact digest mismatch: "
                f"{method_id}"
            )
        checkpoint_id = identity.get("checkpoint_id")
        expected_checkpoint_id = (
            f"{method_id}@{expected_checkpoint_sha256[:20]}"
        )
        if checkpoint_id != expected_checkpoint_id:
            raise ValueError(
                "checkpoint selection identity checkpoint_id mismatch: "
                f"{method_id}"
            )

        training_metadata_path = _locked_file_path(
            identity_path,
            identity.get("training_metadata_path"),
            f"checkpoint selection identity {method_id}.training_metadata_path",
        )
        training_metadata_sha256 = _require_sha256(
            identity.get("training_metadata_sha256"),
            f"checkpoint selection identity {method_id}.training_metadata_sha256",
        )
        training_metadata = _read_frozen_json(
            training_metadata_path,
            expected_sha256=training_metadata_sha256,
            mismatch_message=(
                f"checkpoint training metadata SHA mismatch: {method_id}"
            ),
        )
        metadata_expected = {
            "method_id": method_id,
            "status": "completed",
            "protocol_sha256": protocol_sha256,
            "config_sha256": config_shas[method_id],
            "seed": pilot_seed,
            "evidence_mode": "first_round_pilot",
            "checkpoint_id": checkpoint_id,
        }
        for field, expected in metadata_expected.items():
            if training_metadata.get(field) != expected:
                raise ValueError(
                    "checkpoint training metadata mismatch for "
                    f"{method_id}.{field}: "
                    f"{training_metadata.get(field)!r} != {expected!r}"
                )
        metadata_implementation = training_metadata.get(
            "implementation_digest"
        )
        if metadata_implementation is None:
            implementation_identity = training_metadata.get(
                "implementation_identity"
            )
            if isinstance(implementation_identity, dict):
                metadata_implementation = implementation_identity.get("digest")
        if metadata_implementation is None:
            metadata_implementation = training_metadata.get(
                "implementation_sha256"
            )
        if metadata_implementation != implementation_digest:
            raise ValueError(
                "checkpoint training metadata implementation digest mismatch: "
                f"{method_id}"
            )
        if method_id == "w0_copps_style_transfer_witness":
            if training_metadata.get("model_sha256") != checkpoint_files[0][
                "sha256"
            ]:
                raise ValueError("W0 training metadata model SHA mismatch")
        else:
            metadata_files = _canonical_checkpoint_file_entries(
                training_metadata.get("checkpoint_weight_files"),
                source=f"checkpoint training metadata {method_id}",
            )
            if metadata_files != checkpoint_files:
                raise ValueError(
                    "checkpoint training metadata weight-file identity mismatch: "
                    f"{method_id}"
                )
        summaries[method_id] = {
            "identity_manifest_path": str(identity_path),
            "identity_manifest_sha256": expected_sha,
            "checkpoint_id": checkpoint_id,
            "checkpoint_sha256": declared_checkpoint_sha256,
            "checkpoint_files": [
                {**entry, "path": str(checkpoint_file_paths[entry["name"]])}
                for entry in checkpoint_files
            ],
            "config_sha256": config_shas[method_id],
            "training_metadata_path": str(training_metadata_path),
            "training_metadata_sha256": training_metadata_sha256,
            "implementation_digest": implementation_digest,
            "protocol_sha256": protocol_sha256,
            "seed": pilot_seed,
            "status": "completed",
            "evidence_mode": "first_round_pilot",
        }
    return summaries


def _validate_checkpoint_files(
    *, method_id: str, identity_path: Path, value: Any
) -> tuple[list[dict[str, Any]], dict[str, Path]]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"checkpoint selection identity {method_id} has no checkpoint_files"
        )
    canonical = []
    names = set()
    paths = {}
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(
                f"checkpoint_files[{index}] for {method_id} must be an object"
            )
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"checkpoint_files[{index}].name for {method_id} is empty"
            )
        logical_path = Path(name)
        if logical_path.is_absolute() or ".." in logical_path.parts:
            raise ValueError(
                f"checkpoint_files[{index}].name for {method_id} is unsafe"
            )
        if name in names:
            raise ValueError(f"duplicate checkpoint file name for {method_id}: {name}")
        names.add(name)
        artifact_path = _locked_file_path(
            identity_path,
            entry.get("path"),
            f"checkpoint_files[{index}].path for {method_id}",
        )
        paths[name] = artifact_path
        expected_sha256 = _require_sha256(
            entry.get("sha256"),
            f"checkpoint_files[{index}].sha256 for {method_id}",
        )
        expected_size = entry.get("size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError(
                f"checkpoint_files[{index}].size_bytes for {method_id} is invalid"
            )
        if artifact_path.stat().st_size != expected_size:
            raise ValueError(
                f"checkpoint file size mismatch for {method_id}: {name}"
            )
        if sha256_file(artifact_path) != expected_sha256:
            raise ValueError(
                f"checkpoint file SHA mismatch for {method_id}: {name}"
            )
        canonical.append(
            {
                "name": name,
                "sha256": expected_sha256,
                "size_bytes": expected_size,
            }
        )
    return sorted(canonical, key=lambda entry: entry["name"]), paths


def _canonical_checkpoint_file_entries(
    value: Any, *, source: str
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{source} has no checkpoint files")
    canonical = []
    for index, entry in enumerate(value):
        if not isinstance(entry, dict):
            raise ValueError(f"{source} checkpoint file {index} is not an object")
        name = entry.get("name")
        sha256 = entry.get("sha256")
        size_bytes = entry.get("size_bytes")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"{source} checkpoint file {index} has no name")
        _require_sha256(sha256, f"{source} checkpoint file {index}.sha256")
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            raise ValueError(f"{source} checkpoint file {index} has invalid size")
        canonical.append(
            {"name": name, "sha256": sha256, "size_bytes": size_bytes}
        )
    names = [entry["name"] for entry in canonical]
    if len(names) != len(set(names)):
        raise ValueError(f"{source} has duplicate checkpoint file names")
    return sorted(canonical, key=lambda entry: entry["name"])


def _checkpoint_files_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text(
        json.dumps(list(entries), sort_keys=True, separators=(",", ":"))
    )


def _locked_file_path(identity_path: Path, value: Any, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    path = Path(value)
    if not path.is_absolute():
        path = identity_path.parent / path
    if not path.is_file():
        raise FileNotFoundError(f"missing frozen file for {name}: {path}")
    return path


def _validate_release_input_shas(
    payload: Mapping[str, Any], paths: Mapping[str, Path]
) -> dict[str, dict[str, Any]]:
    frozen = payload.get("input_sha256")
    if not isinstance(frozen, dict):
        raise ValueError("post-selection lock requires input_sha256 mapping")
    missing = set(paths) - set(frozen)
    unexpected = set(frozen) - set(paths)
    if missing or unexpected:
        raise ValueError(
            "post-selection lock input SHA keys differ from the materializer "
            f"contract: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    files = {}
    for name, path in paths.items():
        expected = frozen[name]
        _require_sha256(expected, f"input_sha256.{name}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"post-selection input SHA mismatch: {name}")
        files[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": actual,
        }
    return files


def _revalidate_frozen_state(
    *,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    recipe_checkpoint_lock_path: Path,
    release_lock: Mapping[str, Any],
    frozen_input_paths: Mapping[str, Path],
    enforce_registered_v12_recipe: bool,
) -> None:
    """Rehash every frozen dependency immediately before atomic publication."""

    current_protocol = _load_holdout_protocol(
        protocol_path,
        enforce_registered_v12_recipe=enforce_registered_v12_recipe,
    )
    if current_protocol["sha256"] != protocol["sha256"]:
        raise ValueError("protocol changed during holdout materialization")
    current_release_lock = _load_post_selection_lock(
        recipe_checkpoint_lock_path,
        protocol_sha256=current_protocol["sha256"],
        checkpoint_selection_rule=current_protocol[
            "checkpoint_selection_rule"
        ],
        pilot_seed=current_protocol["pilot_seed"],
    )
    if current_release_lock["sha256"] != release_lock["sha256"]:
        raise ValueError(
            "post-selection lock changed during holdout materialization"
        )
    _validate_release_input_shas(
        current_release_lock["payload"], frozen_input_paths
    )


def _atomic_publish_directory(staged_dir: Path, output_dir: Path) -> None:
    """Publish one complete directory without exposing partial output."""

    if output_dir.exists():
        raise FileExistsError(
            f"holdout output appeared before atomic publication: {output_dir}"
        )
    staged_dir.rename(output_dir)


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _validate_development_lock(
    payload: Mapping[str, Any],
    paths: Mapping[str, Path],
    populations: Sequence[PopulationState],
    *,
    require_frozen_shas: bool,
) -> None:
    configured = payload.get("data", {}).get("development_population", {})
    if not isinstance(configured, dict):
        raise ValueError("data.development_population must be a mapping")
    count_expectations = {
        "train_requests": len(populations[0].request_ids),
        "internal_dev_requests": len(populations[1].request_ids),
        "legacy_compatibility_requests": len(populations[2].request_ids),
    }
    for key, actual in count_expectations.items():
        expected = configured.get(key)
        if require_frozen_shas and expected is None:
            raise ValueError(f"development population lock is missing {key}")
        if expected is not None and expected != actual:
            raise ValueError(
                f"development population lock mismatch for {key}: {actual} != {expected}"
            )
    sha_expectations = {
        "manifest_sha256": paths["manifest"],
        "candidate_manifest_sha256": paths["candidate_manifest"],
        "request_manifest_sha256": paths["request_manifest"],
        "records_train_sha256": paths["records_train"],
        "records_dev_sha256": paths["records_dev"],
    }
    for key, path in sha_expectations.items():
        expected = configured.get(key)
        if require_frozen_shas and expected is None:
            raise ValueError(f"development population lock is missing {key}")
        if expected is not None and sha256_file(path) != expected:
            raise ValueError(f"development population SHA lock mismatch for {key}")
    manifest = _read_json(paths["manifest"])
    expected_version = configured.get("dataset_version")
    if require_frozen_shas and expected_version is None:
        raise ValueError("development population lock is missing dataset_version")
    if expected_version is not None and manifest.get("dataset_version") != expected_version:
        raise ValueError("development population dataset version lock mismatch")
    legacy_sha = manifest.get("confirmation_lock", {}).get("v1_records_sha256")
    if not isinstance(legacy_sha, str):
        if require_frozen_shas:
            raise ValueError("development manifest has no frozen legacy records SHA")
    elif sha256_file(paths["records_confirmation"]) != legacy_sha:
        raise ValueError("legacy confirmation records differ from development manifest")


def _load_population(name: str, path: Path, split: str) -> PopulationState:
    audit_standardized_file(path, split)
    records: dict[str, tuple[str, tuple[str, ...]]] = {}
    sessions: set[str] = set()
    times: list[int] = []
    for row in iter_jsonl(path):
        request_id = str(row["request_id"])
        if request_id in records:
            raise ValueError(f"duplicate request_id in {path}: {request_id}")
        records[request_id] = (
            str(row["query"]),
            tuple(str(candidate["item_id"]) for candidate in row["candidates"]),
        )
        sessions.add(str(row["session_id"]))
        times.append(int(row["ts"]))
    return PopulationState(
        name=name,
        split=split,
        path=path,
        request_ids=frozenset(records),
        session_ids=frozenset(sessions),
        records=records,
        time_min=min(times),
        time_max=max(times),
    )


def _population_from_source_requests(
    name: str, path: Path, requests: Sequence[SourceRequest]
) -> PopulationState:
    records = {
        _request_id(request.key): (
            request.key[2],
            tuple(str(item_id) for item_id in request.candidate_item_ids),
        )
        for request in requests
    }
    if len(records) != len(requests):
        raise ValueError("duplicate request IDs in selected confirmation")
    return PopulationState(
        name=name,
        split="confirmation",
        path=path,
        request_ids=frozenset(records),
        session_ids=frozenset(str(request.key[1]) for request in requests),
        records=records,
        time_min=min(request.key[3] for request in requests),
        time_max=max(request.key[3] for request in requests),
    )


def _cross_population_overlap_audit(
    populations: Sequence[PopulationState],
) -> dict[str, Any]:
    population_summary = {
        population.name: {
            "path": str(population.path),
            "split": population.split,
            "requests": len(population.request_ids),
            "sessions": len(population.session_ids),
            "time_min": population.time_min,
            "time_max": population.time_max,
        }
        for population in populations
    }
    pairs = []
    for left, right in combinations(populations, 2):
        request_overlap = left.request_ids & right.request_ids
        session_overlap = left.session_ids & right.session_ids
        pair = {
            "left": left.name,
            "right": right.name,
            "request_overlap": len(request_overlap),
            "session_overlap": len(session_overlap),
            "request_overlap_examples": sorted(request_overlap)[:5],
            "session_overlap_examples": sorted(session_overlap)[:5],
        }
        pairs.append(pair)
        if request_overlap or session_overlap:
            raise ValueError(
                "cross-population identity overlap: "
                f"{left.name}/{right.name} requests={len(request_overlap)} "
                f"sessions={len(session_overlap)}"
            )
    return {
        "populations": population_summary,
        "pairwise": pairs,
        "all_request_overlaps_zero": True,
        "all_session_overlaps_zero": True,
    }


def _time_isolation_audit(
    populations: Sequence[PopulationState], *, end_before_time: int
) -> dict[str, Any]:
    by_name = {population.name: population for population in populations}
    order = (
        "new_confirmation",
        "development_train",
        "development_internal_dev",
        "legacy_confirmation",
        "subsequent_scout_train",
        "subsequent_scout_dev",
    )
    boundaries = []
    for left_name, right_name in zip(order, order[1:]):
        left = by_name[left_name]
        right = by_name[right_name]
        strict = left.time_max < right.time_min
        boundaries.append(
            {
                "left": left_name,
                "left_time_max": left.time_max,
                "right": right_name,
                "right_time_min": right.time_min,
                "strictly_before": strict,
            }
        )
        if not strict:
            raise ValueError(
                f"time isolation failed: {left_name} max={left.time_max} is not "
                f"before {right_name} min={right.time_min}"
            )
    cutoff_valid = (
        by_name["new_confirmation"].time_max < end_before_time
        <= by_name["development_train"].time_min
    )
    if not cutoff_valid:
        raise ValueError("locked end-before cutoff does not isolate the new holdout")
    return {
        "chronological_order": list(order),
        "direction": "retrospective_confirmation_before_training",
        "forward_temporal_holdout": False,
        "user_item_query_isolation_claimed": False,
        "boundaries": boundaries,
        "end_before_time_exclusive": end_before_time,
        "cutoff_isolates_new_confirmation_from_development": True,
        "all_boundaries_strict": True,
    }


def _retrospective_training_leakage_audit(
    confirmation_requests: Sequence[SourceRequest],
    train_records_path: Path,
) -> dict[str, Any]:
    """Count earlier holdout events visible in later standardized train history."""

    holdout_users = {str(request.key[0]) for request in confirmation_requests}
    event_to_requests: dict[tuple[str, str, int], set[str]] = {}
    for request in confirmation_requests:
        request_id = _request_id(request.key)
        user_id = str(request.key[0])
        request_time = int(request.key[3])
        for item_id in request.clicked_item_ids | request.purchased_item_ids:
            event_to_requests.setdefault(
                (user_id, str(item_id), request_time), set()
            ).add(request_id)

    train_requests_with_same_user = 0
    overlapping_users: set[str] = set()
    train_requests_with_holdout_event_in_history: set[str] = set()
    holdout_requests_with_event_in_train_history: set[str] = set()
    matching_history_event_instances = 0
    for record in iter_jsonl(train_records_path):
        user_id = str(record["user_id"])
        if user_id in holdout_users:
            train_requests_with_same_user += 1
            overlapping_users.add(user_id)
        for event in record.get("history", []):
            matched_requests = event_to_requests.get(
                (user_id, str(event["item_id"]), int(event["ts"]))
            )
            if not matched_requests:
                continue
            matching_history_event_instances += 1
            train_requests_with_holdout_event_in_history.add(
                str(record["request_id"])
            )
            holdout_requests_with_event_in_train_history.update(matched_requests)

    return {
        "audit_is_aggregate_only": True,
        "holdout_users": len(holdout_users),
        "overlapping_users": len(overlapping_users),
        "train_requests_with_same_user": train_requests_with_same_user,
        "holdout_positive_event_instances": sum(
            len(request_ids) for request_ids in event_to_requests.values()
        ),
        "matching_history_event_instances": matching_history_event_instances,
        "train_requests_with_holdout_event_in_history": len(
            train_requests_with_holdout_event_in_history
        ),
        "holdout_requests_with_event_in_train_history": len(
            holdout_requests_with_event_in_train_history
        ),
        "isolation_requirement": "reported_not_enforced_by_registered_rule",
        "interpretation": (
            "Nonzero counts mean later training inputs expose behavior from the "
            "earlier retrospective holdout; this population is not a forward "
            "temporal confirmation."
        ),
    }


def _partition_source_writer_counts(
    counts: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep target-label counts out of every public/model-side artifact."""

    label_suffixes = ("_with_click", "_with_purchase")
    label_derived = {
        str(name): value
        for name, value in counts.items()
        if str(name).endswith(label_suffixes)
    }
    label_opaque = {
        str(name): value
        for name, value in counts.items()
        if str(name) not in label_derived
    }
    return label_opaque, label_derived


def _aggregate_power_counts(requests: Iterable[SourceRequest]) -> dict[str, Any]:
    counts = {
        "requests": 0,
        "observed_positive": 0,
        "no_observed_positive": 0,
        "clicked_positive_requests": 0,
        "purchased_positive_requests": 0,
        "positive_candidate_instances": 0,
        "candidate_overlap": 0,
        "no_candidate_overlap_history_present": 0,
        "no_history": 0,
        "target_repeat": 0,
        "target_nonrepeat_other_candidate_overlap": 0,
        "target_nonrepeat_no_candidate_overlap": 0,
        "target_nonrepeat_no_history": 0,
    }
    for request in requests:
        counts["requests"] += 1
        candidates = set(request.candidate_item_ids)
        positives = set(request.clicked_item_ids | request.purchased_item_ids)
        unknown = positives - candidates
        if unknown:
            raise ValueError(
                "source positives are absent from the candidate slate for "
                f"request_id={_request_id(request.key)}: {sorted(unknown)[:5]}"
            )
        history = {event[1] for event in request.history}
        candidate_overlap = bool(history & candidates)
        counts["clicked_positive_requests"] += int(bool(request.clicked_item_ids))
        counts["purchased_positive_requests"] += int(bool(request.purchased_item_ids))
        counts["positive_candidate_instances"] += len(positives)
        if history:
            counts[
                "candidate_overlap"
                if candidate_overlap
                else "no_candidate_overlap_history_present"
            ] += 1
        else:
            counts["no_history"] += 1
        if not positives:
            counts["no_observed_positive"] += 1
        else:
            counts["observed_positive"] += 1
            if positives & history:
                counts["target_repeat"] += 1
            elif not history:
                counts["target_nonrepeat_no_history"] += 1
            elif candidate_overlap:
                counts["target_nonrepeat_other_candidate_overlap"] += 1
            else:
                counts["target_nonrepeat_no_candidate_overlap"] += 1

    observed_partition = sum(
        counts[name]
        for name in (
            "target_repeat",
            "target_nonrepeat_other_candidate_overlap",
            "target_nonrepeat_no_candidate_overlap",
            "target_nonrepeat_no_history",
        )
    )
    candidate_partition = sum(
        counts[name]
        for name in (
            "candidate_overlap",
            "no_candidate_overlap_history_present",
            "no_history",
        )
    )
    if observed_partition != counts["observed_positive"]:
        raise AssertionError("target-aware power counts do not partition positives")
    if candidate_partition != counts["requests"]:
        raise AssertionError("candidate-overlap power counts do not partition requests")
    counts["observed_positive_partition_verified"] = True
    counts["candidate_overlap_partition_verified"] = True
    return counts


def _validate_identity_manifests(
    populations: Sequence[PopulationState],
    candidate_manifest: Mapping[str, Any],
    request_manifest: Mapping[str, Any],
) -> None:
    expected = {
        (population.split, request_id): visible
        for population in populations
        for request_id, visible in population.records.items()
    }
    candidate_entries = _manifest_entry_map(candidate_manifest, "candidate")
    request_entries = _manifest_entry_map(request_manifest, "request")
    relevant_splits = {population.split for population in populations}
    candidate_entries = {
        key: value for key, value in candidate_entries.items() if key[0] in relevant_splits
    }
    request_entries = {
        key: value for key, value in request_entries.items() if key[0] in relevant_splits
    }
    if set(candidate_entries) != set(expected):
        raise ValueError("candidate manifest coverage differs from standardized records")
    if set(request_entries) != set(expected):
        raise ValueError("request manifest coverage differs from standardized records")
    for key, (query, candidate_ids) in expected.items():
        candidate_entry = candidate_entries[key]
        if tuple(str(value) for value in candidate_entry["candidate_item_ids"]) != candidate_ids:
            raise ValueError(f"candidate manifest identity mismatch for {key}")
        request_entry = request_entries[key]
        candidate_hash = sha256_text(
            json.dumps(list(candidate_ids), separators=(",", ":"))
        )
        if request_entry.get("candidate_item_ids_sha256") != candidate_hash:
            raise ValueError(f"request manifest candidate hash mismatch for {key}")
        if request_entry.get("query_sha256") != sha256_text(query):
            raise ValueError(f"request manifest query hash mismatch for {key}")


def _manifest_entry_map(
    manifest: Mapping[str, Any], kind: str
) -> dict[tuple[str, str], dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{kind} manifest has no entries list")
    result = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError(f"{kind} manifest entry is not an object")
        key = (str(entry.get("split")), str(entry.get("request_id")))
        if key in result:
            raise ValueError(f"duplicate {kind} manifest entry: {key}")
        result[key] = entry
    return result


def _entries_for_splits(
    manifest: Mapping[str, Any], splits: set[str]
) -> list[dict[str, Any]]:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ValueError("manifest has no entries list")
    return [dict(entry) for entry in entries if entry.get("split") in splits]


def _require_files(paths: Mapping[str, Path]) -> None:
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"missing non-empty {name}: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_frozen_json(
    path: Path, *, expected_sha256: str, mismatch_message: str
) -> dict[str, Any]:
    text = path.read_bytes().decode("utf-8")
    if sha256_text(text) != expected_sha256:
        raise ValueError(mismatch_message)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid frozen JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected frozen JSON object: {path}")
    return value


def _file_info(
    path: Path, *, reported_path: Path | None = None
) -> dict[str, Any]:
    return {
        "path": str(reported_path if reported_path is not None else path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"recipe {name} must be a positive integer")
    return value


def _fraction(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"recipe {name} must be numeric")
    value = float(value)
    if not 0.0 < value < 0.5:
        raise ValueError(f"recipe {name} must be in (0, 0.5)")
    return value
