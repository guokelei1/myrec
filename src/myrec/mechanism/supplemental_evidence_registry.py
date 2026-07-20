"""Audit the exhaustive Transformer supplemental-evidence inventory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from myrec.mechanism.deep_dive_evidence_topology import MODEL_IDS
from myrec.mechanism.deep_dive_report_contract import COMPONENT_IDS
from myrec.utils.hashing import sha256_file


REGISTRY_PATH = Path(
    "experiments/motivation/transformer_supplemental_evidence_registry.yaml"
)
REGISTRY_MANIFEST_PATH = Path(
    "experiments/motivation/transformer_supplemental_evidence_registry_manifest.yaml"
)
EXPECTED_SUPPLEMENT_IDS = (
    "d0_embedding_readout_geometry",
    "d1_activation_anisotropy",
    "d1_candidate_block_flow",
    "d1_candidate_residual_geometry",
    "d1_preference_subspace_geometry",
    "d1_query_causal_floor",
    "d2_rmsnorm_flow",
    "d3_attention_pattern_synthesis",
    "d3_full_null_position_shift_audit",
    "d3_qk_stage_geometry_v3",
    "d4_mlp_feature_formation_extension",
    "d5_rope_position_geometry",
    "d6_frozen_logit_lens",
    "d6_native_readout_diagnostics",
    "d7_objective_common_nullspace",
    "d7_q2_objective_family_shares",
    "d7_q2_parameter_update_geometry",
    "d7_q2_update_anisotropy",
    "d7_q3_lora_head_geometry",
    "component_state_reverse_necessity_v2",
    "component_functional_design_gate_synthesis",
)
EVIDENCE_LEVELS = {
    "qrels_blind_descriptive",
    "qrels_blind_measurement_confound_audit",
    "descriptive_derived_from_completed_qrels_evaluator",
    "descriptive_derived_from_registered_objective_family",
    "preregistered_qrels_blind_descriptive",
    "preregistered_reverse_causal_extension",
    "preregistered_design_gate_synthesis",
}
QRELS_POLICIES = {
    "absent_or_false",
    "false_required",
    "true_allowed_from_completed_parent_evaluator",
    "true_required_after_pre_qrels_integrity",
    "false_required_for_synthesis",
}


def audit_supplemental_evidence_registry(
    root: str | Path = ".",
    *,
    registry_path: str | Path = REGISTRY_PATH,
    registry_manifest_path: str | Path | None = REGISTRY_MANIFEST_PATH,
) -> dict[str, Any]:
    """Verify inventory bytes and report pending future supplements without effects."""

    root = Path(root).resolve()
    registry_file = _resolve(root, registry_path)
    registry = _load_yaml(registry_file)
    _validate_registry_schema(registry)
    manifest_identity = None
    if registry_manifest_path is not None:
        manifest_file = _resolve(root, registry_manifest_path)
        manifest = _load_yaml(manifest_file)
        _audit_registry_manifest(root, manifest, registry_file)
        manifest_identity = {
            "path": _display_path(root, manifest_file),
            "sha256": sha256_file(manifest_file),
            "manifest_id": manifest["manifest_id"],
        }
    rows = []
    failures = []
    for entry in registry["entries"]:
        path = _resolve(root, str(entry["path"]))
        row = {
            "evidence_id": entry["evidence_id"],
            "path": str(entry["path"]),
            "status_at_registry": entry["status_at_registry"],
            "evidence_level": entry["evidence_level"],
            "model_scope": list(entry["model_scope"]),
            "components": list(entry["components"]),
            "may_change_design_ranking": entry["may_change_design_ranking"],
        }
        if not path.is_file():
            if entry["status_at_registry"] == "completed":
                row["status"] = "failed_missing_frozen_output"
                failures.append(f"missing frozen supplement: {entry['evidence_id']}")
            else:
                row["status"] = "pending"
            rows.append(row)
            continue
        observed_sha = sha256_file(path)
        row["sha256"] = observed_sha
        frozen_sha = entry.get("frozen_sha256")
        if frozen_sha is not None and observed_sha != frozen_sha:
            row["status"] = "failed_frozen_hash_drift"
            failures.append(f"supplement hash drift: {entry['evidence_id']}")
            rows.append(row)
            continue
        try:
            payload = _load_json(path)
            _audit_payload_contract(payload, entry)
        except (ValueError, KeyError, TypeError) as exc:
            row["status"] = "failed_output_contract"
            row["failure"] = str(exc)
            failures.append(f"supplement contract drift: {entry['evidence_id']}")
            rows.append(row)
            continue
        row["status"] = "completed"
        row["analysis_type"] = payload["analysis_type"]
        row["command"] = _normalized_command(payload.get("command"))
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "schema_version": 1,
        "analysis_type": "transformer_supplemental_evidence_registry_audit",
        "status": "failed" if failures else ("pending" if counts.get("pending") else "completed"),
        "registry": {
            "path": _display_path(root, registry_file),
            "sha256": sha256_file(registry_file),
            "registry_id": registry["registry_id"],
        },
        "registry_manifest": manifest_identity,
        "entries": rows,
        "entry_count": len(rows),
        "status_counts": dict(sorted(counts.items())),
        "failures": failures,
        "design_ranking_entry": "component_functional_design_gate_synthesis",
        "completed_metrics_schema_opened": True,
        "effect_values_used_for_completion_or_selection": False,
        "scientific_effect_values_emitted": False,
        "qrels_files_opened_by_this_audit": False,
        "source_test_opened": False,
    }


def _validate_registry_schema(registry: Mapping[str, Any]) -> None:
    if (
        registry.get("schema_version") != 1
        or registry.get("registry_id")
        != "motivation_transformer_supplemental_evidence_v1"
        or registry.get("status") != "frozen_inventory_and_future_output_contract"
    ):
        raise ValueError("supplemental evidence registry header drift")
    scope = registry.get("scope")
    if not isinstance(scope, Mapping) or scope.get(
        "modifies_parent_deep_dive_families"
    ) is not False or scope.get(
        "completed_descriptive_entries_may_upgrade_confirmatory_claims"
    ) is not False or scope.get(
        "pending_entries_registered_before_their_outputs"
    ) is not True or scope.get(
        "exact_layer_index_is_architecture_evidence"
    ) is not False:
        raise ValueError("supplemental evidence registry scope drift")
    entries = registry.get("entries")
    if not isinstance(entries, list):
        raise ValueError("supplemental evidence entries missing")
    ids = [str(entry.get("evidence_id")) for entry in entries]
    if len(ids) != len(set(ids)) or set(ids) != set(EXPECTED_SUPPLEMENT_IDS):
        raise ValueError("supplemental evidence ID coverage drift")
    allowed_models = set(MODEL_IDS)
    allowed_components = set(COMPONENT_IDS)
    for entry in entries:
        required = {
            "evidence_id",
            "path",
            "expected_analysis_type",
            "status_at_registry",
            "evidence_level",
            "model_scope",
            "components",
            "qrels_policy",
            "may_change_design_ranking",
        }
        if not required.issubset(entry):
            raise ValueError(f"supplement entry fields missing: {entry.get('evidence_id')}")
        if entry["status_at_registry"] not in {"completed", "pending"}:
            raise ValueError("supplement registry status is invalid")
        if entry["status_at_registry"] == "completed" and not entry.get(
            "frozen_sha256"
        ):
            raise ValueError("completed supplement lacks frozen SHA")
        if entry["status_at_registry"] == "pending" and entry.get("frozen_sha256"):
            raise ValueError("pending supplement cannot freeze an outcome SHA")
        if entry["evidence_level"] not in EVIDENCE_LEVELS:
            raise ValueError("supplement evidence level is invalid")
        if entry["qrels_policy"] not in QRELS_POLICIES:
            raise ValueError("supplement qrels policy is invalid")
        if (
            not isinstance(entry["model_scope"], list)
            or not entry["model_scope"]
            or not set(entry["model_scope"]).issubset(allowed_models)
            or not isinstance(entry["components"], list)
            or not entry["components"]
            or not set(entry["components"]).issubset(allowed_components)
        ):
            raise ValueError("supplement model/component scope is invalid")
    design_entries = [
        entry["evidence_id"]
        for entry in entries
        if entry["may_change_design_ranking"] is True
    ]
    if design_entries != ["component_functional_design_gate_synthesis"]:
        raise ValueError("supplement design-ranking authority drift")


def _audit_registry_manifest(
    root: Path, manifest: Mapping[str, Any], registry_file: Path
) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("manifest_id")
        != "motivation_transformer_supplemental_evidence_registry_manifest_v1"
        or manifest.get("status")
        != "frozen_before_four_pending_supplement_outputs"
    ):
        raise ValueError("supplement registry manifest header drift")
    registry = manifest.get("registry", {})
    if (
        _resolve(root, str(registry.get("path") or "")).resolve()
        != registry_file.resolve()
        or registry.get("sha256") != sha256_file(registry_file)
        or registry.get("total_entries") != len(EXPECTED_SUPPLEMENT_IDS)
        or registry.get("completed_retrospective_inventory_entries") != 17
        or registry.get("pending_pre_output_entries") != 4
    ):
        raise ValueError("supplement registry manifest binding drift")
    parent_paths = {
        "comprehensive_report_plan_sha256": "experiments/motivation/transformer_comprehensive_report_plan.md",
        "deep_dive_plan_sha256": "experiments/motivation/transformer_deep_dive_plan.md",
        "deep_dive_manifest_sha256": "experiments/motivation/transformer_deep_dive_manifest.yaml",
        "component_necessity_plan_v2_sha256": "experiments/motivation/transformer_component_necessity_extension_plan_v2.md",
        "component_necessity_manifest_v2_sha256": "experiments/motivation/transformer_component_necessity_extension_manifest_v2.yaml",
    }
    parents = manifest.get("parent_contracts", {})
    if any(
        parents.get(key) != sha256_file(root / path)
        for key, path in parent_paths.items()
    ):
        raise ValueError("supplement registry parent binding drift")
    boundary = manifest.get("freeze_boundary", {})
    required = {
        "completed_entries_are_exhaustive_retrospective_inventory": True,
        "completed_descriptive_entries_cannot_upgrade_confirmatory_claims": True,
        "pending_entries_frozen_before_output": True,
        "pending_effect_values_read_before_freeze": False,
        "pending_qrels_read_before_freeze": False,
        "source_test_opened": False,
        "parent_families_modified": False,
        "exact_layer_index_is_architecture_evidence": False,
        "only_component_functional_design_gate_may_change_design_ranking": True,
    }
    if boundary != required:
        raise ValueError("supplement registry freeze boundary drift")


def _audit_payload_contract(
    payload: Mapping[str, Any], entry: Mapping[str, Any]
) -> None:
    if (
        payload.get("analysis_type") != entry["expected_analysis_type"]
        or payload.get("status") != "completed"
    ):
        raise ValueError("analysis type or completion status differs")
    _normalized_command(payload.get("command"))
    policy = entry["qrels_policy"]
    qrels_read = payload.get("qrels_read")
    if policy == "absent_or_false" and qrels_read not in (None, False):
        raise ValueError("qrels-blind descriptive supplement opened qrels")
    if policy == "false_required" and qrels_read is not False:
        raise ValueError("pending qrels-blind supplement lacks qrels_read=false")
    if policy == "true_required_after_pre_qrels_integrity" and (
        qrels_read is not True
        or payload.get("pre_qrels_audit_path") is None
        or payload.get("other_fold_qrels_opened") is not False
    ):
        raise ValueError("reverse-necessity qrels boundary differs")
    if policy == "false_required_for_synthesis" and payload.get(
        "qrels_read_by_this_synthesis"
    ) is not False:
        raise ValueError("design synthesis qrels boundary differs")
    if payload.get("source_test_opened") not in (None, False):
        raise ValueError("supplement opened source test")
    if entry["may_change_design_ranking"] is True:
        boundary = payload.get("claim_boundary")
        if (
            not isinstance(boundary, Mapping)
            or boundary.get("exact_layer_index_is_architecture_evidence") is not False
            or boundary.get("operator_necessity_authorized") is not False
            or boundary.get("single_model_support_may_change_global_architecture_ranking")
            is not False
        ):
            raise ValueError("design synthesis claim boundary differs")


def _normalized_command(value: Any) -> list[str]:
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    if (
        isinstance(value, list)
        and value
        and all(isinstance(item, str) and item.strip() for item in value)
    ):
        return [item.strip() for item in value]
    raise ValueError("supplement output lacks a reproducible command")


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML object: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _display_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)
