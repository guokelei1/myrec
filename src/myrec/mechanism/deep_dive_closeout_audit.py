"""Read-only completion and data-boundary audit for Transformer deep-dive."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_FROZEN_ASSETS = {
    "experiments/motivation/transformer_deep_dive_plan.md": (
        "07440f4ad82c841281aa09e061a3095f48d030015a3eee6e4da8322ea7e1a584"
    ),
    "experiments/motivation/transformer_deep_dive_manifest.yaml": (
        "76445ae3c43f6ab21a708f50cc64f1e81d04d0a8541884769a596d320251a758"
    ),
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "fixed_candidate_rows_v1/manifest.json": (
        "84cdf68a0fabefcab055806bb690adf96f2a36ad2921c2d10c5d0aae8310aa61"
    ),
    "artifacts/motivation_transformer_deep_dive/frozen_controls/"
    "content_neutral_v1/manifest.json": (
        "934cea39662585329fa5f8330f07b5a8decc0233d5a0ed610fb06cfb45dcbd24"
    ),
}

EXPECTED_DELIVERABLES = {
    "d1_representation": "runs/20260718_kuaisearch_mech_d1_region_synthesis_v2/metrics.json",
    "d2_q3_native_gate": "runs/20260718_kuaisearch_mech_d2_q3_native_gate_eval_v1/metrics.json",
    "d2_postblock": "runs/20260718_kuaisearch_mech_d2_postblock_synthesis_v1/metrics.json",
    "d2_selected_branches": "runs/20260718_kuaisearch_mech_d2_selected_branch_synthesis_v1/metrics.json",
    "d3_attention_edges": "runs/20260718_kuaisearch_mech_d3_attention_edges_eval_v1/metrics.json",
    "d3_attention_heads": "runs/20260718_kuaisearch_mech_d3_attention_heads_eval_v1/metrics.json",
    "d3_attention_groups": "runs/20260718_kuaisearch_mech_d3_attention_groups_eval_v1/metrics.json",
    "d4_mlp_groups": "runs/20260718_kuaisearch_mech_d4_mlp_groups_eval_v1/metrics.json",
    "d5_context": "runs/20260718_kuaisearch_mech_d5_context_eval_v1/metrics.json",
    "d5_rope": "runs/20260718_kuaisearch_mech_d5_rope_eval_v1/metrics.json",
    "d6_q2_native_readout": "runs/20260718_kuaisearch_mech_d6_q2_native_readout_eval_v1/metrics.json",
    "d6_q3_native_readout": "runs/20260718_kuaisearch_mech_d6_q3_native_readout_eval_v1/metrics.json",
    "d6_q0_trajectory": "runs/20260718_kuaisearch_mech_d6_q0_trajectory_eval_v1/metrics.json",
    "d6_q1_trajectory": "runs/20260718_kuaisearch_mech_d6_q1_trajectory_eval_v1/metrics.json",
    "d6_q0_q1_branches": "runs/20260718_kuaisearch_mech_d6_q0_q1_branch_eval_v1/metrics.json",
    "d6_q0_q1_readouts": "runs/20260718_kuaisearch_mech_d6_q0_q1_final_readout_eval_v1/metrics.json",
    "d7_q2_objective": "runs/20260718_kuaisearch_mech_d7_q2_objective_conflict_synthesis_v1/metrics.json",
    "d7_q3_lora_path": "runs/20260718_kuaisearch_mech_d7_q3_lora_path_v1/lora_path_analysis.json",
    "d7_optimizer_replay": "runs/20260718_kuaisearch_mech_d7_optimizer_replay_eval_v1/metrics.json",
}

EXPECTED_DELIVERABLE_CONTRACTS: dict[str, dict[str, Mapping[str, Any]]] = {
    "d1_representation": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d1_region_decoding_synthesis",
            "family.planned_size": 96,
            "family.observed_size": 96,
        },
        "lengths": {"cells": 96},
    },
    "d2_q3_native_gate": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d2_q3_native_position_gate",
            "multiple_testing.gate_family_size": 2,
            "multiple_testing.scope_family_size": 2,
        },
        "lengths": {"block_results": 2},
    },
    "d2_postblock": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d2_postblock_synthesis",
            "multiple_testing.all_layer_family_size_per_endpoint": 30,
            "multiple_testing.adjacent_family_size_per_endpoint": 28,
            "multiple_testing.planned_family_size_never_shrinks": True,
        },
        "nonempty": {
            "sources.q2_recranker_generalqwen.implementation_digest": True,
            "sources.q3_tallrec_generalqwen.implementation_digest": True,
        },
    },
    "d2_selected_branches": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d2_selected_branch_synthesis",
            "multiple_testing.planned_family_size_never_shrinks": True,
        },
        "lengths": {"families": 10, "rows": 192},
    },
    "d3_attention_edges": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d3_attention_edges",
            "multiple_testing.family_size": 36,
        },
        "lengths": {"family_rows": 36},
        "nonempty": {"implementation_digest": True},
    },
    "d3_attention_heads": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d3_attention_head_observation",
            "sample_rows": 512,
            "query_heads": 16,
            "kv_heads": 8,
            "gqa_heads_per_kv": 2,
            "blocks": [13, 20, 27],
            "descriptive_only": True,
        },
        "nonempty": {"implementation_digest": True},
    },
    "d3_attention_groups": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d3_attention_gqa_causal_localization",
            "blocks": [13, 20, 27],
            "gqa_groups": 8,
            "query_heads_per_group": 2,
            "sample_rows_per_bundle": 512,
            "confirmatory_family_membership": False,
        },
        "lengths": {"methods": 2},
        "nonempty": {"implementation_digest": True},
    },
    "d4_mlp_groups": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d4_mlp_groups",
            "blocks": [13, 20, 27],
            "groups": 16,
            "sample_rows": 512,
            "descriptive_only": True,
        },
        "lengths": {"methods": 2},
        "nonempty": {"implementation_digest": True},
    },
    "d5_context": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d5_contextual_controls",
            "multiple_testing.family_size": 8,
        },
        "lengths": {"family_rows": 8},
        "nonempty": {"implementation_digest": True},
    },
    "d5_rope": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d5_rope",
            "multiple_testing.family_size": 36,
            "position_support_gate.active_contrast": (
                "compression_minus_baseline"
            ),
            "position_support_gate.active_endpoint": "ndcg@10",
            "position_support_gate.active_ci95_equivalence_band": [
                -0.005,
                0.005,
            ],
            "position_support_gate.requires_compression_minus_expansion_bh_q_below_alpha_0p05": True,
            "position_support_gate.requires_all_fold0_fold1_same_nonzero_direction": True,
            "position_support_gate.active_contrast_is_confirmatory_family_member": False,
        },
        "lengths": {"family_rows": 36},
        "nonempty": {"implementation_digest": True},
    },
    "d6_q2_native_readout": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d6_q2_native_readout",
            "multiple_testing.family_size": 12,
        },
        "lengths": {"family_rows": 12},
    },
    "d6_q3_native_readout": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d6_q3_native_readout",
            "multiple_testing.family_size": 24,
        },
        "lengths": {"family_rows": 24},
        "nonempty": {"implementation_digest": True},
    },
    "d6_q0_trajectory": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d6_q0_all_layer_trajectory",
            "request_count": 8000,
            "candidate_count": 160753,
        },
        "lengths": {"hidden_state_indices": 29},
        "nonempty": {"implementation_digest": True},
    },
    "d6_q1_trajectory": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d6_q1_kv_trajectory",
            "request_count": 8000,
            "candidate_count": 160753,
        },
        "nonempty": {"implementation_digest": True},
    },
    "d6_q0_q1_branches": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d6_q0_q1_branch_extension",
            "blocks": [13, 20, 27],
            "multiple_testing.family_size": 96,
        },
        "lengths": {"methods": 2, "family_rows": 96},
        "nonempty": {
            "implementation_digests.q0_qwen3_reranker_06b": True,
            "implementation_digests.q1_instructrec_generalqwen": True,
        },
    },
    "d6_q0_q1_readouts": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d6_q0_q1_final_readout",
            "confirmatory_family_membership": False,
        },
        "nonempty": {"implementation_digest": True},
        "lengths": {
            "methods": 2,
            "nodes": 2,
            "comparisons": 2,
            "endpoints": 2,
            "rows": 16,
        },
    },
    "d7_q2_objective": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d7_q2_objective_conflict",
            "family.registered_size": 12,
            "family.observed_size": 12,
        },
        "lengths": {"family_rows": 12},
    },
    "d7_q3_lora_path": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d7_q3_lora_path",
            "mechanical_controls.base_b_exact_zero": True,
            "mechanical_controls.orthogonal_gauge_identity_passed": True,
        },
        "lengths": {"states": 3, "comparisons": 168},
    },
    "d7_optimizer_replay": {
        "equals": {
            "analysis_type": "transformer_deep_dive_d7_exact_optimizer_replay",
            "step": 501,
            "replay_blocks_per_control_surface": 6,
            "requests_per_block": 16,
            "dev_confirmation_test_qrels_read": False,
            "source_test_opened": False,
        },
        "lengths": {"controls": 2, "surfaces": 3, "q2.objectives": 3, "q3.coordinate_modes": 3},
        "nonempty": {
            "implementation_digests.q2": True,
            "implementation_digests.q3": True,
        },
    },
}

IN_FLIGHT = {"initializing", "selection_finalized", "running", "wall_time_exhausted"}
BAD_TERMINAL = {"failed", "error", "aborted", "cancelled"}
PROHIBITED_VALUE_TOKENS = (
    "records_test",
    "qrels_test",
    "records_confirmation",
    "qrels_confirmation",
    "source_test.json",
    "data/raw/",
)
RUN_ARTIFACT_HASH_FIELDS = {
    "scores_sha256": "scores.jsonl",
    "rows_sha256": "rows.jsonl",
    "observations_sha256": "observations.jsonl",
    "groups_sha256": "groups.jsonl",
    "replays_sha256": "replays.jsonl",
    "index_sha256": "index.json",
}
FAILURE_PRESERVED_HASH_FIELDS = {
    "metadata_sha256": "metadata.json",
    "progress_sha256": "progress.json",
    "rows_partial_sha256": "rows.partial.jsonl",
    "observations_partial_sha256": "observations.partial.jsonl",
    "groups_partial_sha256": "groups.partial.jsonl",
    "scores_partial_sha256": "scores.partial.jsonl",
}


def audit_deep_dive_closeout(
    root: str | Path,
    *,
    expected_deliverables: Mapping[str, str] = EXPECTED_DELIVERABLES,
    frozen_assets: Mapping[str, str] = EXPECTED_FROZEN_ASSETS,
    deliverable_contracts: Mapping[str, Mapping[str, Mapping[str, Any]]] = (
        EXPECTED_DELIVERABLE_CONTRACTS
    ),
    dev_eval_log: str = "reports/dev_eval_log.jsonl",
) -> dict[str, Any]:
    """Audit immutable inputs, required outputs and every deep-dive declaration."""

    root = Path(root)
    failures: list[str] = []
    pending: list[str] = []
    ledger_path = root / dev_eval_log
    ledger_entries: list[dict[str, Any]] = []
    if ledger_path.is_file():
        with ledger_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    ledger_entries.append(_decode_json(line))
                except Exception as exc:
                    failures.append(
                        f"invalid dev-eval ledger JSON {dev_eval_log}:{line_number}: {exc}"
                    )

    assets = {}
    for relative, expected_sha in frozen_assets.items():
        path = root / relative
        if not path.is_file():
            failures.append(f"missing frozen asset: {relative}")
            continue
        observed = _sha256_file(path)
        assets[relative] = observed
        if observed != expected_sha:
            failures.append(f"frozen asset hash drift: {relative}")

    deliverables = {}
    for name, relative in expected_deliverables.items():
        path = root / relative
        if not path.is_file():
            pending.append(f"missing deliverable: {name}")
            deliverables[name] = {"path": relative, "status": "pending"}
            continue
        try:
            value = _load_json(path)
        except Exception as exc:
            failures.append(f"invalid deliverable JSON {name}: {exc}")
            continue
        status = str(value.get("status", "completed"))
        if status != "completed":
            if status in IN_FLIGHT:
                pending.append(f"in-flight deliverable: {name} ({status})")
            else:
                failures.append(f"deliverable terminal status: {name} ({status})")
        failures.extend(_declaration_failures(value, relative))
        contract = deliverable_contracts.get(name)
        if contract is not None:
            failures.extend(_contract_failures(value, name, contract))
        observed_sha = _sha256_file(path)
        if value.get("qrels_read") is True:
            run_id = str(
                value.get("analysis_run_id")
                or value.get("run_id")
                or path.parent.name
            )
            failures.extend(
                _ledger_failures(
                    ledger_entries,
                    run_id=run_id,
                    metrics_path=relative,
                    metrics_sha256=observed_sha,
                    ledger_path=dev_eval_log,
                )
            )
        deliverables[name] = {
            "path": relative,
            "status": status,
            "sha256": observed_sha,
        }

    run_declarations = []
    mechanical_terminal_runs: list[tuple[str, str]] = []
    inflight_formal_runs: list[tuple[str, str, str]] = []
    formal_completed_integrity_checked = 0
    runs_root = root / "runs"
    if runs_root.is_dir():
        for path in sorted(runs_root.glob("20260718_kuaisearch_mech_d*/metadata.json")):
            relative = path.relative_to(root).as_posix()
            try:
                value = _load_json(path)
            except Exception as exc:
                failures.append(f"invalid run metadata {relative}: {exc}")
                continue
            failures.extend(_declaration_failures(value, relative))
            status = str(value.get("status", "missing"))
            if status == "mechanical_failure" and value.get("result_eligible") is True:
                mechanical_terminal_runs.append(
                    (str(value.get("run_id") or path.parent.name), relative)
                )
            if status in IN_FLIGHT and value.get("result_eligible") is True:
                inflight_formal_runs.append(
                    (str(value.get("run_id") or path.parent.name), relative, status)
                )
            if status in BAD_TERMINAL and value.get("result_eligible") is True:
                failures.append(f"formal run terminal failure: {relative} ({status})")
            if status == "completed" and value.get("result_eligible") is True:
                formal_completed_integrity_checked += 1
                failures.extend(
                    _formal_run_integrity_failures(value, path, relative)
                )
            run_declarations.append(
                {
                    "path": relative,
                    "run_id": value.get("run_id") or path.parent.name,
                    "analysis_stage": value.get("analysis_stage"),
                    "method_id": value.get("method_id"),
                    "status": status,
                    "result_eligible": value.get("result_eligible"),
                    "command": value.get("command"),
                    "sha256": _sha256_file(path),
                }
            )

    mechanical_failure_records = []
    valid_bound_failure_run_ids: set[str] = set()
    if runs_root.is_dir():
        for path in sorted(
            runs_root.glob(
                "20260718_kuaisearch_mech_d*/mechanical_failure_record.json"
            )
        ):
            relative = path.relative_to(root).as_posix()
            try:
                value = _load_json(path)
            except Exception as exc:
                failures.append(f"invalid mechanical failure record {relative}: {exc}")
                continue
            record_failures = _declaration_failures(value, relative)
            record_failures.extend(
                _mechanical_failure_record_failures(value, path, relative)
            )
            failures.extend(record_failures)
            if not record_failures:
                valid_bound_failure_run_ids.add(str(value.get("run_id") or ""))
            mechanical_failure_records.append(
                {
                    "path": relative,
                    "run_id": value.get("run_id"),
                    "status": value.get("status"),
                    "sha256": _sha256_file(path),
                }
            )

    bound_failure_run_ids = {
        str(record.get("run_id") or "") for record in mechanical_failure_records
    }
    for run_id, relative in mechanical_terminal_runs:
        if run_id not in bound_failure_run_ids:
            failures.append(
                f"mechanical terminal run lacks bound failure record: {relative}"
            )

    # A valid record deliberately preserves the original metadata/progress/
    # partial bytes. Those bytes can therefore still say ``running`` even
    # though the attempted formal run was mechanically terminated and replaced.
    # Treat only a hash-bound, schema-valid record as the effective terminal
    # declaration; every other in-flight formal run remains fail-closed pending.
    for run_id, relative, raw_status in inflight_formal_runs:
        if run_id not in valid_bound_failure_run_ids:
            pending.append(f"in-flight formal run: {relative} ({raw_status})")
    for declaration in run_declarations:
        run_id = str(declaration.get("run_id") or "")
        if (
            run_id in valid_bound_failure_run_ids
            and declaration.get("status") in IN_FLIGHT
        ):
            declaration["metadata_status"] = declaration["status"]
            declaration["status"] = "mechanical_failure"

    status = "failed" if failures else "pending" if pending else "completed"
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_closeout_boundary_audit",
        "status": status,
        "source_test_content_read_by_this_audit": False,
        "qrels_content_read_by_this_audit": False,
        "frozen_assets": assets,
        "dev_eval_ledger": {
            "path": dev_eval_log,
            "exists": ledger_path.is_file(),
            "entry_count": len(ledger_entries),
            "sha256": _sha256_file(ledger_path) if ledger_path.is_file() else None,
        },
        "deliverables": deliverables,
        "run_declarations": run_declarations,
        "mechanical_failure_records": mechanical_failure_records,
        "formal_completed_integrity_checked": formal_completed_integrity_checked,
        "failures": failures,
        "pending": pending,
    }


def _formal_run_integrity_failures(
    metadata: Mapping[str, Any], metadata_path: Path, label: str
) -> list[str]:
    """Recheck implementation lineage and declared published bytes."""

    failures = []
    identity = metadata.get("implementation_identity")
    contract = metadata.get("run_contract")
    declared_contract_sha = metadata.get("run_contract_sha256")
    lineage_declared = any(
        value is not None for value in (identity, contract, declared_contract_sha)
    )
    if lineage_declared:
        digest = str(identity.get("digest") or "") if isinstance(identity, Mapping) else ""
        if not digest:
            failures.append(f"formal run implementation digest missing: {label}")
        if not isinstance(contract, Mapping):
            failures.append(f"formal run contract missing: {label}")
        else:
            if contract.get("implementation_digest") != digest:
                failures.append(
                    f"formal run implementation/contract binding mismatch: {label}"
                )
            observed_contract_sha = hashlib.sha256(
                json.dumps(
                    contract,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if declared_contract_sha != observed_contract_sha:
                failures.append(f"formal run contract hash mismatch: {label}")

    for field, filename in RUN_ARTIFACT_HASH_FIELDS.items():
        expected = metadata.get(field)
        if expected is None:
            continue
        artifact_path = metadata_path.parent / filename
        if not artifact_path.is_file():
            failures.append(f"formal run artifact missing: {label}:{filename}")
        elif expected != _sha256_file(artifact_path):
            failures.append(f"formal run artifact hash mismatch: {label}:{filename}")
    return failures


def _mechanical_failure_record_failures(
    value: Mapping[str, Any], path: Path, label: str
) -> list[str]:
    failures = []
    expected = {
        "analysis_type": "transformer_deep_dive_mechanical_failure_record",
        "status": "mechanical_failure",
        "result_eligible": False,
        "scientific_effect_values_read": False,
        "qrels_read": False,
        "source_test_opened": False,
    }
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            failures.append(f"mechanical failure declaration mismatch: {label}:{key}")
    if value.get("run_id") != path.parent.name:
        failures.append(f"mechanical failure run-id/path mismatch: {label}")
    preserved = value.get("preserved_inputs")
    if not isinstance(preserved, Mapping) or not preserved:
        failures.append(f"mechanical failure preserved inputs missing: {label}")
        return failures
    for field, expected_sha in preserved.items():
        filename = FAILURE_PRESERVED_HASH_FIELDS.get(str(field))
        if filename is None:
            failures.append(f"mechanical failure unknown preserved hash: {label}:{field}")
            continue
        target = path.parent / filename
        if not target.is_file():
            failures.append(f"mechanical failure preserved file missing: {label}:{filename}")
        elif expected_sha != _sha256_file(target):
            failures.append(f"mechanical failure preserved hash mismatch: {label}:{filename}")
    return failures


def _declaration_failures(value: Any, label: str) -> list[str]:
    failures = []
    for keys, child in _walk(value):
        key = keys[-1] if keys else ""
        if key == "source_test_opened" and child is not False:
            failures.append(f"source_test_opened is not false: {label}:{'.'.join(keys)}")
        if key == "dev_confirmation_test_qrels_read" and child is not False:
            failures.append(
                f"dev/test qrels declaration is not false: {label}:{'.'.join(keys)}"
            )
        if isinstance(child, str):
            lowered = child.casefold()
            if any(token in lowered for token in PROHIBITED_VALUE_TOKENS):
                failures.append(
                    f"prohibited data declaration: {label}:{'.'.join(keys)}={child}"
                )
        if isinstance(child, float) and not math.isfinite(child):
            failures.append(f"non-finite declaration: {label}:{'.'.join(keys)}")
    return failures


def _contract_failures(
    value: Mapping[str, Any],
    name: str,
    contract: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    failures = []
    for path, expected in contract.get("equals", {}).items():
        try:
            observed = _nested_value(value, path)
        except (KeyError, TypeError):
            failures.append(f"deliverable contract missing field: {name}:{path}")
            continue
        if observed != expected:
            failures.append(
                f"deliverable contract value mismatch: {name}:{path} "
                f"(expected {expected!r}, observed {observed!r})"
            )
    for path, expected in contract.get("lengths", {}).items():
        try:
            observed = _nested_value(value, path)
        except (KeyError, TypeError):
            failures.append(f"deliverable contract missing field: {name}:{path}")
            continue
        try:
            observed_length = len(observed)
        except TypeError:
            failures.append(f"deliverable contract field has no length: {name}:{path}")
            continue
        if observed_length != expected:
            failures.append(
                f"deliverable contract length mismatch: {name}:{path} "
                f"(expected {expected}, observed {observed_length})"
            )
    for path in contract.get("nonempty", {}):
        try:
            observed = _nested_value(value, path)
        except (KeyError, TypeError):
            failures.append(f"deliverable contract missing field: {name}:{path}")
            continue
        if observed is None or observed == "" or observed == [] or observed == {}:
            failures.append(f"deliverable contract empty field: {name}:{path}")
    return failures


def _nested_value(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping):
            raise TypeError(path)
        current = current[part]
    return current


def _walk(value: Any, keys: tuple[str, ...] = ()):
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = (*keys, str(key))
            yield path, child
            yield from _walk(child, path)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            path = (*keys, f"[{index}]")
            yield path, child
            yield from _walk(child, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return _decode_json(handle.read())


def _decode_json(raw: str) -> dict[str, Any]:
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"non-finite JSON constant: {value}")

    value = json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=constant,
    )
    if not isinstance(value, dict):
        raise ValueError("top-level JSON value is not an object")
    return value


def _ledger_failures(
    entries: Sequence[Mapping[str, Any]],
    *,
    run_id: str,
    metrics_path: str,
    metrics_sha256: str,
    ledger_path: str,
) -> list[str]:
    matched = [entry for entry in entries if entry.get("run_id") == run_id]
    if not matched:
        return [f"missing dev-eval ledger entry: {run_id} ({ledger_path})"]
    if len(matched) != 1:
        return [f"non-unique dev-eval ledger entry: {run_id} ({len(matched)})"]
    entry = matched[0]
    failures = []
    if entry.get("metrics_path") != metrics_path:
        failures.append(f"dev-eval ledger metrics path mismatch: {run_id}")
    if entry.get("metrics_sha256") != metrics_sha256:
        failures.append(f"dev-eval ledger metrics hash mismatch: {run_id}")
    return failures


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
