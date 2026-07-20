"""Qrels-blind mechanical progress census for the D2 causal core."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CORE_MODELS = (
    "q2_recranker_generalqwen",
    "q3_tallrec_generalqwen",
)
CORE_BLOCKS = tuple(range(13, 28))
CORE_FOLDS = (0, 1)
FIXED_SCIENTIFIC_BUNDLES = len(CORE_MODELS) * len(CORE_BLOCKS) * len(CORE_FOLDS)
MAX_CONDITIONAL_BRANCH_BUNDLES = len(CORE_MODELS)
MAX_SCIENTIFIC_BUNDLES = FIXED_SCIENTIFIC_BUNDLES + MAX_CONDITIONAL_BRANCH_BUNDLES
IN_FLIGHT = {"initializing", "running", "wall_time_exhausted"}
SELECTED_NODES = (
    "block_input_residual",
    "input_rmsnorm_output",
    "attention_o_projection",
    "post_attention_residual",
    "post_attention_rmsnorm_output",
    "mlp_down_projection",
    "block_output_residual",
)


def audit_deep_dive_progress(root: str | Path) -> dict[str, Any]:
    """Count registered scientific bundles without opening scores or qrels."""

    root_path = Path(root).resolve()
    fixed_rows = []
    errors = []
    for method_id in CORE_MODELS:
        short = method_id.split("_", 1)[0]
        for fold in CORE_FOLDS:
            for block in CORE_BLOCKS:
                run_id = (
                    f"20260718_kuaisearch_mech_d2_{short}_postblock_"
                    f"b{block}_fold{fold}_v1"
                )
                path = root_path / "runs" / run_id / "metadata.json"
                status = "missing"
                if path.is_file():
                    try:
                        value = _read_json(path)
                        status = str(value.get("status") or "missing")
                        if value.get("run_id") != run_id:
                            errors.append(f"run-id mismatch: {path.relative_to(root_path)}")
                        expected_identity = {
                            "method_id": method_id,
                            "block_zero_based": block,
                            "normalized_query_fold": fold,
                            "evidence_mode": "registered_mechanism_diagnostic",
                            "qrels_read": False,
                            "source_test_opened": False,
                        }
                        for key, expected in expected_identity.items():
                            if value.get(key) != expected:
                                errors.append(
                                    f"fixed bundle {key} mismatch: "
                                    f"{path.relative_to(root_path)}"
                                )
                        if value.get("result_eligible") is not True:
                            errors.append(
                                f"fixed bundle is not result-eligible: {path.relative_to(root_path)}"
                            )
                    except (OSError, json.JSONDecodeError, ValueError) as exc:
                        status = "invalid"
                        errors.append(
                            f"invalid metadata: {path.relative_to(root_path)} ({exc})"
                        )
                fixed_rows.append(
                    {
                        "method_id": method_id,
                        "block_zero_based": block,
                        "fold": fold,
                        "run_id": run_id,
                        "status": status,
                    }
                )

    branch_rows = [
        _branch_progress(root_path, method_id, errors) for method_id in CORE_MODELS
    ]
    in_flight_progress = [
        _in_flight_progress(root_path, row, errors)
        for row in fixed_rows
        if row["status"] in IN_FLIGHT
    ]
    fixed_completed = sum(row["status"] == "completed" for row in fixed_rows)
    in_flight_bundle_equivalents = sum(
        float(row["request_completion_fraction"])
        for row in in_flight_progress
        if row["request_completion_fraction"] is not None
    )
    fixed_executed_bundle_equivalents = (
        fixed_completed + in_flight_bundle_equivalents
    )
    branch_completed = sum(row["status"] == "completed" for row in branch_rows)
    branch_gate_stopped = sum(row["status"] == "gate_stopped" for row in branch_rows)
    branch_in_flight_bundle_equivalents = sum(
        float(row["request_completion_fraction"])
        for row in branch_rows
        if row["status"] == "pending_branch"
        and row["request_completion_fraction"] is not None
    )
    branch_executed_bundle_equivalents = sum(
        float(row["request_completion_fraction"])
        for row in branch_rows
        if row["request_completion_fraction"] is not None
    )
    completed = fixed_completed + branch_completed
    resolved = completed + branch_gate_stopped
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_d2_mechanical_progress",
        "status": "failed" if errors else "ok",
        "scientific_effect_values_read": False,
        "qrels_read": False,
        "source_test_opened": False,
        "fixed": {
            "registered_bundles": FIXED_SCIENTIFIC_BUNDLES,
            "completed_bundles": fixed_completed,
            "remaining_mandatory_bundles": FIXED_SCIENTIFIC_BUNDLES
            - fixed_completed,
            "in_flight_bundles": sum(
                row["status"] in IN_FLIGHT for row in fixed_rows
            ),
            "in_flight_bundle_equivalents": in_flight_bundle_equivalents,
            "request_weighted_executed_bundle_equivalents": (
                fixed_executed_bundle_equivalents
            ),
            "request_weighted_execution_fraction": (
                fixed_executed_bundle_equivalents / FIXED_SCIENTIFIC_BUNDLES
            ),
            "request_weighted_remaining_bundle_equivalents": (
                FIXED_SCIENTIFIC_BUNDLES - fixed_executed_bundle_equivalents
            ),
            "in_flight_progress": in_flight_progress,
            "status_counts": _status_counts(fixed_rows),
        },
        "conditional_selected_branches": {
            "maximum_registered_bundles": MAX_CONDITIONAL_BRANCH_BUNDLES,
            "completed_bundles": branch_completed,
            "gate_stopped_models": branch_gate_stopped,
            "in_flight_bundle_equivalents": (
                branch_in_flight_bundle_equivalents
            ),
            "request_weighted_executed_bundle_equivalents": (
                branch_executed_bundle_equivalents
            ),
            "request_weighted_execution_fraction": (
                branch_executed_bundle_equivalents
                / MAX_CONDITIONAL_BRANCH_BUNDLES
            ),
            "maximum_remaining_bundles": MAX_CONDITIONAL_BRANCH_BUNDLES
            - branch_completed
            - branch_gate_stopped,
            "models": branch_rows,
        },
        "maximum_total_scientific_bundles": MAX_SCIENTIFIC_BUNDLES,
        "completed_scientific_bundles": completed,
        "resolved_scientific_units": resolved,
        "resolution_fraction": resolved / MAX_SCIENTIFIC_BUNDLES,
        "maximum_remaining_scientific_bundles": (
            FIXED_SCIENTIFIC_BUNDLES
            - fixed_completed
            + MAX_CONDITIONAL_BRANCH_BUNDLES
            - branch_completed
            - branch_gate_stopped
        ),
        "maximum_completion_fraction": completed / MAX_SCIENTIFIC_BUNDLES,
        "maximum_request_weighted_executed_bundle_equivalents": (
            fixed_executed_bundle_equivalents
            + branch_executed_bundle_equivalents
        ),
        "maximum_request_weighted_execution_fraction": (
            (
                fixed_executed_bundle_equivalents
                + branch_executed_bundle_equivalents
            )
            / MAX_SCIENTIFIC_BUNDLES
        ),
        "errors": errors,
    }


def _branch_progress(
    root: Path, method_id: str, errors: list[str]
) -> dict[str, Any]:
    short = method_id.split("_", 1)[0]
    contract_path = (
        root
        / "runs"
        / f"20260718_kuaisearch_mech_d2_{short}_selected_branch_contract_v1"
        / "contract.json"
    )
    eval_path = (
        root
        / "runs"
        / f"20260718_kuaisearch_mech_d2_{short}_selected_branch_eval_v1"
        / "metrics.json"
    )
    status = "pending_contract"
    eligible = None
    if contract_path.is_file():
        try:
            contract = _read_json(contract_path)
            if contract.get("status") != "completed":
                status = str(contract.get("status") or "invalid_contract")
            elif type(contract.get("branch_scoring_eligible")) is not bool:
                status = "invalid_contract"
                errors.append(
                    f"selected contract eligibility invalid: {contract_path.relative_to(root)}"
                )
            else:
                eligible = contract["branch_scoring_eligible"]
                contract_errors = _selected_contract_errors(contract, method_id)
                if contract_errors:
                    status = "invalid_contract"
                    errors.extend(
                        f"selected contract {message}: {contract_path.relative_to(root)}"
                        for message in contract_errors
                    )
                else:
                    status = "pending_branch" if eligible else "gate_stopped"
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            status = "invalid_contract"
            errors.append(
                f"invalid selected contract: {contract_path.relative_to(root)} ({exc})"
            )
    if eligible is True and eval_path.is_file():
        try:
            metrics = _read_json(eval_path)
            status = (
                "completed"
                if metrics.get("status") == "completed"
                else str(metrics.get("status") or "invalid_evaluation")
            )
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            status = "invalid_evaluation"
            errors.append(
                f"invalid selected evaluation: {eval_path.relative_to(root)} ({exc})"
            )
    shards: list[dict[str, Any]] = []
    request_completion_fraction: float | None = None
    if eligible is True and status == "pending_branch":
        shards = [
            _selected_branch_shard_progress(root, method_id, index, errors)
            for index in range(2)
        ]
        fractions = [
            row["fold_request_completion_fraction"]
            for row in shards
            if row["fold_request_completion_fraction"] is not None
        ]
        request_completion_fraction = sum(float(value) for value in fractions)
        if request_completion_fraction > 1.0:
            errors.append(
                f"selected branch shard progress exceeds fold total: {method_id}"
            )
            request_completion_fraction = None
    elif status == "completed":
        request_completion_fraction = 1.0
    elif status == "gate_stopped":
        request_completion_fraction = 0.0
    return {
        "method_id": method_id,
        "status": status,
        "branch_scoring_eligible": eligible,
        "request_completion_fraction": request_completion_fraction,
        "shards": shards,
        "contract_path": contract_path.relative_to(root).as_posix(),
        "evaluation_path": eval_path.relative_to(root).as_posix(),
    }


def _selected_branch_shard_progress(
    root: Path,
    method_id: str,
    shard_index: int,
    errors: list[str],
) -> dict[str, Any]:
    short = method_id.split("_", 1)[0]
    run_id = (
        f"20260718_kuaisearch_mech_d2_{short}_selected_branch_"
        f"fold1_shard{shard_index}of2_v1"
    )
    run_dir = root / "runs" / run_id
    metadata_path = run_dir / "metadata.json"
    progress_path = run_dir / "progress.json"
    status = "missing"
    shard_request_count = None
    fold_request_count = None
    completed_requests = None
    progress_status = "missing"
    updated_at = None
    contract_sha = None
    metadata_trusted = False
    if metadata_path.is_file():
        try:
            metadata = _read_json(metadata_path)
            status = str(metadata.get("status") or "invalid")
            expected = {
                "run_id": run_id,
                "method_id": method_id,
                "normalized_query_fold": 1,
                "evidence_mode": "registered_mechanism_diagnostic_request_shard",
                "result_eligible": False,
                "qrels_read": False,
                "source_test_opened": False,
                "selected_nodes": list(SELECTED_NODES),
            }
            identity_ok = True
            for key, value in expected.items():
                if metadata.get(key) != value:
                    identity_ok = False
                    errors.append(
                        f"selected shard {key} mismatch: "
                        f"{metadata_path.relative_to(root)}"
                    )
            raw_fold_count = metadata.get("fold1_request_count")
            request_shard = metadata.get("request_shard")
            if type(raw_fold_count) is not int or raw_fold_count <= 0:
                identity_ok = False
                errors.append(
                    f"selected shard fold request count invalid: "
                    f"{metadata_path.relative_to(root)}"
                )
            elif not isinstance(request_shard, dict):
                identity_ok = False
                errors.append(
                    f"selected shard request contract invalid: "
                    f"{metadata_path.relative_to(root)}"
                )
            else:
                expected_shard_count = (
                    (raw_fold_count + 1) // 2
                    if shard_index == 0
                    else raw_fold_count // 2
                )
                expected_shard = {
                    "index": shard_index,
                    "count": 2,
                    "request_count": expected_shard_count,
                    "rule": "fold1_ordinal_mod_request_shard_count",
                }
                if request_shard != expected_shard:
                    identity_ok = False
                    errors.append(
                        f"selected shard request contract mismatch: "
                        f"{metadata_path.relative_to(root)}"
                    )
                else:
                    fold_request_count = raw_fold_count
                    shard_request_count = expected_shard_count
            raw_contract_sha = metadata.get("run_contract_sha256")
            if not isinstance(raw_contract_sha, str) or len(raw_contract_sha) != 64:
                identity_ok = False
                errors.append(
                    f"selected shard run contract invalid: "
                    f"{metadata_path.relative_to(root)}"
                )
            else:
                contract_sha = raw_contract_sha
            metadata_trusted = identity_ok
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            status = "invalid"
            errors.append(
                f"invalid selected shard metadata: "
                f"{metadata_path.relative_to(root)} ({exc})"
            )

    progress_trusted = False
    if metadata_trusted and status == "completed":
        completed_requests = shard_request_count
        progress_status = "completed"
        progress_trusted = True
    elif metadata_trusted and progress_path.is_file():
        try:
            progress = _read_json(progress_path)
            progress_status = str(progress.get("status") or "missing")
            raw_completed = progress.get("completed_requests")
            if type(raw_completed) is int and raw_completed >= 0:
                completed_requests = raw_completed
            updated_at = progress.get("updated_at")
            if progress.get("run_contract_sha256") != contract_sha:
                errors.append(
                    f"selected shard progress contract mismatch: "
                    f"{progress_path.relative_to(root)}"
                )
            else:
                progress_trusted = True
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            progress_status = "invalid"
            errors.append(
                f"invalid selected shard progress: "
                f"{progress_path.relative_to(root)} ({exc})"
            )
    if (
        completed_requests is not None
        and shard_request_count is not None
        and completed_requests > shard_request_count
    ):
        errors.append(
            f"selected shard progress exceeds request total: "
            f"{progress_path.relative_to(root)}"
        )
        progress_trusted = False
    fraction = None
    if (
        metadata_trusted
        and progress_trusted
        and completed_requests is not None
        and fold_request_count is not None
    ):
        fraction = completed_requests / fold_request_count
    return {
        "run_id": run_id,
        "shard_index": shard_index,
        "status": status,
        "progress_status": progress_status,
        "completed_requests": completed_requests,
        "shard_request_count": shard_request_count,
        "fold_request_count": fold_request_count,
        "fold_request_completion_fraction": fraction,
        "updated_at": updated_at,
    }


def _selected_contract_errors(
    contract: dict[str, Any], method_id: str
) -> list[str]:
    errors = []
    expected = {
        "contract_type": "transformer_deep_dive_d2_selected_branch_contract",
        "method_id": method_id,
        "scoring_population": "normalized_query_fold_1",
        "qrels_values_exposed_to_scorer": False,
        "source_test_opened": False,
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            errors.append(f"{key} mismatch")
    if contract.get("selected_nodes") != list(SELECTED_NODES):
        errors.append("selected_nodes mismatch")
    eligible = contract.get("branch_scoring_eligible")
    selected = contract.get("selected_block")
    reproduced = contract.get("fold1_negative_transition_reproduced")
    role = contract.get("evidence_role")
    if eligible is True:
        if type(selected) is not int or not 14 <= selected <= 27:
            errors.append("selected_block mismatch")
        if type(reproduced) is not bool:
            errors.append("fold1 reproduction mismatch")
        expected_role = (
            "registered_confirmatory_branch_localization"
            if reproduced is True
            else "exploratory_unresolved_transition_branch_localization"
        )
        if role != expected_role:
            errors.append("evidence_role mismatch")
    elif eligible is False:
        if selected is not None:
            errors.append("gate-stopped selected_block mismatch")
        if reproduced is not False:
            errors.append("gate-stopped fold1 reproduction mismatch")
        if role != "stopped_no_negative_fold0_transition":
            errors.append("gate-stopped evidence_role mismatch")
    return errors


def _in_flight_progress(
    root: Path,
    row: dict[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    run_dir = root / "runs" / row["run_id"]
    metadata_path = run_dir / "metadata.json"
    progress_path = run_dir / "progress.json"
    total_requests = None
    completed_requests = None
    progress_status = "missing"
    updated_at = None
    metadata_contract_sha = None
    try:
        metadata = _read_json(metadata_path)
        raw_total = metadata.get("fold_request_count")
        if type(raw_total) is int and raw_total > 0:
            total_requests = raw_total
        raw_contract_sha = metadata.get("run_contract_sha256")
        if isinstance(raw_contract_sha, str) and len(raw_contract_sha) == 64:
            metadata_contract_sha = raw_contract_sha
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    progress_trusted = False
    if progress_path.is_file():
        try:
            progress = _read_json(progress_path)
            progress_status = str(progress.get("status") or "missing")
            raw_completed = progress.get("completed_requests")
            if type(raw_completed) is int and raw_completed >= 0:
                completed_requests = raw_completed
            updated_at = progress.get("updated_at")
            progress_contract_sha = progress.get("run_contract_sha256")
            if (
                metadata_contract_sha is None
                or progress_contract_sha != metadata_contract_sha
            ):
                errors.append(
                    f"progress contract mismatch: {progress_path.relative_to(root)}"
                )
            else:
                progress_trusted = True
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            progress_status = "invalid"
            errors.append(
                f"invalid progress: {progress_path.relative_to(root)} ({exc})"
            )
    if (
        completed_requests is not None
        and total_requests is not None
        and completed_requests > total_requests
    ):
        errors.append(
            f"progress exceeds request total: {progress_path.relative_to(root)}"
        )
    fraction = None
    if (
        progress_trusted
        and completed_requests is not None
        and total_requests is not None
        and completed_requests <= total_requests
    ):
        fraction = completed_requests / total_requests
    return {
        "method_id": row["method_id"],
        "block_zero_based": row["block_zero_based"],
        "fold": row["fold"],
        "run_id": row["run_id"],
        "status": row["status"],
        "progress_status": progress_status,
        "completed_requests": completed_requests,
        "total_requests": total_requests,
        "request_completion_fraction": fraction,
        "updated_at": updated_at,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _status_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        status = str(row["status"])
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))
