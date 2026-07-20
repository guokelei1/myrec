"""Outcome-independent completion overview for the Transformer deep dive."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from myrec.mechanism.deep_dive_closeout_audit import (
    EXPECTED_DELIVERABLES,
    audit_deep_dive_closeout,
)
from myrec.mechanism.deep_dive_evidence_topology import (
    DELIVERABLE_MODEL_COVERAGE,
    MODEL_IDS,
)
from myrec.mechanism.deep_dive_progress import audit_deep_dive_progress
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_DELIVERABLE_MODEL_COVERAGE,
    COMPONENT_IDS,
    COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE,
    HYPOTHESIS_ALLOWED_DELIVERABLES,
    HYPOTHESIS_IDS,
    HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS,
    OPPORTUNITY_ALLOWED_DELIVERABLES,
    OPPORTUNITY_ALLOWED_MODEL_SCOPE,
    OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS,
    OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS,
)
from myrec.mechanism.deep_dive_opportunity_catalog import OPPORTUNITY_IDS


STAGE_IDS = tuple(f"d{index}" for index in range(1, 8))


def component_model_coverage_debt() -> dict[str, Any]:
    """Return frozen per-model causal coverage debt, independent of outcomes."""

    component_rows = [
        _component_completion_row(component_id, set())
        for component_id in COMPONENT_IDS
    ]
    return _component_coverage_debt(component_rows)


def build_deep_dive_overview(root: str | Path) -> dict[str, Any]:
    """Combine D2, closeout and component breadth without using outcomes."""

    closeout = audit_deep_dive_closeout(root)
    d2_progress = audit_deep_dive_progress(root)
    deliverables = closeout.get("deliverables")
    if (
        not isinstance(deliverables, Mapping)
        or set(deliverables) != set(EXPECTED_DELIVERABLES)
    ):
        raise ValueError("deep-dive overview deliverable coverage differs")
    completed = {
        name
        for name, identity in deliverables.items()
        if isinstance(identity, Mapping) and identity.get("status") == "completed"
    }
    formal_runs = [
        row
        for row in closeout.get("run_declarations", [])
        if isinstance(row, Mapping) and row.get("result_eligible") is True
    ]
    formal_statuses = Counter(
        str(row.get("status") or "missing") for row in formal_runs
    )
    component_rows = [
        _component_completion_row(component_id, completed)
        for component_id in COMPONENT_IDS
    ]
    hypothesis_rows = [
        _hypothesis_readiness_row(hypothesis_id, completed)
        for hypothesis_id in HYPOTHESIS_IDS
    ]
    opportunity_rows = [
        _opportunity_readiness_row(opportunity_id, completed)
        for opportunity_id in OPPORTUNITY_IDS
    ]
    stage_rows = []
    for stage_id in STAGE_IDS:
        registered = sorted(
            name for name in EXPECTED_DELIVERABLES if name.startswith(f"{stage_id}_")
        )
        closed = sorted(set(registered) & completed)
        stage_rows.append(
            {
                "stage_id": stage_id.upper(),
                "registered_deliverables": registered,
                "completed_deliverables": closed,
                "pending_deliverables": sorted(set(registered) - completed),
                "completed_count": len(closed),
                "registered_count": len(registered),
                "completion_fraction": len(closed) / len(registered),
            }
        )
    causal_applicable = [
        row for row in component_rows if row["causal_support_capable_deliverables"]
    ]
    coverage_debt = _component_coverage_debt(component_rows)
    audit_failed = bool(closeout.get("failures")) or d2_progress.get("status") != "ok"
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_outcome_independent_overview",
        "status": "failed" if audit_failed else "ok",
        "completion_status": closeout.get("status"),
        "deliverable_summary": {
            "completed": len(completed),
            "registered": len(EXPECTED_DELIVERABLES),
            "completion_fraction": len(completed) / len(EXPECTED_DELIVERABLES),
            "completed_deliverables": sorted(completed),
            "pending_deliverables": sorted(set(EXPECTED_DELIVERABLES) - completed),
        },
        "stage_completion": stage_rows,
        "component_completion": component_rows,
        "component_summary": {
            "registered_components": len(component_rows),
            "with_any_registered_output_closed": sum(
                bool(row["completed_registered_deliverables"])
                for row in component_rows
            ),
            "with_all_registered_outputs_closed": sum(
                row["all_registered_outputs_closed"] for row in component_rows
            ),
            "with_no_registered_output_closed": sum(
                not row["completed_registered_deliverables"]
                for row in component_rows
            ),
            "causal_support_capable_components": len(causal_applicable),
            "causal_support_capable_component_ids": sorted(
                row["component_id"] for row in causal_applicable
            ),
            "registered_but_descriptive_only_components": (
                coverage_debt["globally_registered_but_descriptive_only_components"]
            ),
            "with_all_causal_capable_outputs_closed": sum(
                row["all_causal_capable_outputs_closed"] is True
                for row in causal_applicable
            ),
        },
        "model_component_summary": coverage_debt["per_model"],
        "component_model_cell_summary": {
            "component_count": coverage_debt["component_count"],
            "model_count": coverage_debt["model_count"],
            "cell_count": coverage_debt["component_model_cell_count"],
            "registered_cell_count": coverage_debt[
                "registered_component_model_cell_count"
            ],
            "causal_support_capable_cell_count": coverage_debt[
                "causal_support_capable_component_model_cell_count"
            ],
            "registered_but_descriptive_only_cell_count": coverage_debt[
                "registered_but_descriptive_only_component_model_cell_count"
            ],
            "not_directly_registered_cell_count": coverage_debt[
                "not_directly_registered_component_model_cell_count"
            ],
            "disposition_counts": coverage_debt[
                "component_model_cell_disposition_counts"
            ],
            "all_cells_have_outcome_independent_disposition": coverage_debt[
                "all_component_model_cells_have_outcome_independent_disposition"
            ],
            "scientific_support_inferred_from_coverage": False,
        },
        "hypothesis_evidence_readiness": hypothesis_rows,
        "hypothesis_readiness_summary": {
            "registered_hypotheses": len(hypothesis_rows),
            "support_authorized_in_this_stage": sum(
                row["support_authorized_in_this_stage"]
                for row in hypothesis_rows
            ),
            "with_all_required_groups_having_a_completed_source": sum(
                row["all_required_groups_have_completed_source"] is True
                for row in hypothesis_rows
            ),
        },
        "opportunity_evidence_readiness": opportunity_rows,
        "opportunity_readiness_summary": {
            "registered_opportunities": len(opportunity_rows),
            "with_all_global_groups_having_a_completed_source": sum(
                row["all_global_groups_have_completed_source"]
                for row in opportunity_rows
            ),
            "with_any_model_structurally_ready": sum(
                any(
                    model["all_required_groups_have_completed_source"] is True
                    for model in row["per_model_readiness"].values()
                )
                for row in opportunity_rows
            ),
        },
        "d2_causal_core_progress": d2_progress,
        "formal_run_summary": {
            "declared": len(formal_runs),
            "status_counts": dict(sorted(formal_statuses.items())),
            "completed_integrity_checked": int(
                closeout.get("formal_completed_integrity_checked", 0)
            ),
        },
        "closeout_summary": {
            "audit_failures": list(closeout.get("failures", [])),
            "pending_conditions": len(closeout.get("pending", [])),
            "mechanical_failure_records": len(
                closeout.get("mechanical_failure_records", [])
            ),
        },
        "source_test_content_read_by_this_overview": bool(
            closeout.get("source_test_content_read_by_this_audit")
        ),
        "qrels_content_read_by_this_overview": bool(
            closeout.get("qrels_content_read_by_this_audit")
        ),
        "outcome_values_used_for_progress": False,
        "scientific_effect_values_summarized": False,
        "hypothesis_support_inferred_from_readiness": False,
        "opportunity_priority_inferred_from_readiness": False,
    }


def _component_completion_row(
    component_id: str, completed: set[str]
) -> dict[str, Any]:
    registered = set(COMPONENT_ALLOWED_DELIVERABLES[component_id])
    causal = set(COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component_id])
    completed_registered = registered & completed
    completed_causal = causal & completed
    per_model = {}
    for model_id in MODEL_IDS:
        model_registered = {
            deliverable
            for deliverable in registered
            if model_id
            in COMPONENT_DELIVERABLE_MODEL_COVERAGE[component_id][deliverable]
        }
        model_causal = model_registered & causal
        per_model[model_id] = {
            "registered_deliverables": sorted(model_registered),
            "completed_registered_deliverables": sorted(
                model_registered & completed
            ),
            "pending_registered_deliverables": sorted(
                model_registered - completed
            ),
            "causal_support_capable_deliverables": sorted(model_causal),
            "completed_causal_capable_deliverables": sorted(
                model_causal & completed
            ),
            "pending_causal_capable_deliverables": sorted(
                model_causal - completed
            ),
        }
    return {
        "component_id": component_id,
        "registered_deliverables": sorted(registered),
        "completed_registered_deliverables": sorted(completed_registered),
        "pending_registered_deliverables": sorted(registered - completed),
        "registered_completion_fraction": len(completed_registered) / len(registered),
        "causal_support_capable_deliverables": sorted(causal),
        "completed_causal_capable_deliverables": sorted(completed_causal),
        "pending_causal_capable_deliverables": sorted(causal - completed),
        "all_registered_outputs_closed": completed_registered == registered,
        "all_causal_capable_outputs_closed": (
            completed_causal == causal if causal else None
        ),
        "per_model": per_model,
        "scientific_support_inferred_from_completion": False,
    }


def _component_coverage_debt(
    component_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    descriptive_only_components = sorted(
        row["component_id"]
        for row in component_rows
        if not row["causal_support_capable_deliverables"]
    )
    per_model = {}
    for model_id in MODEL_IDS:
        registered_components = sorted(
            row["component_id"]
            for row in component_rows
            if row["per_model"][model_id]["registered_deliverables"]
        )
        causal_components = sorted(
            row["component_id"]
            for row in component_rows
            if row["per_model"][model_id]["causal_support_capable_deliverables"]
        )
        per_model[model_id] = {
            "registered_components": registered_components,
            "causal_support_capable_components": causal_components,
            "registered_but_descriptive_only_components": sorted(
                set(registered_components) - set(causal_components)
            ),
            "not_directly_registered_components": sorted(
                set(COMPONENT_IDS) - set(registered_components)
            ),
            "registered_component_count": len(registered_components),
            "causal_support_capable_component_count": len(causal_components),
        }
    registered_cell_count = sum(
        row["registered_component_count"] for row in per_model.values()
    )
    causal_cell_count = sum(
        row["causal_support_capable_component_count"]
        for row in per_model.values()
    )
    component_model_cell_count = len(COMPONENT_IDS) * len(MODEL_IDS)
    return {
        "component_count": len(COMPONENT_IDS),
        "model_count": len(MODEL_IDS),
        "component_model_cell_count": component_model_cell_count,
        "registered_component_model_cell_count": registered_cell_count,
        "causal_support_capable_component_model_cell_count": causal_cell_count,
        "registered_but_descriptive_only_component_model_cell_count": (
            registered_cell_count - causal_cell_count
        ),
        "not_directly_registered_component_model_cell_count": (
            component_model_cell_count - registered_cell_count
        ),
        "component_model_cell_disposition_counts": {
            "registered_causal_support_capable": causal_cell_count,
            "registered_descriptive_only": registered_cell_count
            - causal_cell_count,
            "not_directly_registered": component_model_cell_count
            - registered_cell_count,
        },
        "all_component_model_cells_have_outcome_independent_disposition": True,
        "globally_registered_but_descriptive_only_components": (
            descriptive_only_components
        ),
        "per_model": per_model,
        "scientific_support_inferred_from_coverage": False,
    }


def _hypothesis_readiness_row(
    hypothesis_id: str, completed: set[str]
) -> dict[str, Any]:
    allowed = set(HYPOTHESIS_ALLOWED_DELIVERABLES[hypothesis_id])
    groups = HYPOTHESIS_SUPPORTED_REQUIRED_EVIDENCE_GROUPS[hypothesis_id]
    group_rows = _evidence_group_rows(groups, completed)
    return {
        "hypothesis_id": hypothesis_id,
        "allowed_deliverables": sorted(allowed),
        "completed_allowed_deliverables": sorted(allowed & completed),
        "pending_allowed_deliverables": sorted(allowed - completed),
        "support_authorized_in_this_stage": bool(groups),
        "required_evidence_groups": group_rows,
        "all_required_groups_have_completed_source": (
            all(row["has_completed_source"] for row in group_rows)
            if groups
            else None
        ),
        "hypothesis_support_inferred_from_readiness": False,
    }


def _opportunity_readiness_row(
    opportunity_id: str, completed: set[str]
) -> dict[str, Any]:
    allowed = set(OPPORTUNITY_ALLOWED_DELIVERABLES[opportunity_id])
    global_groups = OPPORTUNITY_PRIMARY_REQUIRED_EVIDENCE_GROUPS[opportunity_id]
    global_rows = _evidence_group_rows(global_groups, completed)
    per_model_groups = OPPORTUNITY_PRIMARY_PER_MODEL_EVIDENCE_GROUPS.get(
        opportunity_id, ()
    )
    per_model = {}
    for model_id in sorted(OPPORTUNITY_ALLOWED_MODEL_SCOPE[opportunity_id]):
        model_rows = []
        for index, group in enumerate(per_model_groups):
            model_sources = {
                deliverable
                for deliverable in group
                if model_id in DELIVERABLE_MODEL_COVERAGE[deliverable]
            }
            if not model_sources:
                raise ValueError(
                    "opportunity readiness has no model-local source: "
                    f"{opportunity_id}.{model_id}.group_{index}"
                )
            model_rows.extend(
                _evidence_group_rows((model_sources,), completed, start=index)
            )
        per_model[model_id] = {
            "required_evidence_groups": model_rows,
            "all_required_groups_have_completed_source": (
                all(row["has_completed_source"] for row in model_rows)
                if per_model_groups
                else None
            ),
        }
    return {
        "opportunity_id": opportunity_id,
        "allowed_deliverables": sorted(allowed),
        "completed_allowed_deliverables": sorted(allowed & completed),
        "pending_allowed_deliverables": sorted(allowed - completed),
        "global_required_evidence_groups": global_rows,
        "all_global_groups_have_completed_source": all(
            row["has_completed_source"] for row in global_rows
        ),
        "allowed_model_scope": sorted(
            OPPORTUNITY_ALLOWED_MODEL_SCOPE[opportunity_id]
        ),
        "per_model_readiness": per_model,
        "opportunity_priority_inferred_from_readiness": False,
    }


def _evidence_group_rows(
    groups: tuple[set[str], ...] | tuple[frozenset[str], ...],
    completed: set[str],
    *,
    start: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    for offset, group in enumerate(groups):
        sources = set(group)
        closed = sources & completed
        rows.append(
            {
                "group_index": start + offset,
                "registered_sources": sorted(sources),
                "completed_sources": sorted(closed),
                "pending_sources": sorted(sources - completed),
                "has_completed_source": bool(closed),
                "scientific_gate_passed": None,
            }
        )
    return rows
