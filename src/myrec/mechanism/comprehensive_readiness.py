"""Outcome-independent readiness for the complete Transformer exploration report."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from myrec.mechanism.deep_dive_closeout_audit import (
    EXPECTED_DELIVERABLES,
    audit_deep_dive_closeout,
)
from myrec.mechanism.deep_dive_progress import audit_deep_dive_progress
from myrec.mechanism.deep_dive_producer_topology import (
    audit_deep_dive_producer_topology,
)
from myrec.mechanism.deep_dive_evidence_topology import (
    DELIVERABLE_MODEL_COVERAGE,
    MODEL_IDS,
)
from myrec.mechanism.frozen_model_architecture_audit import (
    audit_frozen_model_architecture,
)
from myrec.mechanism.deep_dive_report_contract import (
    COMPONENT_ALLOWED_DELIVERABLES,
    COMPONENT_IDS,
    COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE,
)
from myrec.mechanism.supplemental_evidence_registry import (
    audit_supplemental_evidence_registry,
)
from myrec.mechanism.transformer_interface_inventory import (
    build_transformer_interface_coverage,
)


CAUSAL_SUPPLEMENT_EVIDENCE_LEVELS = {
    "preregistered_reverse_causal_extension",
    "preregistered_design_gate_synthesis",
}
PRIMARY_TRANSFER_MODELS = frozenset(MODEL_IDS[2:])


def build_comprehensive_readiness(root: str | Path = ".") -> dict[str, Any]:
    """Combine formal and supplemental completion without inspecting effect values."""

    root = Path(root).resolve()
    closeout = audit_deep_dive_closeout(root)
    d2 = audit_deep_dive_progress(root)
    supplements = audit_supplemental_evidence_registry(root)
    producer_topology = audit_deep_dive_producer_topology(root)
    architecture_audit = audit_frozen_model_architecture(root)
    deliverables = closeout.get("deliverables", {})
    if not isinstance(deliverables, Mapping) or set(deliverables) != set(
        EXPECTED_DELIVERABLES
    ):
        raise ValueError("comprehensive readiness formal deliverable coverage drift")
    completed_formal = {
        deliverable
        for deliverable, identity in deliverables.items()
        if isinstance(identity, Mapping) and identity.get("status") == "completed"
    }
    completed_supplements = {
        row["evidence_id"]
        for row in supplements["entries"]
        if row["status"] == "completed"
    }
    interface_coverage = build_transformer_interface_coverage(
        completed_formal=completed_formal,
        completed_supplements=completed_supplements,
        supplement_model_scopes={
            str(row["evidence_id"]): set(row["model_scope"])
            for row in supplements["entries"]
        },
        supplement_component_scopes={
            str(row["evidence_id"]): set(row["components"])
            for row in supplements["entries"]
        },
    )
    mechanical_nonresults = [
        row
        for row in closeout.get("mechanical_failure_records", [])
        if isinstance(row, Mapping) and row.get("status") == "mechanical_failure"
    ]
    mechanical_nonresult_ids = [str(row["run_id"]) for row in mechanical_nonresults]
    if len(mechanical_nonresult_ids) != len(set(mechanical_nonresult_ids)):
        raise ValueError("comprehensive readiness mechanical non-result identity drift")
    component_rows = []
    for component_id in COMPONENT_IDS:
        formal = set(COMPONENT_ALLOWED_DELIVERABLES[component_id])
        component_supplements = [
            row
            for row in supplements["entries"]
            if component_id in row["components"]
        ]
        registered_supplements = {
            row["evidence_id"] for row in component_supplements
        }
        registered_causal_formal = (
            formal & set(COMPONENT_SUPPORT_REQUIRED_CAUSAL_EVIDENCE[component_id])
        )
        registered_causal_supplements = {
            row["evidence_id"]
            for row in component_supplements
            if row["evidence_level"] in CAUSAL_SUPPLEMENT_EVIDENCE_LEVELS
        }
        completed_causal_formal = registered_causal_formal & completed_formal
        completed_causal_supplements = (
            registered_causal_supplements & completed_supplements
        )
        causal_formal_models_registered = set().union(
            *(DELIVERABLE_MODEL_COVERAGE[item] for item in registered_causal_formal),
            set(),
        )
        causal_formal_models_completed = set().union(
            *(DELIVERABLE_MODEL_COVERAGE[item] for item in completed_causal_formal),
            set(),
        )
        causal_supplement_models_registered = set().union(
            *(
                set(row["model_scope"])
                for row in component_supplements
                if row["evidence_id"] in registered_causal_supplements
            ),
            set(),
        )
        causal_supplement_models_completed = set().union(
            *(
                set(row["model_scope"])
                for row in component_supplements
                if row["evidence_id"] in completed_causal_supplements
            ),
            set(),
        )
        causal_models_registered = (
            causal_formal_models_registered | causal_supplement_models_registered
        )
        causal_models_completed = (
            causal_formal_models_completed | causal_supplement_models_completed
        )
        component_rows.append(
            {
                "component_id": component_id,
                "formal_registered": sorted(formal),
                "formal_completed": sorted(formal & completed_formal),
                "formal_pending": sorted(formal - completed_formal),
                "supplements_registered": sorted(registered_supplements),
                "supplements_completed": sorted(
                    registered_supplements & completed_supplements
                ),
                "supplements_pending": sorted(
                    registered_supplements - completed_supplements
                ),
                "causal_role_formal_registered": sorted(
                    registered_causal_formal
                ),
                "causal_role_formal_completed": sorted(
                    completed_causal_formal
                ),
                "causal_role_supplements_registered": sorted(
                    registered_causal_supplements
                ),
                "causal_role_supplements_completed": sorted(
                    completed_causal_supplements
                ),
                "causal_role_artifact_registered": bool(
                    registered_causal_formal or registered_causal_supplements
                ),
                "causal_role_artifact_completed": bool(
                    completed_causal_formal or completed_causal_supplements
                ),
                "causal_role_model_scope_registered": sorted(
                    causal_models_registered
                ),
                "causal_role_model_scope_completed": sorted(
                    causal_models_completed
                ),
                "q2_q3_causal_role_artifacts_registered": (
                    PRIMARY_TRANSFER_MODELS.issubset(causal_models_registered)
                ),
                "q2_q3_causal_role_artifacts_completed": (
                    PRIMARY_TRANSFER_MODELS.issubset(causal_models_completed)
                ),
                "any_evidence_artifact_completed": bool(
                    (formal & completed_formal)
                    or (registered_supplements & completed_supplements)
                ),
                "artifact_role_interpretation": (
                    "Completion reports availability only. A causal-role artifact "
                    "still requires its registered sign, uncertainty, specificity, "
                    "and cross-model gates before it can support a mechanism."
                ),
                "scientific_support_inferred_from_completion": False,
            }
        )
    formal_total = len(EXPECTED_DELIVERABLES)
    supplement_total = supplements["entry_count"]
    artifact_completed = len(completed_formal) + len(completed_supplements)
    artifact_total = formal_total + supplement_total
    d2_terminal = bool(
        d2.get("status") == "ok"
        and d2["fixed"]["remaining_mandatory_bundles"] == 0
        and d2["maximum_remaining_scientific_bundles"] == 0
    )
    readiness_gates = {
        "formal_19_closeout_completed": closeout.get("status") == "completed",
        "d2_all_fixed_and_conditional_units_terminal": d2_terminal,
        "supplement_21_registry_completed": supplements.get("status") == "completed",
        "producer_topology_complete": producer_topology.get("status") == "completed",
        "frozen_model_architecture_audit_completed": (
            architecture_audit.get("status") == "completed"
            and not architecture_audit.get("failures")
        ),
        "component_functional_design_gate_completed": (
            "component_functional_design_gate_synthesis" in completed_supplements
        ),
        "no_closeout_audit_failures": not closeout.get("failures"),
        "no_supplement_registry_failures": not supplements.get("failures"),
        "source_test_unopened": bool(
            closeout.get("source_test_content_read_by_this_audit") is False
            and supplements.get("source_test_opened") is False
        ),
    }
    failed = bool(
        closeout.get("failures")
        or supplements.get("failures")
        or d2.get("status") != "ok"
        or producer_topology.get("status") != "completed"
        or architecture_audit.get("status") != "completed"
    )
    final_ready = all(readiness_gates.values())
    return {
        "schema_version": 1,
        "analysis_type": "transformer_comprehensive_report_readiness",
        "status": "failed" if failed else ("completed" if final_ready else "pending"),
        "final_comprehensive_report_ready": final_ready,
        "readiness_gates": readiness_gates,
        "formal": {
            "completed": len(completed_formal),
            "registered": formal_total,
            "completion_fraction": len(completed_formal) / formal_total,
            "completed_deliverables": sorted(completed_formal),
            "pending_deliverables": sorted(
                set(EXPECTED_DELIVERABLES) - completed_formal
            ),
        },
        "supplements": {
            "completed": len(completed_supplements),
            "registered": supplement_total,
            "completion_fraction": len(completed_supplements) / supplement_total,
            "completed_evidence": sorted(completed_supplements),
            "pending_evidence": sorted(
                set(row["evidence_id"] for row in supplements["entries"])
                - completed_supplements
            ),
            "registry": supplements["registry"],
            "registry_manifest": supplements["registry_manifest"],
        },
        "combined_artifact_completion": {
            "completed": artifact_completed,
            "registered": artifact_total,
            "completion_fraction": artifact_completed / artifact_total,
            "interpretation": (
                "artifact closure only; not compute progress, statistical power, or "
                "scientific certainty"
            ),
        },
        "mechanical_nonresults": {
            "retained": len(mechanical_nonresults),
            "run_ids": sorted(mechanical_nonresult_ids),
            "interpretation": (
                "Bound mechanical failures are retained for reproducibility. They "
                "are neither scientific null results nor evidence against a mechanism."
            ),
        },
        "producer_topology": {
            "status": producer_topology["status"],
            "formal_covered": producer_topology["formal_covered"],
            "formal_registered": producer_topology["formal_registered"],
            "supplements_covered": producer_topology["supplements_covered"],
            "supplements_registered": producer_topology[
                "supplements_registered"
            ],
            "queued_or_watched": producer_topology["queued_or_watched"],
            "failures": list(producer_topology["failures"]),
            "scientific_effect_values_read": producer_topology[
                "scientific_effect_values_read"
            ],
        },
        "frozen_model_architecture_audit": architecture_audit,
        "d2_causal_core": {
            "resolved_scientific_units": d2["resolved_scientific_units"],
            "maximum_total_scientific_bundles": d2[
                "maximum_total_scientific_bundles"
            ],
            "resolution_fraction": d2["resolution_fraction"],
            "fixed_completed": d2["fixed"]["completed_bundles"],
            "fixed_registered": d2["fixed"]["registered_bundles"],
            "request_weighted_execution_fraction": d2["fixed"][
                "request_weighted_execution_fraction"
            ],
            "scientific_effect_values_read": d2["scientific_effect_values_read"],
        },
        "component_artifact_coverage": component_rows,
        "component_count": len(component_rows),
        "components_with_any_completed_artifact": sum(
            row["any_evidence_artifact_completed"] for row in component_rows
        ),
        "components_with_registered_causal_role_artifact": sum(
            row["causal_role_artifact_registered"] for row in component_rows
        ),
        "components_without_registered_causal_role_artifact": sorted(
            row["component_id"]
            for row in component_rows
            if not row["causal_role_artifact_registered"]
        ),
        "components_with_completed_causal_role_artifact": sum(
            row["causal_role_artifact_completed"] for row in component_rows
        ),
        "components_with_completed_q2_q3_causal_role_artifacts": sum(
            row["q2_q3_causal_role_artifacts_completed"] for row in component_rows
        ),
        "component_coverage_interpretation": (
            "Any-artifact coverage includes descriptive geometry. Causal-role "
            "coverage means only that a preregistered interventional evaluator is "
            "available. Q2/Q3 completion additionally requires artifact availability "
            "in both primary transfer models; none of these counts is scientific "
            "support or design qualification."
        ),
        "transformer_internal_interface_coverage": interface_coverage,
        "effect_values_used_for_readiness": False,
        "scientific_support_inferred_from_completion": False,
        "qrels_files_opened_by_this_readiness": False,
        "source_test_opened": False,
    }
