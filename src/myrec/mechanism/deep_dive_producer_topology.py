"""Static producer coverage for every registered deep-dive artifact.

This module deliberately audits source topology only.  It never opens model
scores, qrels, or scientific effect fields.  The purpose is to fail closed if
an output remains registered while its project-owned producer or queue entry
point is renamed, removed, or left undocumented.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import yaml

from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.supplemental_evidence_registry import (
    EXPECTED_SUPPLEMENT_IDS,
    REGISTRY_PATH,
)


GENERIC_WATCHER = "scripts/watch_then_run.sh"


def _spec(
    output_path: str,
    producer_script: str,
    *upstream_families: str,
    orchestrators: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "output_path": output_path,
        "producer_script": producer_script,
        "orchestrator_scripts": orchestrators or (producer_script,),
        "upstream_families": upstream_families,
    }


FORMAL_PRODUCER_TOPOLOGY: dict[str, dict[str, Any]] = {
    "d1_representation": _spec(
        EXPECTED_DELIVERABLES["d1_representation"],
        "scripts/synthesize_deep_dive_d1.py",
        "q2_q3_region_probe_evaluations",
    ),
    "d2_q3_native_gate": _spec(
        EXPECTED_DELIVERABLES["d2_q3_native_gate"],
        "scripts/evaluate_deep_dive_q3_native_gates.py",
        "q3_native_position_patch_bundles",
    ),
    "d2_postblock": _spec(
        EXPECTED_DELIVERABLES["d2_postblock"],
        "scripts/synthesize_deep_dive_postblock.py",
        "q2_q3_fold0_selections",
        "q2_q3_fold1_confirmations",
        orchestrators=("scripts/run_deep_dive_d2_synthesis_queue.sh",),
    ),
    "d2_selected_branches": _spec(
        EXPECTED_DELIVERABLES["d2_selected_branches"],
        "scripts/synthesize_deep_dive_selected_branches.py",
        "q2_q3_selected_branch_contracts",
        "eligible_selected_branch_evaluations",
        orchestrators=("scripts/run_deep_dive_d2_synthesis_queue.sh",),
    ),
    "d3_attention_edges": _spec(
        EXPECTED_DELIVERABLES["d3_attention_edges"],
        "scripts/evaluate_deep_dive_attention_edges.py",
        "q2_q3_attention_edge_bundles_b13_b20_b27",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d3_attention_heads": _spec(
        EXPECTED_DELIVERABLES["d3_attention_heads"],
        "scripts/evaluate_deep_dive_attention_heads.py",
        "q2_q3_attention_head_observations_b13_b20_b27",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d3_attention_groups": _spec(
        EXPECTED_DELIVERABLES["d3_attention_groups"],
        "scripts/evaluate_deep_dive_attention_groups.py",
        "q2_q3_attention_group_bundles_b13_b20_b27",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d4_mlp_groups": _spec(
        EXPECTED_DELIVERABLES["d4_mlp_groups"],
        "scripts/evaluate_deep_dive_mlp_groups.py",
        "q2_q3_mlp_group_bundles_b13_b20_b27",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d5_context": _spec(
        EXPECTED_DELIVERABLES["d5_context"],
        "scripts/evaluate_deep_dive_contextual_controls.py",
        "q2_q3_contextual_control_bundles",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d5_rope": _spec(
        EXPECTED_DELIVERABLES["d5_rope"],
        "scripts/evaluate_deep_dive_rope.py",
        "q2_q3_rope_bundles_b13_b20_b27",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d6_q2_native_readout": _spec(
        EXPECTED_DELIVERABLES["d6_q2_native_readout"],
        "scripts/evaluate_deep_dive_q2_native_readout.py",
        "q2_native_readout_bundle",
    ),
    "d6_q3_native_readout": _spec(
        EXPECTED_DELIVERABLES["d6_q3_native_readout"],
        "scripts/evaluate_deep_dive_q3_native_readout.py",
        "q3_native_readout_bundle",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d6_q0_trajectory": _spec(
        EXPECTED_DELIVERABLES["d6_q0_trajectory"],
        "scripts/evaluate_deep_dive_q0_trajectory.py",
        "q0_full_null_trajectory_bundles",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d6_q1_trajectory": _spec(
        EXPECTED_DELIVERABLES["d6_q1_trajectory"],
        "scripts/evaluate_deep_dive_q1_trajectory.py",
        "q1_kv_trajectory_bundle",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d6_q0_q1_branches": _spec(
        EXPECTED_DELIVERABLES["d6_q0_q1_branches"],
        "scripts/evaluate_deep_dive_breadth_branches.py",
        "q0_q1_branch_bundles_b13_b20_b27",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d6_q0_q1_readouts": _spec(
        EXPECTED_DELIVERABLES["d6_q0_q1_readouts"],
        "scripts/evaluate_deep_dive_breadth_readouts.py",
        "q0_q1_final_readout_bundles",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d7_q2_objective": _spec(
        EXPECTED_DELIVERABLES["d7_q2_objective"],
        "scripts/synthesize_deep_dive_objective_conflict.py",
        "q2_base_final_objective_conflict_runs",
    ),
    "d7_q3_lora_path": _spec(
        EXPECTED_DELIVERABLES["d7_q3_lora_path"],
        "scripts/analyze_deep_dive_q3_lora_path.py",
        "q3_lora_checkpoint_states",
    ),
    "d7_optimizer_replay": _spec(
        EXPECTED_DELIVERABLES["d7_optimizer_replay"],
        "scripts/evaluate_deep_dive_optimizer_replays.py",
        "q2_q3_exact_optimizer_replay_bundles",
        orchestrators=(GENERIC_WATCHER,),
    ),
}


SUPPLEMENT_PRODUCER_TOPOLOGY: dict[str, dict[str, Any]] = {
    "d0_embedding_readout_geometry": _spec(
        "runs/20260718_kuaisearch_mech_d0_embedding_readout_v1/embedding_readout.json",
        "scripts/analyze_deep_dive_embedding_readout.py",
        "frozen_q2_q3_embeddings_and_readouts",
    ),
    "d1_activation_anisotropy": _spec(
        "runs/20260718_kuaisearch_mech_d1_activation_anisotropy_v1/activation_anisotropy.json",
        "scripts/analyze_deep_dive_activation_anisotropy.py",
        "completed_d1_representation_bundles",
    ),
    "d1_candidate_block_flow": _spec(
        "runs/20260718_kuaisearch_mech_d1_candidate_block_flow_v1/metrics.json",
        "scripts/analyze_deep_dive_candidate_block_flow.py",
        "completed_d1_representation_bundles",
    ),
    "d1_candidate_residual_geometry": _spec(
        "runs/20260718_kuaisearch_mech_d1_candidate_residual_v1/metrics.json",
        "scripts/analyze_deep_dive_candidate_residual.py",
        "completed_d1_representation_bundles",
    ),
    "d1_preference_subspace_geometry": _spec(
        "runs/20260718_kuaisearch_mech_d1_preference_subspace_v1/metrics.json",
        "scripts/analyze_deep_dive_preference_subspace.py",
        "completed_d1_representation_bundles",
    ),
    "d1_query_causal_floor": _spec(
        "runs/20260718_kuaisearch_mech_d1_causal_floor_v1/causal_floor.json",
        "scripts/analyze_deep_dive_causal_floor.py",
        "completed_d1_representation_bundles",
    ),
    "d2_rmsnorm_flow": _spec(
        "runs/20260718_kuaisearch_mech_d2_rmsnorm_flow_v1/rmsnorm_flow.json",
        "scripts/analyze_deep_dive_rmsnorm_flow.py",
        "completed_d1_representation_bundles",
    ),
    "d3_attention_pattern_synthesis": _spec(
        "runs/20260719_kuaisearch_mech_d3_attention_patterns_v1/metrics.json",
        "scripts/analyze_deep_dive_attention_patterns.py",
        "completed_attention_head_observations",
    ),
    "d3_full_null_position_shift_audit": _spec(
        "runs/20260719_kuaisearch_mech_d3_position_shift_audit_v1/metrics.json",
        "scripts/analyze_deep_dive_attention_position_shifts.py",
        "completed_attention_head_observations",
    ),
    "d3_qk_stage_geometry_v3": _spec(
        "runs/20260719_kuaisearch_mech_d3_qk_geometry_v3/metrics.json",
        "scripts/analyze_deep_dive_attention_qk_geometry.py",
        "completed_attention_head_observations",
    ),
    "d4_mlp_feature_formation_extension": _spec(
        "runs/20260719_kuaisearch_mech_d4_mlp_formation_eval_v1/metrics.json",
        "scripts/evaluate_deep_dive_mlp_features.py",
        "q2_q3_mlp_feature_bundles_b13_b20_b27",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "d5_rope_position_geometry": _spec(
        "runs/20260718_kuaisearch_mech_d5_rope_geometry_v1/metrics.json",
        "scripts/analyze_deep_dive_rope_geometry.py",
        "frozen_position_geometry_inputs",
    ),
    "d6_frozen_logit_lens": _spec(
        "runs/20260718_kuaisearch_mech_d6_logit_lens_v1/logit_lens.json",
        "scripts/analyze_deep_dive_logit_lens.py",
        "frozen_q2_q3_hidden_states_and_readouts",
    ),
    "d7_objective_common_nullspace": _spec(
        "runs/20260718_kuaisearch_mech_d7_objective_nullspace_v1/objective_nullspace.json",
        "scripts/analyze_deep_dive_objective_nullspace.py",
        "completed_q2_objective_family",
    ),
    "d7_q2_objective_family_shares": _spec(
        "runs/20260718_kuaisearch_mech_d7_q2_objective_conflict_synthesis_v1/"
        "parameter_family_share_analysis.json",
        "scripts/analyze_deep_dive_q2_objective_family_shares.py",
        "completed_q2_objective_family",
    ),
    "d7_q2_parameter_update_geometry": _spec(
        "runs/20260718_kuaisearch_mech_d7_q2_parameter_update_geometry_v1/"
        "parameter_update_geometry.json",
        "scripts/analyze_deep_dive_q2_parameter_updates.py",
        "q2_checkpoint_optimizer_states",
    ),
    "d7_q2_update_anisotropy": _spec(
        "runs/20260718_kuaisearch_mech_d7_q2_update_anisotropy_v1/update_anisotropy.json",
        "scripts/analyze_deep_dive_q2_update_anisotropy.py",
        "q2_checkpoint_optimizer_states",
    ),
    "d7_q3_lora_head_geometry": _spec(
        "runs/20260718_kuaisearch_mech_d7_q3_lora_head_geometry_v1/lora_head_geometry.json",
        "scripts/analyze_deep_dive_q3_lora_head_geometry.py",
        "completed_q3_lora_path",
    ),
    "d6_native_readout_diagnostics": _spec(
        "runs/20260719_kuaisearch_mech_d6_native_readout_diagnostics_v1/metrics.json",
        "scripts/analyze_deep_dive_native_readout_diagnostics.py",
        "q2_q3_native_readout_bundles",
        orchestrators=(GENERIC_WATCHER,),
    ),
    "component_state_reverse_necessity_v2": _spec(
        "runs/20260719_kuaisearch_mech_component_necessity_eval_v1/metrics.json",
        "scripts/evaluate_deep_dive_component_necessity.py",
        "q2_q3_selected_branch_contracts",
        "q2_q3_selected_branch_bundles",
        orchestrators=("scripts/run_deep_dive_component_necessity_eval_queue.sh",),
    ),
    "component_functional_design_gate_synthesis": _spec(
        "runs/20260719_kuaisearch_mech_component_design_synthesis_v1/metrics.json",
        "scripts/synthesize_deep_dive_component_design.py",
        "d2_selected_branch_synthesis",
        "component_state_reverse_necessity_v2",
        orchestrators=("scripts/run_deep_dive_component_design_synthesis_queue.sh",),
    ),
}


def audit_deep_dive_producer_topology(
    root: str | Path = ".",
    *,
    formal_topology: Mapping[str, Mapping[str, Any]] = FORMAL_PRODUCER_TOPOLOGY,
    supplement_topology: Mapping[str, Mapping[str, Any]] = (
        SUPPLEMENT_PRODUCER_TOPOLOGY
    ),
) -> dict[str, Any]:
    """Verify exhaustive producer coverage without opening experiment outputs."""

    root_path = Path(root).resolve()
    failures: list[str] = []
    registry_outputs = _supplement_registry_outputs(root_path)
    _audit_key_and_output_coverage(
        "formal",
        formal_topology,
        EXPECTED_DELIVERABLES,
        failures,
    )
    _audit_key_and_output_coverage(
        "supplement",
        supplement_topology,
        registry_outputs,
        failures,
    )
    rows = []
    for family, topology in (
        ("formal", formal_topology),
        ("supplement", supplement_topology),
    ):
        for evidence_id, spec in sorted(topology.items()):
            row_failures = _audit_spec(root_path, family, evidence_id, spec)
            failures.extend(row_failures)
            output_path = str(spec.get("output_path") or "")
            rows.append(
                {
                    "family": family,
                    "evidence_id": evidence_id,
                    "output_path": output_path,
                    "output_present": bool(output_path and (root_path / output_path).is_file()),
                    "producer_script": spec.get("producer_script"),
                    "orchestrator_scripts": list(spec.get("orchestrator_scripts") or ()),
                    "upstream_families": list(spec.get("upstream_families") or ()),
                    "topology_status": "failed" if row_failures else "covered",
                }
            )
    return {
        "schema_version": 1,
        "analysis_type": "transformer_deep_dive_producer_topology_audit",
        "status": "failed" if failures else "completed",
        "formal_registered": len(EXPECTED_DELIVERABLES),
        "formal_covered": sum(
            row["family"] == "formal" and row["topology_status"] == "covered"
            for row in rows
        ),
        "supplements_registered": len(EXPECTED_SUPPLEMENT_IDS),
        "supplements_covered": sum(
            row["family"] == "supplement" and row["topology_status"] == "covered"
            for row in rows
        ),
        "queued_or_watched": sum(
            any(path != row["producer_script"] for path in row["orchestrator_scripts"])
            for row in rows
        ),
        "rows": rows,
        "failures": failures,
        "scientific_effect_values_read": False,
        "qrels_files_opened": False,
        "source_test_opened": False,
    }


def _audit_key_and_output_coverage(
    family: str,
    topology: Mapping[str, Mapping[str, Any]],
    expected_outputs: Mapping[str, str],
    failures: list[str],
) -> None:
    if set(topology) != set(expected_outputs):
        failures.append(f"{family} producer ID coverage drift")
    observed_outputs = {
        evidence_id: str(spec.get("output_path") or "")
        for evidence_id, spec in topology.items()
    }
    if observed_outputs != dict(expected_outputs):
        failures.append(f"{family} producer output-path coverage drift")
    output_values = list(observed_outputs.values())
    if len(output_values) != len(set(output_values)):
        failures.append(f"{family} producer outputs are not unique")


def _audit_spec(
    root: Path,
    family: str,
    evidence_id: str,
    spec: Mapping[str, Any],
) -> list[str]:
    failures = []
    required = {
        "output_path",
        "producer_script",
        "orchestrator_scripts",
        "upstream_families",
    }
    if set(spec) != required:
        return [f"{family} producer fields differ: {evidence_id}"]
    output = str(spec["output_path"])
    producer = str(spec["producer_script"])
    orchestrators = spec["orchestrator_scripts"]
    upstream = spec["upstream_families"]
    if not _safe_relative(output, prefix="runs/"):
        failures.append(f"unsafe producer output path: {family}.{evidence_id}")
    if not _safe_relative(producer, prefix="scripts/") or not producer.endswith(".py"):
        failures.append(f"invalid producer script path: {family}.{evidence_id}")
    elif not (root / producer).is_file():
        failures.append(f"missing producer script: {family}.{evidence_id}")
    if (
        not isinstance(orchestrators, tuple)
        or not orchestrators
        or len(orchestrators) != len(set(orchestrators))
    ):
        failures.append(f"invalid orchestrator coverage: {family}.{evidence_id}")
    else:
        for orchestrator in orchestrators:
            if not _safe_relative(str(orchestrator), prefix="scripts/"):
                failures.append(f"unsafe orchestrator path: {family}.{evidence_id}")
                continue
            orchestrator_path = root / str(orchestrator)
            if not orchestrator_path.is_file():
                failures.append(f"missing orchestrator script: {family}.{evidence_id}")
                continue
            if orchestrator not in {producer, GENERIC_WATCHER}:
                script_text = orchestrator_path.read_text(encoding="utf-8")
                if Path(producer).name not in script_text:
                    failures.append(
                        f"orchestrator does not bind producer: {family}.{evidence_id}"
                    )
    if (
        not isinstance(upstream, tuple)
        or not upstream
        or len(upstream) != len(set(upstream))
        or any(
            not isinstance(value, str)
            or not value
            or not value.replace("_", "a").isalnum()
            for value in upstream
        )
    ):
        failures.append(f"invalid upstream-family coverage: {family}.{evidence_id}")
    prohibited = ("source_test", "qrels_test", "records_test", "data/raw")
    serialized = " ".join(
        [
            output,
            producer,
            *(str(value) for value in orchestrators),
            *(str(value) for value in upstream),
        ]
    ).lower()
    if any(token in serialized for token in prohibited):
        failures.append(f"producer topology crosses data boundary: {family}.{evidence_id}")
    return failures


def _supplement_registry_outputs(root: Path) -> dict[str, str]:
    path = root / REGISTRY_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = payload.get("entries") if isinstance(payload, Mapping) else None
    if not isinstance(entries, list):
        raise ValueError("supplement registry entries missing for producer topology")
    return {
        str(entry["evidence_id"]): str(entry["path"])
        for entry in entries
        if isinstance(entry, Mapping)
    }


def _safe_relative(path: str, *, prefix: str) -> bool:
    value = PurePosixPath(path)
    return bool(
        path.startswith(prefix)
        and not value.is_absolute()
        and ".." not in value.parts
        and path == value.as_posix()
    )
