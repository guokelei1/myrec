#!/usr/bin/env python
"""Finalize the revised C5-R insight gate from the locked C3-R report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import write_json


SOURCE = Path("reports/pps_c3r_history_identity_control.json")
OUTPUT = Path("reports/pps_c5_insight_audit.json")
PROTOCOL = Path("doc/17_intro_motivation_repair_protocol.md")


def main() -> int:
    with SOURCE.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if source["status"] != "passed" or not all(source["gate_checks"].values()):
        raise ValueError("C3-R controls did not pass; C5-R cannot be finalized")
    summary = source["three_seed_summary"]
    report = {
        "report": "pps_c5_insight_audit",
        "date": "2026-07-10",
        "status": "passed",
        "gate_version": "C5-R",
        "protocol_amendment": {
            "type": "post-hoc claim replacement locked before new evaluation",
            "locked_scope": "observation, matched-history construction, decision rule, and permitted interaction-aware design transition",
            "design_mapping_provenance": "The query-anchored personalized-residual name was finalized after the gate passed. It is a constrained architecture hypothesis, not a pre-registered empirical outcome.",
            "path": str(PROTOCOL),
            "sha256": sha256_file(PROTOCOL),
            "executable_config": source["config_path"],
            "executable_config_sha256": source["config_sha256"],
        },
        "insight": {
            "name": "query-anchored personalized residual",
            "observation": (
                "Inside a query-conditioned candidate pool, query scores and the "
                "target user's prior behavior are complementary, but the behavioral "
                "gain is identity-specific and unavailable when history is absent."
            ),
            "architecture_consequence": (
                "Learn a query-candidate base and add a masked, candidate-specific "
                "residual from joint interactions with the target user's history."
            ),
            "falsification": (
                "Matched wrong-user history must erase the history contribution, the "
                "same-query subset must preserve the direction, and no-history records "
                "must be identical to query-only."
            ),
        },
        "evidence": {
            "b7_vs_b0b": source["existing_aggregate_comparisons"]["b7_vs_b0b"],
            "b7_vs_b2z": source["existing_aggregate_comparisons"]["b7_vs_b2z"],
            "history_present_true_minus_wrong_b7": summary[
                "true_minus_wrong_b7_history_present"
            ],
            "same_query_true_minus_wrong_b7": summary[
                "true_minus_wrong_b7_same_query"
            ],
            "same_query_requests": source["subset_counts"]["same_query_all_seeds"],
            "no_history_equivalence": source["no_history_equivalence"],
            "history_structure": source["history_structure"],
        },
        "historical_results_preserved": {
            "m3_m4": "construct-validity failed; not restored by C5-R",
            "insight_1_slot_complementarity": "retired untested after premise failure",
            "insight_2_consensus_law": "falsified; rho=-0.0110",
        },
        "claim_boundary": {
            "supported": [
                "aggregate query/history complementarity",
                "identity-specific predictive value of correct-user history",
                "exact query-only fallback when history is absent",
                "designing a query-anchored personalized-residual system",
            ],
            "not_supported": [
                "per-request fixed-channel oracle headroom",
                "learnable channel routing",
                "entropy-conditioned personalization",
                "slot-complementarity law",
                "deployed randomized causal effect",
            ],
        },
        "authorized_next_stage": (
            "Freeze and execute a proposed-system protocol for the query-anchored "
            "personalized residual."
        ),
        "confirmation_boundary": (
            "C5-R is an internal design gate on a previously analyzed dev split. "
            "The matched-control outcome was unseen when locked, but final paper "
            "confirmation still requires the untouched test or a secondary track "
            "after the system configuration is frozen."
        ),
        "source_report": str(SOURCE),
        "source_report_sha256": sha256_file(SOURCE),
        "test_split_read": False,
    }
    write_json(OUTPUT, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
