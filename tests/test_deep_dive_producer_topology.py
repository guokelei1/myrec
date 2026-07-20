from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from myrec.mechanism.deep_dive_closeout_audit import EXPECTED_DELIVERABLES
from myrec.mechanism.deep_dive_producer_topology import (
    FORMAL_PRODUCER_TOPOLOGY,
    SUPPLEMENT_PRODUCER_TOPOLOGY,
    audit_deep_dive_producer_topology,
)
from myrec.mechanism.supplemental_evidence_registry import (
    EXPECTED_SUPPLEMENT_IDS,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_producer_topology_exhaustively_covers_registered_artifacts():
    result = audit_deep_dive_producer_topology(ROOT)
    assert result["status"] == "completed"
    assert result["formal_registered"] == result["formal_covered"] == 19
    assert result["supplements_registered"] == result["supplements_covered"] == 21
    assert len(result["rows"]) == 40
    assert set(FORMAL_PRODUCER_TOPOLOGY) == set(EXPECTED_DELIVERABLES)
    assert set(SUPPLEMENT_PRODUCER_TOPOLOGY) == set(EXPECTED_SUPPLEMENT_IDS)
    assert result["failures"] == []
    assert result["scientific_effect_values_read"] is False
    assert result["qrels_files_opened"] is False
    assert result["source_test_opened"] is False


def test_output_path_drift_fails_closed():
    formal = deepcopy(FORMAL_PRODUCER_TOPOLOGY)
    formal["d2_postblock"]["output_path"] = "runs/drift/metrics.json"
    result = audit_deep_dive_producer_topology(ROOT, formal_topology=formal)
    assert result["status"] == "failed"
    assert "formal producer output-path coverage drift" in result["failures"]


def test_missing_producer_and_unbound_dedicated_orchestrator_fail_closed():
    formal = deepcopy(FORMAL_PRODUCER_TOPOLOGY)
    formal["d2_postblock"]["producer_script"] = "scripts/missing_producer.py"
    result = audit_deep_dive_producer_topology(ROOT, formal_topology=formal)
    assert result["status"] == "failed"
    assert "missing producer script: formal.d2_postblock" in result["failures"]
    assert "orchestrator does not bind producer: formal.d2_postblock" in result[
        "failures"
    ]


def test_topology_cannot_cross_source_test_boundary():
    supplements = deepcopy(SUPPLEMENT_PRODUCER_TOPOLOGY)
    supplements["d6_native_readout_diagnostics"]["upstream_families"] = (
        "source_test_scores",
    )
    result = audit_deep_dive_producer_topology(
        ROOT, supplement_topology=supplements
    )
    assert result["status"] == "failed"
    assert (
        "producer topology crosses data boundary: "
        "supplement.d6_native_readout_diagnostics"
    ) in result["failures"]
