from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmp.audit_mechanism_closeout_boundaries import (
    EXPECTED_PROBE_SHA256,
    audit_expected_completion,
    audit_mechanism_declarations,
    expected_formal_run_ids,
)


def _write_probe_metadata(root: Path, identity: object) -> None:
    path = root / "artifacts/motivation_m2/probes/q2/metadata.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps({"probe_manifest_sha256": identity}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "identity",
    [None, 7, "", "not-a-sha256", "0" * 64, EXPECTED_PROBE_SHA256.upper()],
)
def test_probe_manifest_identity_is_fail_closed(
    tmp_path: Path,
    identity: object,
) -> None:
    _write_probe_metadata(tmp_path, identity)

    check = audit_mechanism_declarations(tmp_path, manifest=None)

    assert check.status == "FAIL"
    assert any("probe-manifest lineage mismatch" in message for message in check.failures)


def test_probe_manifest_identity_requires_exact_expected_sha(tmp_path: Path) -> None:
    _write_probe_metadata(tmp_path, EXPECTED_PROBE_SHA256)

    check = audit_mechanism_declarations(tmp_path, manifest=None)

    assert check.status == "PASS"
    assert check.failures == []


@pytest.mark.parametrize(
    "payload",
    [
        '{"source_test_opened": true, "source_test_opened": false}',
        '{"source_test_opened": "yes"}',
        '{"sourceTestOpened": true}',
        '{"claim": "源测试集已经打开"}',
        '{"claim": "The held-out population was opened and used."}',
        '{"value": NaN}',
    ],
)
def test_boundary_declarations_reject_malformed_or_affirmative_access(
    tmp_path: Path, payload: str
) -> None:
    path = (
        tmp_path
        / "runs/20260717_kuaisearch_mech_m1_extra_analysis/metadata.json"
    )
    path.parent.mkdir(parents=True)
    path.write_text(payload, encoding="utf-8")

    check = audit_mechanism_declarations(tmp_path, manifest=None)

    assert check.status == "FAIL"


def test_unregistered_formal_run_is_outcome_selection_failure(tmp_path: Path) -> None:
    path = tmp_path / "runs/20260717_kuaisearch_mech_m9_favorable_analysis"
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(
        json.dumps(
            {
                "analysis_run_id": path.name,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    check = audit_expected_completion(tmp_path)

    assert check.status == "FAIL"
    assert any("outcome-selection ambiguity" in value for value in check.failures)


def test_matched_did_is_a_registered_internal_dev_formal_run() -> None:
    expected = expected_formal_run_ids()
    assert len(expected) == 88
    assert expected[
        "20260717_kuaisearch_mech_m3_q2_matched_did_analysis"
    ] == "internal_dev"
