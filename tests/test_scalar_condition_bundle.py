from __future__ import annotations

import json

import pytest

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.scalar_condition_bundle import (
    append_scalar_request,
    audit_scalar_partial,
    finalize_scalar_bundle,
    prepare_scalar_bundle,
)


def _record() -> ModelRecord:
    return ModelRecord(
        request_id="r1",
        query="q",
        history=(),
        candidates=({"item_id": "a"}, {"item_id": "b"}),
    )


def test_scalar_bundle_roundtrip_and_identity_gate(tmp_path):
    records = [_record()]
    conditions = ("baseline", "identity", "active")
    prepared = prepare_scalar_bundle(
        tmp_path / "run",
        metadata={"run_contract_sha256": "x", "status": "initializing"},
        contract_sha256="x",
        records=records,
        conditions=conditions,
        resume=False,
    )
    rows = [
        {
            "request_id": "r1",
            "candidate_item_id": candidate,
            "candidate_ordinal": index,
            "conditions": {"baseline": 1.0, "identity": 1.0, "active": 2.0},
        }
        for index, candidate in enumerate(("a", "b"))
    ]
    import hashlib

    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    append_scalar_request(
        tmp_path / "run",
        {
            "ordinal": 0,
            "request_id": "r1",
            "rows": rows,
            "rows_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        },
        prepared,
    )
    observed = audit_scalar_partial(
        tmp_path / "run" / "scores.partial.jsonl", records, conditions
    )
    assert observed["completed_requests"] == 1
    assert observed["completed_score_rows"] == 2
    metadata = finalize_scalar_bundle(
        tmp_path / "run", prepared, records, conditions, maximum_identity_delta=0.0
    )
    assert metadata["status"] == "completed"
    assert metadata["complete_finite_score_coverage"] is True


def test_scalar_bundle_rejects_nonfinite_score(tmp_path):
    path = tmp_path / "bad.jsonl"
    rows = [
        {
            "request_id": "r1",
            "candidate_item_id": "a",
            "candidate_ordinal": 0,
            "conditions": {"a": float("nan")},
        },
        {
            "request_id": "r1",
            "candidate_item_id": "b",
            "candidate_ordinal": 1,
            "conditions": {"a": 1.0},
        },
    ]
    import hashlib

    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    path.write_text(
        json.dumps(
            {
                "ordinal": 0,
                "request_id": "r1",
                "rows": rows,
                "rows_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
            }
        )
        + "\n"
    )
    with pytest.raises(FloatingPointError):
        audit_scalar_partial(path, [_record()], ("a",))
