from __future__ import annotations

import hashlib
import json
from pathlib import Path

from myrec.mechanism.deep_dive_closeout_audit import audit_deep_dive_closeout


def test_closeout_audit_reports_missing_required_output_as_pending(tmp_path: Path):
    result = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={"attention": "runs/attention/metrics.json"},
        frozen_assets={},
    )
    assert result["status"] == "pending"
    assert result["pending"] == ["missing deliverable: attention"]
    assert result["source_test_content_read_by_this_audit"] is False


def test_closeout_audit_fails_closed_on_source_test_or_duplicate_json(tmp_path: Path):
    path = tmp_path / "runs/20260718_kuaisearch_mech_d3_bad/metadata.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        '{"status":"completed","source_test_opened":true,'
        '"source_test_opened":false}',
        encoding="utf-8",
    )
    result = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={},
        frozen_assets={},
    )
    assert result["status"] == "failed"
    assert any("duplicate JSON key" in value for value in result["failures"])


def test_closeout_audit_rejects_explicit_source_test_access(tmp_path: Path):
    path = tmp_path / "runs/20260718_kuaisearch_mech_d3_bad/metadata.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "source_test_opened": True,
                "records_path": "data/standardized/kuaisearch/records_test.jsonl",
            }
        ),
        encoding="utf-8",
    )
    result = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={},
        frozen_assets={},
    )
    assert result["status"] == "failed"
    assert any("source_test_opened is not false" in value for value in result["failures"])
    assert any("prohibited data declaration" in value for value in result["failures"])


def test_closeout_audit_accepts_completed_qrels_blind_deliverable(tmp_path: Path):
    path = tmp_path / "runs/d6/metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "completed",
                "qrels_read": False,
                "source_test_opened": False,
            }
        ),
        encoding="utf-8",
    )
    result = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={"readout": "runs/d6/metrics.json"},
        frozen_assets={},
    )
    assert result["status"] == "completed"
    assert result["failures"] == []


def test_closeout_audit_requires_exact_ledger_binding_for_dev_qrels(tmp_path: Path):
    path = tmp_path / "runs/d6/metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "analysis_run_id": "d6_eval",
                "status": "completed",
                "qrels_read": True,
                "source_test_opened": False,
            }
        ),
        encoding="utf-8",
    )
    missing = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={"readout": "runs/d6/metrics.json"},
        frozen_assets={},
    )
    assert missing["status"] == "failed"
    assert missing["failures"] == [
        "missing dev-eval ledger entry: d6_eval (reports/dev_eval_log.jsonl)"
    ]

    ledger = tmp_path / "reports/dev_eval_log.jsonl"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        json.dumps(
            {
                "run_id": "d6_eval",
                "metrics_path": "runs/d6/metrics.json",
                "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    completed = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={"readout": "runs/d6/metrics.json"},
        frozen_assets={},
    )
    assert completed["status"] == "completed"
    assert completed["failures"] == []


def test_closeout_audit_rejects_shrunken_registered_family(tmp_path: Path):
    path = tmp_path / "runs/d6/metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d6_q2_native_readout",
                "status": "completed",
                "qrels_read": False,
                "source_test_opened": False,
                "multiple_testing": {"family_size": 12},
                "family_rows": [{}] * 11,
            }
        ),
        encoding="utf-8",
    )
    result = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={
            "d6_q2_native_readout": "runs/d6/metrics.json"
        },
        frozen_assets={},
    )
    assert result["status"] == "failed"
    assert any("contract length mismatch" in value for value in result["failures"])


def test_closeout_audit_requires_the_frozen_rope_support_gate(tmp_path: Path):
    path = tmp_path / "runs/d5/metrics.json"
    path.parent.mkdir(parents=True)
    metrics = {
        "analysis_type": "transformer_deep_dive_d5_rope",
        "status": "completed",
        "qrels_read": False,
        "source_test_opened": False,
        "multiple_testing": {"family_size": 36},
        "family_rows": [{}] * 36,
        "implementation_digest": "fixed",
    }
    path.write_text(json.dumps(metrics), encoding="utf-8")
    missing = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={"d5_rope": "runs/d5/metrics.json"},
        frozen_assets={},
    )
    assert missing["status"] == "failed"
    assert any(
        "position_support_gate.active_contrast" in value
        for value in missing["failures"]
    )

    metrics["position_support_gate"] = {
        "active_contrast": "compression_minus_baseline",
        "active_endpoint": "ndcg@10",
        "active_ci95_equivalence_band": [-0.005, 0.005],
        "requires_compression_minus_expansion_bh_q_below_alpha_0p05": True,
        "requires_all_fold0_fold1_same_nonzero_direction": True,
        "active_contrast_is_confirmatory_family_member": False,
    }
    path.write_text(json.dumps(metrics), encoding="utf-8")
    completed = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={"d5_rope": "runs/d5/metrics.json"},
        frozen_assets={},
    )
    assert completed["status"] == "completed"
    assert completed["failures"] == []


def test_closeout_audit_rechecks_formal_run_contract_and_artifact_hash(tmp_path: Path):
    run_dir = tmp_path / "runs/20260718_kuaisearch_mech_d3_formal"
    run_dir.mkdir(parents=True)
    scores_path = run_dir / "scores.jsonl"
    scores_path.write_text('{"score":1}\n', encoding="utf-8")
    contract = {"implementation_digest": "runtime", "run_id": "formal"}
    contract_sha = hashlib.sha256(
        json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    metadata_path = run_dir / "metadata.json"
    metadata = {
        "status": "completed",
        "result_eligible": True,
        "source_test_opened": False,
        "qrels_read": False,
        "implementation_identity": {"digest": "runtime"},
        "run_contract": contract,
        "run_contract_sha256": contract_sha,
        "scores_sha256": hashlib.sha256(scores_path.read_bytes()).hexdigest(),
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    passed = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert passed["status"] == "completed"
    assert passed["formal_completed_integrity_checked"] == 1

    scores_path.write_text('{"score":2}\n', encoding="utf-8")
    failed = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert failed["status"] == "failed"
    assert any("artifact hash mismatch" in value for value in failed["failures"])


def test_closeout_audit_keeps_inflight_formal_run_pending(tmp_path: Path):
    run_dir = tmp_path / "runs/20260718_kuaisearch_mech_d3_formal"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "status": "wall_time_exhausted",
                "result_eligible": True,
                "source_test_opened": False,
                "qrels_read": False,
            }
        ),
        encoding="utf-8",
    )
    result = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert result["status"] == "pending"
    assert result["pending"] == [
        "in-flight formal run: "
        "runs/20260718_kuaisearch_mech_d3_formal/metadata.json "
        "(wall_time_exhausted)"
    ]


def test_closeout_audit_rejects_unbound_mechanical_terminal_run(tmp_path: Path):
    run_dir = tmp_path / "runs/20260718_kuaisearch_mech_d3_failed"
    run_dir.mkdir(parents=True)
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "mechanical_failure",
                "result_eligible": True,
                "source_test_opened": False,
                "qrels_read": False,
            }
        ),
        encoding="utf-8",
    )
    result = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert result["status"] == "failed"
    assert result["failures"] == [
        "mechanical terminal run lacks bound failure record: "
        "runs/20260718_kuaisearch_mech_d3_failed/metadata.json"
    ]


def test_closeout_audit_rejects_formal_run_implementation_contract_drift(tmp_path: Path):
    run_dir = tmp_path / "runs/20260718_kuaisearch_mech_d7_formal"
    run_dir.mkdir(parents=True)
    contract = {"implementation_digest": "drifted"}
    metadata = {
        "status": "completed",
        "result_eligible": True,
        "implementation_identity": {"digest": "runtime"},
        "run_contract": contract,
        "run_contract_sha256": hashlib.sha256(
            json.dumps(
                contract,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    result = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert result["status"] == "failed"
    assert any(
        "implementation/contract binding mismatch" in value
        for value in result["failures"]
    )


def test_closeout_audit_rejects_empty_required_lineage_field(tmp_path: Path):
    path = tmp_path / "runs/d3/metrics.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "analysis_type": "transformer_deep_dive_d3_attention_edges",
                "status": "completed",
                "qrels_read": False,
                "source_test_opened": False,
                "multiple_testing": {"family_size": 36},
                "family_rows": [{}] * 36,
                "implementation_digest": "",
            }
        ),
        encoding="utf-8",
    )
    result = audit_deep_dive_closeout(
        tmp_path,
        expected_deliverables={"d3_attention_edges": "runs/d3/metrics.json"},
        frozen_assets={},
    )
    assert result["status"] == "failed"
    assert any("contract empty field" in value for value in result["failures"])


def test_closeout_audit_binds_mechanical_failure_record_to_preserved_bytes(tmp_path: Path):
    run_dir = tmp_path / "runs/20260718_kuaisearch_mech_d4_failed"
    run_dir.mkdir(parents=True)
    metadata_path = run_dir / "metadata.json"
    progress_path = run_dir / "progress.json"
    partial_path = run_dir / "rows.partial.jsonl"
    metadata_path.write_text('{"status":"running"}\n', encoding="utf-8")
    progress_path.write_text('{"completed_rows":1}\n', encoding="utf-8")
    partial_path.write_text('{"row":1}\n', encoding="utf-8")
    record = {
        "analysis_type": "transformer_deep_dive_mechanical_failure_record",
        "run_id": run_dir.name,
        "status": "mechanical_failure",
        "result_eligible": False,
        "scientific_effect_values_read": False,
        "qrels_read": False,
        "source_test_opened": False,
        "preserved_inputs": {
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "progress_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
            "rows_partial_sha256": hashlib.sha256(partial_path.read_bytes()).hexdigest(),
        },
    }
    record_path = run_dir / "mechanical_failure_record.json"
    record_path.write_text(json.dumps(record), encoding="utf-8")
    passed = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert passed["status"] == "completed"
    assert len(passed["mechanical_failure_records"]) == 1

    partial_path.write_text('{"row":2}\n', encoding="utf-8")
    failed = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert failed["status"] == "failed"
    assert any("preserved hash mismatch" in value for value in failed["failures"])


def test_bound_mechanical_record_terminates_preserved_running_formal_metadata(
    tmp_path: Path,
):
    run_dir = tmp_path / "runs/20260718_kuaisearch_mech_d2_superseded"
    run_dir.mkdir(parents=True)
    metadata_path = run_dir / "metadata.json"
    progress_path = run_dir / "progress.json"
    partial_path = run_dir / "scores.partial.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "run_id": run_dir.name,
                "status": "running",
                "result_eligible": True,
                "source_test_opened": False,
                "qrels_read": False,
            }
        ),
        encoding="utf-8",
    )
    progress_path.write_text('{"completed_requests":1}\n', encoding="utf-8")
    partial_path.write_text('{"score":1}\n', encoding="utf-8")
    record = {
        "analysis_type": "transformer_deep_dive_mechanical_failure_record",
        "run_id": run_dir.name,
        "status": "mechanical_failure",
        "result_eligible": False,
        "scientific_effect_values_read": False,
        "qrels_read": False,
        "source_test_opened": False,
        "preserved_inputs": {
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
            "progress_sha256": hashlib.sha256(progress_path.read_bytes()).hexdigest(),
            "scores_partial_sha256": hashlib.sha256(partial_path.read_bytes()).hexdigest(),
        },
    }
    (run_dir / "mechanical_failure_record.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    passed = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert passed["status"] == "completed"
    assert passed["pending"] == []
    assert passed["run_declarations"] == [
        {
            "path": f"runs/{run_dir.name}/metadata.json",
            "run_id": run_dir.name,
            "analysis_stage": None,
            "method_id": None,
            "status": "mechanical_failure",
            "metadata_status": "running",
            "result_eligible": True,
            "command": None,
            "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        }
    ]


def test_mechanical_record_run_id_must_match_its_directory(tmp_path: Path):
    run_dir = tmp_path / "runs/20260718_kuaisearch_mech_d2_mismatch"
    run_dir.mkdir(parents=True)
    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text('{"status":"running"}\n', encoding="utf-8")
    record = {
        "analysis_type": "transformer_deep_dive_mechanical_failure_record",
        "run_id": "different-run-id",
        "status": "mechanical_failure",
        "result_eligible": False,
        "scientific_effect_values_read": False,
        "qrels_read": False,
        "source_test_opened": False,
        "preserved_inputs": {
            "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        },
    }
    (run_dir / "mechanical_failure_record.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    failed = audit_deep_dive_closeout(
        tmp_path, expected_deliverables={}, frozen_assets={}
    )
    assert failed["status"] == "failed"
    assert any("run-id/path mismatch" in value for value in failed["failures"])
