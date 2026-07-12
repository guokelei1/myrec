from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from train.real_data import (
    FEATURE_ROLES,
    assert_numeric_repair_lock_semantics,
    load_config,
    validate_execution_authority,
)
from train.real_data import read_json, sha256_file, write_json
from train.run_real_gate import (
    _begin_attempt,
    _record_pre_a_attempt_failure,
    run_train_variant,
)


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = SYSTEM_ROOT / "configs" / "c06_real_mechanism_gate.yaml"
PARENT_CONFIG_PATH = (
    SYSTEM_ROOT / "configs" / "c06_real_mechanism_gate_parent_v1.yaml"
)


def test_real_gate_config_freezes_cohorts_variants_and_gpu_mapping() -> None:
    config = load_config(CONFIG_PATH)
    assert {
        role: config["selection"][f"{role}_requests"]
        for role in ("fit", "internal_A", "internal_B", "escrow", "nohistory")
    } == {
        "fit": 12000,
        "internal_A": 1200,
        "internal_B": 600,
        "escrow": 515,
        "nohistory": 512,
    }
    assert tuple(config["training"]["variants"]) == (
        "local_hodge",
        "untrusted",
        "direct_learned",
        "centered_cross_attention",
    )
    assert config["resources"]["variant_physical_gpus"] == {
        "local_hodge": 0,
        "untrusted": 1,
        "direct_learned": 2,
        "centered_cross_attention": 3,
    }
    assert FEATURE_ROLES == ("fit", "internal_A", "nohistory")


@pytest.mark.parametrize(
    "stage,variant",
    [
        ("cohort_materialization", None),
        ("gpu_smoke", "local_hodge"),
        ("train_variants", "untrusted"),
        ("a0_a1_audit", None),
    ],
)
def test_every_data_bearing_stage_requires_its_explicit_authorization(
    stage: str, variant: str | None
) -> None:
    config = load_config(CONFIG_PATH)
    config["authorization"][stage] = False
    with pytest.raises(PermissionError):
        validate_execution_authority(
            config, stage=stage, variant=variant, device="cuda:0"
        )


def test_real_gate_has_no_dev_test_or_qrels_input_path() -> None:
    config = load_config(CONFIG_PATH)
    lowered = [str(value).lower() for value in config["paths"].values()]
    assert not any("qrels" in value for value in lowered)
    assert not any("records_dev" in value or "records_test" in value for value in lowered)
    assert config["authorization"]["dev"] is False
    assert config["authorization"]["test"] is False
    assert config["authorization"]["cohort_materialization"] is True
    assert config["authorization"]["gpu_smoke"] is True
    assert config["authorization"]["train_variants"] is True
    assert config["authorization"]["a0_a1_audit"] is True
    assert config["integrity"]["primary_dev_evaluator_calls_authorized"] == 0


def test_review1_preserves_byte_identical_parent_config() -> None:
    config = load_config(CONFIG_PATH)
    assert sha256_file(PARENT_CONFIG_PATH) == config["numeric_repair"][
        "parent_config_sha256"
    ]
    with PARENT_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        parent = yaml.safe_load(handle)
    assert parent["selection"] == config["selection"]
    assert parent["base"] == config["base"]
    assert parent["model"] == config["model"]
    assert parent["controls"] == config["controls"]
    assert parent["training"] == config["training"]
    assert parent["a0_gate"] == config["a0_gate"]
    assert parent["a1_gate"] == config["a1_gate"]


def test_centered_v1_is_hard_rejected_from_numeric_repair() -> None:
    with pytest.raises(PermissionError, match="centered v1 is immutable"):
        run_train_variant(
            CONFIG_PATH,
            "cuda:0",
            variant="centered_cross_attention",
        )


def test_numeric_repair_uses_distinct_ledger_and_preserves_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(CONFIG_PATH)
    monkeypatch.chdir(tmp_path)
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    parent_path = artifact_root / "formal_attempt_local_hodge.json"
    parent = {
        "candidate_id": "c06",
        "attempts": [
            {
                "attempt": 1,
                "stage": "started",
                "internal_A_features_scored": False,
                "internal_A_labels_opened": False,
            }
        ],
    }
    write_json(parent_path, parent)
    parent_hash = sha256_file(parent_path)
    config["paths"]["artifact_root"] = str(artifact_root)
    review_lock_path = tmp_path / "review1.json"
    write_json(
        review_lock_path,
        {
            "failed_variant_evidence": {
                "local_hodge": {"ledger_sha256": parent_hash}
            }
        },
    )
    config["paths"]["real_gate_lock"] = str(review_lock_path)
    context = _begin_attempt(
        config,
        CONFIG_PATH,
        "review1-hash",
        kind="variant_numeric_repair",
        variant="local_hodge",
    )
    assert sha256_file(parent_path) == parent_hash
    assert Path(context["ledger_path"]).name == (
        "formal_attempt_repair1_local_hodge.json"
    )
    retry = read_json(context["ledger_path"])
    assert retry["formal_attempt_number"] == 2
    assert retry["parent_attempt_ledger_sha256"] == parent_hash
    assert retry["attempts"][0]["repair_attempt"] == 1
    assert retry["attempts"][0]["run_id"].endswith(
        "local_hodge_repair1_s20260708"
    )


def test_pre_a_exception_is_persisted_before_reraise(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    metadata_path = tmp_path / "metadata.json"
    write_json(
        ledger_path,
        {
            "attempts": [
                {
                    "stage": "started",
                    "internal_A_features_scored": False,
                    "internal_A_labels_opened": False,
                }
            ]
        },
    )
    write_json(metadata_path, {"current_stage": "started"})
    _record_pre_a_attempt_failure(
        {"ledger_path": str(ledger_path), "metadata_path": str(metadata_path)},
        FloatingPointError("numeric guard"),
    )
    row = read_json(ledger_path)["attempts"][0]
    assert row["stage"] == "failed"
    assert row["exception_type"] == "FloatingPointError"
    assert row["internal_A_features_scored"] is False
    assert row["internal_A_labels_opened"] is False
    assert read_json(metadata_path)["current_stage"] == "failed"


def _valid_review1_semantics(config: dict) -> dict:
    repair = config["numeric_repair"]
    return {
        "declarations": {
            "numeric_implementation_failure_observed": True,
            "fit_training_telemetry_observed": True,
            "internal_A_features_scored_before_review1": False,
            "internal_A_labels_opened_before_review1": False,
            "internal_B_or_escrow_opened_before_review1": False,
            "dev_or_test_observed_before_review1": False,
            "internal_A_or_later_ranking_outcome_observed_before_review1": False,
            "thresholds_model_data_seed_or_training_changed": False,
            "repair_choice_used_A_or_comparative_ranking_quality": False,
        },
        "repair_scope": {
            "repair_id": repair["repair_id"],
            "eligible_variants": list(repair["eligible_variants"]),
            "repair_attempts_per_variant": 1,
            "centered_cross_attention_rerun": False,
            "fallback_complexity_per_row": "O(C*r)",
            "fallback_dtype": "float64",
            "fallback_requires_primitive_absolute_forward_error_identity": True,
        },
    }


def test_review1_semantics_reject_A_informed_repair_choice() -> None:
    config = load_config(CONFIG_PATH)
    lock = _valid_review1_semantics(config)
    assert_numeric_repair_lock_semantics(config, lock)
    lock["declarations"][
        "repair_choice_used_A_or_comparative_ranking_quality"
    ] = True
    with pytest.raises(ValueError, match="no-outcome declaration"):
        assert_numeric_repair_lock_semantics(config, lock)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("eligible_variants", ["local_hodge"]),
        ("repair_attempts_per_variant", 2),
        ("centered_cross_attention_rerun", True),
        ("fallback_dtype", "float32"),
        (
            "fallback_requires_primitive_absolute_forward_error_identity",
            False,
        ),
    ],
)
def test_review1_semantics_reject_scope_tampering(
    field: str, bad_value: object
) -> None:
    config = load_config(CONFIG_PATH)
    lock = _valid_review1_semantics(config)
    lock["repair_scope"][field] = bad_value
    with pytest.raises(ValueError, match="scope differs"):
        assert_numeric_repair_lock_semantics(config, lock)
