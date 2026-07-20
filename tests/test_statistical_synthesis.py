from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from myrec.mechanism.statistical_synthesis import (
    BOOTSTRAP_CLUSTER,
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
    ENDPOINT_SPECS,
    PROBE_MANIFEST_PATH,
    PROBE_MANIFEST_SHA256,
    STRICT_TRANSFER_SURFACE,
    benjamini_hochberg,
    cluster_bootstrap_draws,
    direction_consistent,
    load_analysis_artifacts,
    load_family_registration,
    normalized_query_fold,
    percentile_ci,
    summarize_analysis,
    summarize_two_folds,
    synthesize_mechanism_statistics,
    two_sided_bootstrap_p,
)


def _cluster_for_fold(fold: int) -> str:
    for index in range(10_000):
        cluster = f"query{index}"
        if normalized_query_fold(cluster) == fold:
            return cluster
    raise AssertionError("unable to find fold fixture")


def _row(
    request_id: str,
    cluster: str,
    ndcg: float,
    margin: float | None,
    *,
    treatment: str = "treatment",
    control: str = "control",
) -> dict:
    return {
        "control_condition_id": control,
        "margin_eligible": margin is not None,
        "normalized_query_cluster": cluster,
        "request_id": request_id,
        "target_aware_surface": STRICT_TRANSFER_SURFACE,
        "target_margin_change": margin,
        "treatment_condition_id": treatment,
        "treatment_minus_control_ndcg@10": ndcg,
    }


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_two_fold_statistics_are_hand_computed_and_cluster_kept_intact() -> None:
    fold0 = _cluster_for_fold(0)
    fold1 = _cluster_for_fold(1)
    rows = [
        _row("a", fold0, 1.0, 2.0),
        _row("b", fold0, 3.0, None),
        _row("c", fold1, -2.0, -4.0),
    ]
    result = summarize_two_folds(rows)
    assert result["0"]["num_requests"] == 2
    assert result["0"]["num_query_clusters"] == 1
    assert result["0"]["endpoints"]["strict_transfer_ndcg_delta"] == {
        "mean": 2.0,
        "num_query_clusters": 1,
        "num_requests": 2,
    }
    assert result["0"]["endpoints"]["strict_transfer_target_margin_delta"] == {
        "mean": 2.0,
        "num_query_clusters": 1,
        "num_requests": 1,
    }
    assert result["1"]["endpoints"]["strict_transfer_ndcg_delta"]["mean"] == -2.0
    assert result["1"]["endpoints"]["strict_transfer_target_margin_delta"]["mean"] == -4.0


def test_cluster_bootstrap_draws_are_request_weighted_and_deterministic() -> None:
    rows = [
        _row("a1", "a", 1.0, 10.0),
        _row("a2", "a", 3.0, 30.0),
        _row("b1", "b", 9.0, 90.0),
    ]
    first = cluster_bootstrap_draws(rows, samples=3, seed=7)
    second = cluster_bootstrap_draws(rows, samples=3, seed=7)
    assert first == second
    # random.Random(7) selects cluster indexes [1,0], [1,0], [0,0].
    assert first["strict_transfer_ndcg_delta"] == pytest.approx(
        [13.0 / 3.0, 13.0 / 3.0, 2.0]
    )
    assert first["strict_transfer_target_margin_delta"] == pytest.approx(
        [130.0 / 3.0, 130.0 / 3.0, 20.0]
    )


def test_percentile_and_zero_inclusive_bootstrap_p_are_hand_computed() -> None:
    assert percentile_ci(list(range(100))) == [2.0, 97.0]
    result = two_sided_bootstrap_p([-2.0, -1.0, 0.0, 1.0, 2.0], 1.0)
    assert result["opposite_direction_or_zero_comparison"] == "draw<=0"
    assert result["opposite_direction_or_zero_draws"] == 3
    assert result["lower_inclusive_zero_draws"] == 3
    assert result["upper_inclusive_zero_draws"] == 3
    assert result["one_sided_corrected_tail"] == pytest.approx(4.0 / 6.0)
    assert result["two_sided_p"] == 1.0
    zero = two_sided_bootstrap_p([1.0, 2.0], 0.0)
    assert zero["opposite_direction_or_zero_draws"] == 2
    assert zero["two_sided_p"] == 1.0


def test_bh_q_values_are_hand_computed_and_all_hypotheses_retained() -> None:
    rows = [
        {"hypothesis_id": "a", "raw_p": 0.01},
        {"hypothesis_id": "b", "raw_p": 0.04},
        {"hypothesis_id": "c", "raw_p": 0.03},
        {"hypothesis_id": "d", "raw_p": 0.20},
    ]
    result = benjamini_hochberg(rows)
    assert [row["hypothesis_id"] for row in result] == ["a", "b", "c", "d"]
    assert [row["q_value"] for row in result] == pytest.approx(
        [0.04, 0.04 * 4.0 / 3.0, 0.04 * 4.0 / 3.0, 0.20]
    )
    assert [row["reject_at_0_05"] for row in result] == [True, False, False, False]


@pytest.mark.parametrize(
    ("overall", "folds", "expected"),
    [
        (1.0, {"0": 2.0, "1": 3.0}, True),
        (-1.0, {"0": -2.0, "1": -3.0}, True),
        (1.0, {"0": 2.0, "1": -3.0}, False),
        (1.0, {"0": 0.0, "1": 3.0}, False),
        (1.0, {"0": None, "1": 3.0}, False),
        (0.0, {"0": 2.0, "1": 3.0}, False),
    ],
)
def test_direction_consistency_requires_three_nonzero_matching_signs(
    overall: float,
    folds: dict[str, float | None],
    expected: bool,
) -> None:
    assert direction_consistent(overall, folds) is expected


def _probe_admission() -> dict:
    return {
        "actual_path": "/frozen/repository/experiments/motivation/probe_manifest.yaml",
        "actual_sha256": PROBE_MANIFEST_SHA256,
        "expected_path": PROBE_MANIFEST_PATH.as_posix(),
        "expected_sha256": PROBE_MANIFEST_SHA256,
        "pair_contains_mechanism_intervention": True,
        "runs": {},
    }


def _make_analysis(
    root: Path,
    *,
    method_id: str,
    treatment: str,
    control: str,
    ordinal: int,
    checkpoint_id: str | None = None,
    score_namespace: str | None = None,
) -> Path:
    analysis_dir = root / f"analysis-{ordinal:03d}"
    fold0 = _cluster_for_fold(0)
    fold1 = _cluster_for_fold(1)
    rows = [
        _row(f"r{ordinal}-0a", fold0, 0.1, 1.0, treatment=treatment, control=control),
        _row(f"r{ordinal}-0b", fold0, 0.3, 3.0, treatment=treatment, control=control),
        _row(f"r{ordinal}-1a", fold1, 0.2, 2.0, treatment=treatment, control=control),
        _row(f"r{ordinal}-1b", fold1, 0.4, 4.0, treatment=treatment, control=control),
    ]
    draws = cluster_bootstrap_draws(
        rows,
        samples=BOOTSTRAP_SAMPLES,
        seed=BOOTSTRAP_SEED,
    )
    means = {
        endpoint.endpoint_id: sum(float(row[endpoint.row_key]) for row in rows)
        / len(rows)
        for endpoint in ENDPOINT_SPECS
    }
    intervals = {
        endpoint.row_key: percentile_ci(draws[endpoint.endpoint_id])
        for endpoint in ENDPOINT_SPECS
    }
    strict = {
        "mean_target_margin_change": means["strict_transfer_target_margin_delta"],
        "mean_treatment_minus_control_ndcg@10": means["strict_transfer_ndcg_delta"],
        "num_margin_eligible_requests": 4,
        "num_query_clusters": 2,
        "num_requests": 4,
        "query_cluster_ci95": intervals,
        "target_margin_change": {
            "mean": means["strict_transfer_target_margin_delta"],
            "query_cluster_ci95": intervals["target_margin_change"],
        },
        "treatment_minus_control_ndcg@10": {
            "mean": means["strict_transfer_ndcg_delta"],
            "query_cluster_ci95": intervals["treatment_minus_control_ndcg@10"],
        },
    }
    admission = _probe_admission()
    analysis_run_id = f"analysis-{ordinal:03d}"
    checkpoint_id = checkpoint_id or f"{method_id}@fixture-checkpoint"
    score_namespace = score_namespace or method_id
    base_scoring_signature = {
        "checkpoint_id": checkpoint_id,
        "feature_contract": "fixture-visible-fields-v1",
        "probe": "fixture-paired-probe",
    }
    score_identities = {}
    for role, condition_id in (("treatment", treatment), ("control", control)):
        run_id = f"{score_namespace}-{condition_id}-score"
        score_identities[role] = {
            "condition_id": condition_id,
            "metadata_sha256": hashlib.sha256(
                f"metadata:{run_id}".encode("utf-8")
            ).hexdigest(),
            "qrels_read": False,
            "run_id": run_id,
            "scores_sha256": hashlib.sha256(
                f"scores:{run_id}".encode("utf-8")
            ).hexdigest(),
        }
    metrics = {
        "analysis_run_id": analysis_run_id,
        "analysis_type": "motivation_mechanism_paired_probe",
        "bootstrap": {
            "cluster": BOOTSTRAP_CLUSTER,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": BOOTSTRAP_SEED,
        },
        "checkpoint_id": checkpoint_id,
        "control_condition_id": control,
        "label_mode": "graded",
        "mechanism_probe_manifest_admission": admission,
        "method_id": method_id,
        "num_requests": 4,
        "split": "dev",
        "surfaces": {STRICT_TRANSFER_SURFACE: strict},
        "treatment_condition_id": treatment,
    }
    audit = {
        "analysis_type": "motivation_mechanism_pre_qrels_score_audit",
        "checks": {"complete_finite_score_coverage": True},
        "input_runs": score_identities,
        "invariants": {
            "base_scoring_signature": base_scoring_signature,
            "checkpoint_id": checkpoint_id,
            "method_id": method_id,
        },
        "mechanism_probe_manifest_admission": admission,
        "num_requests": 4,
        "qrels_read": False,
        "split": "dev",
        "status": "passed",
    }
    _write_json(analysis_dir / "pre_qrels_audit.json", audit)
    pre_audit_sha256 = hashlib.sha256(
        (analysis_dir / "pre_qrels_audit.json").read_bytes()
    ).hexdigest()
    metadata = {
        "analysis_run_id": analysis_run_id,
        "analysis_type": "motivation_mechanism_paired_probe",
        "bootstrap": metrics["bootstrap"],
        "conditions": {
            "control": {
                "condition_id": control,
                "run_id": score_identities["control"]["run_id"],
            },
            "treatment": {
                "condition_id": treatment,
                "run_id": score_identities["treatment"]["run_id"],
            },
        },
        "invariants": {
            "base_scoring_signature": base_scoring_signature,
            "checkpoint_id": checkpoint_id,
            "method_id": method_id,
        },
        "label_mode": "graded",
        "mechanism_probe_manifest_admission": admission,
        "pre_qrels_audit_sha256": pre_audit_sha256,
        "split": "dev",
    }
    _write_json(analysis_dir / "metrics.json", metrics)
    _write_json(analysis_dir / "metadata.json", metadata)
    _write_jsonl(analysis_dir / "per_request.jsonl", rows)
    # A malformed forbidden file makes any accidental external-label read fail.
    (analysis_dir / "qrels_dev.jsonl").write_text("not-json\n", encoding="utf-8")
    return analysis_dir


def test_complete_family_synthesis_is_qrels_free_fdr_complete_and_sha_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registration = load_family_registration("m1_input_interventions")
    analysis_dirs = []
    ordinal = 0
    for method_id in registration.models:
        for treatment, control in registration.comparisons:
            analysis_dirs.append(
                _make_analysis(
                    tmp_path,
                    method_id=method_id,
                    treatment=treatment,
                    control=control,
                    ordinal=ordinal,
                )
            )
            ordinal += 1
    original_open = Path.open

    def guarded_open(path: Path, *args: object, **kwargs: object):
        if path.name in {"qrels_dev.jsonl", "records_dev.jsonl", "qrels_test.jsonl"}:
            raise AssertionError(f"forbidden input opened: {path.name}")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    output_path = tmp_path / "synthesis.json"
    report = synthesize_mechanism_statistics(
        family="m1_input_interventions",
        analysis_dirs=list(reversed(analysis_dirs)),
        output_path=output_path,
        command=["unit-test"],
    )
    assert output_path.is_file()
    assert report["probe_manifest"]["sha256"] == PROBE_MANIFEST_SHA256
    assert len(report["analyses"]) == 24
    assert report["fdr"]["family_size"] == 48
    assert report["fdr"]["missing_hypothesis_ids"] == []
    assert report["fdr"]["unexpected_hypothesis_ids"] == []
    assert len(report["fdr"]["registered_hypothesis_ids"]) == 48
    assert len(set(report["fdr"]["registered_hypothesis_ids"])) == 48
    assert all(row["reject_at_0_05"] for row in report["fdr"]["results"])
    assert all(
        analysis["strict_transfer"]["endpoints"][endpoint.endpoint_id][
            "direction_consistent"
        ]
        for analysis in report["analyses"]
        for endpoint in registration.endpoints
    )
    assert len(report["input_analyses"]) == 24
    assert all(
        set(value["files"]) == {
            "metrics.json",
            "per_request.jsonl",
            "metadata.json",
            "pre_qrels_audit.json",
        }
        for value in report["input_analyses"]
    )
    persisted = json.loads(output_path.read_text(encoding="utf-8"))
    assert persisted["input_set_sha256"] == report["input_set_sha256"]
    assert persisted["fdr"] == report["fdr"]


def test_analysis_summary_rejects_metrics_point_or_ci_drift(tmp_path: Path) -> None:
    registration = load_family_registration("m1_input_interventions")
    method_id = registration.models[0]
    treatment, control = registration.comparisons[0]
    analysis_dir = _make_analysis(
        tmp_path,
        method_id=method_id,
        treatment=treatment,
        control=control,
        ordinal=0,
    )
    metrics_path = analysis_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["surfaces"][STRICT_TRANSFER_SURFACE][
        "mean_treatment_minus_control_ndcg@10"
    ] += 0.01
    _write_json(metrics_path, metrics)
    with pytest.raises(ValueError, match="metrics mean drift"):
        summarize_analysis(load_analysis_artifacts(analysis_dir), registration)


def test_family_coverage_rejects_missing_registered_analyses(tmp_path: Path) -> None:
    registration = load_family_registration("m1_input_interventions")
    method_id = registration.models[0]
    treatment, control = registration.comparisons[0]
    only = _make_analysis(
        tmp_path,
        method_id=method_id,
        treatment=treatment,
        control=control,
        ordinal=0,
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        synthesize_mechanism_statistics(
            family="m1_input_interventions",
            analysis_dirs=[only],
            output_path=tmp_path / "must-not-exist.json",
            command=["unit-test"],
        )
    assert not (tmp_path / "must-not-exist.json").exists()


def _make_m0_family(
    root: Path,
    *,
    label_checkpoint_id: str = "m0-probe@label-shuffle",
    label_score_namespace: str = "label-shuffle",
    label_method_id: str = "m0_bge_pairwise_transfer_probe",
) -> tuple[list[Path], object]:
    registration = load_family_registration("m0_recoverability")
    result = []
    for ordinal, cell in enumerate(registration.cells):
        is_label = cell.variant_id == "within_request_label_shuffle"
        result.append(
            _make_analysis(
                root,
                method_id=(
                    label_method_id
                    if is_label
                    else "m0_bge_pairwise_transfer_probe"
                ),
                treatment=cell.treatment_condition_id,
                control=cell.control_condition_id,
                ordinal=ordinal,
                checkpoint_id=(
                    label_checkpoint_id if is_label else "m0-probe@real"
                ),
                score_namespace=(
                    label_score_namespace if is_label else "m0-real"
                ),
            )
        )
    return result, registration


def test_m0_registration_is_derived_from_frozen_conditions_and_negative_control() -> None:
    registration = load_family_registration("m0_recoverability")
    assert registration.models == ()
    assert registration.source_registration == {
        "conditions": [
            "full",
            "null",
            "history_shuffle",
            "routing_query_shuffle",
        ],
        "null_condition_yaml_value_normalized_from_none": True,
        "separately_fitted_negative_control": "within_request_label_shuffle",
    }
    assert len(registration.cells) == 6
    assert [cell.cell_id for cell in registration.cells] == [
        "real__full__vs__null",
        "real__history_shuffle__vs__null",
        "real__routing_query_shuffle__vs__null",
        "within_request_label_shuffle__full__vs__null",
        "real__full__vs__history_shuffle",
        "real__full__vs__routing_query_shuffle",
    ]
    assert len(registration.endpoints) == 2


def test_complete_m0_family_has_six_identity_gated_cells_and_12_hypotheses(
    tmp_path: Path,
) -> None:
    analysis_dirs, registration = _make_m0_family(tmp_path)
    report = synthesize_mechanism_statistics(
        family="m0_recoverability",
        analysis_dirs=list(reversed(analysis_dirs)),
        output_path=tmp_path / "m0-synthesis.json",
        command=["unit-test"],
    )
    assert len(report["analyses"]) == 6
    assert report["fdr"]["family_size"] == 12
    assert report["fdr"]["missing_hypothesis_ids"] == []
    assert len(set(report["fdr"]["registered_hypothesis_ids"])) == 12
    assert [row["registered_cell_id"] for row in report["analyses"]] == [
        cell.cell_id for cell in registration.cells
    ]
    gate = report["family_identity_gate"]
    assert gate["kind"] == (
        "m0_real_vs_separately_fitted_negative_control_identity_gate"
    )
    assert gate["real"]["checkpoint_id"] == "m0-probe@real"
    assert gate["label_shuffle"]["checkpoint_id"] == "m0-probe@label-shuffle"
    assert all(gate["checks"].values())
    assert report["registration"]["source_registration"][
        "separately_fitted_negative_control"
    ] == "within_request_label_shuffle"


def test_m0_rejects_condition_identical_label_control_with_real_checkpoint(
    tmp_path: Path,
) -> None:
    analysis_dirs, _ = _make_m0_family(
        tmp_path,
        label_checkpoint_id="m0-probe@real",
    )
    with pytest.raises(ValueError, match="checkpoint identity is ambiguous"):
        synthesize_mechanism_statistics(
            family="m0_recoverability",
            analysis_dirs=analysis_dirs,
            output_path=tmp_path / "must-not-exist.json",
            command=["unit-test"],
        )


def test_m0_rejects_label_control_score_identifiers_reused_from_real(
    tmp_path: Path,
) -> None:
    analysis_dirs, _ = _make_m0_family(
        tmp_path,
        label_score_namespace="m0-real",
    )
    with pytest.raises(ValueError, match="score identity overlaps real controls"):
        synthesize_mechanism_statistics(
            family="m0_recoverability",
            analysis_dirs=analysis_dirs,
            output_path=tmp_path / "must-not-exist.json",
            command=["unit-test"],
        )


def test_m0_rejects_label_control_method_identity_drift(tmp_path: Path) -> None:
    analysis_dirs, _ = _make_m0_family(
        tmp_path,
        label_method_id="different_probe_method",
    )
    with pytest.raises(ValueError, match="method identity is not common"):
        synthesize_mechanism_statistics(
            family="m0_recoverability",
            analysis_dirs=analysis_dirs,
            output_path=tmp_path / "must-not-exist.json",
            command=["unit-test"],
        )
