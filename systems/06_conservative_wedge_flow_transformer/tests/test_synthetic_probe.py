from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from experiments.run_synthetic_mechanism_probe import (  # noqa: E402
    CONFIG_REL,
    OUTPUT_REL,
    _write_json_exclusive,
    _row_spearman,
    build_pre_run_manifest,
    compute_gate_scores,
    generate_synthetic_batch,
    paired_bootstrap_many,
    request_binary_ndcg,
    request_pairwise_accuracy,
    resolve_fixed_cli_path,
    validate_frozen_config,
    verify_preoutcome_lock,
)


def test_frozen_formal_config_is_well_formed_without_running_it() -> None:
    path = SYSTEM_ROOT / "configs" / "c06_synthetic_mechanism_probe.yaml"
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_frozen_config(config)
    assert config["generation"]["requests_per_seed"] == 4096
    assert config["execution"]["repository_data_access"] == "forbidden"
    manifest = build_pre_run_manifest()
    assert manifest["files"][CONFIG_REL]


def test_tiny_generator_has_strict_cycle_and_shared_world_marginals() -> None:
    batch = generate_synthetic_batch(
        np.random.default_rng(701),
        request_count=8,
        candidates=6,
        history_events=3,
    )
    cycle = batch["cycle"]
    assert cycle.dtype == np.float64
    assert np.max(np.abs(cycle + np.swapaxes(cycle, -1, -2))) <= 1.0e-12
    assert np.max(np.abs(cycle.sum(axis=-1))) <= 1.0e-12

    energy = batch["normalized_cycle_energy"].reshape(8, -1)
    variances = {
        name: world["noise_variance"].reshape(8, -1)
        for name, world in batch["worlds"].items()
    }
    reference = np.sort(variances["reliability_aligned"], axis=1)
    assert np.array_equal(
        reference, np.sort(variances["reliability_decoupled"], axis=1)
    )
    assert np.array_equal(
        reference, np.sort(variances["reliability_adversarial"], axis=1)
    )
    assert np.allclose(
        _row_spearman(energy, variances["reliability_aligned"]), 1.0
    )
    assert np.allclose(
        _row_spearman(energy, variances["reliability_adversarial"]), -1.0
    )

    for world in batch["worlds"].values():
        observed = world["observed_potential"]
        flow = observed[..., :, None] - observed[..., None, :] + cycle
        assert np.max(np.abs(flow.mean(axis=-1) - observed)) <= 1.0e-12
        scores, diagnostics = compute_gate_scores(
            observed, batch["cycle_energy"], world["noise_variance"]
        )
        assert set(scores) == {
            "local_hodge",
            "global_event",
            "t_one",
            "direct_reliability_oracle",
        }
        assert all(value.shape == (8, 6) for value in scores.values())
        assert all(np.isfinite(value).all() for value in scores.values())
        assert np.all((diagnostics["local_hodge"] >= 0.0))
        assert np.all((diagnostics["local_hodge"] <= 1.0))


def test_tiny_metrics_and_bootstrap_are_deterministic() -> None:
    target = np.asarray(
        [[3.0, 2.0, 1.0, 0.0], [0.0, 2.0, 3.0, 1.0]], dtype=np.float64
    )
    perfect = request_pairwise_accuracy(
        target,
        target,
        true_tie_tolerance=1.0e-10,
        predicted_tie_credit=0.5,
    )
    assert np.array_equal(perfect, np.ones(2))
    ndcg = request_binary_ndcg(
        target,
        target,
        ["tiny_0", "tiny_1"],
        relevant_candidates=2,
        cutoff=4,
        tie_break_salt="20260708",
    )
    assert np.allclose(ndcg, 1.0)

    differences = {
        "a": np.asarray([0.1, 0.2, -0.1, 0.3]),
        "b": np.asarray([-0.2, 0.0, 0.2, 0.1]),
    }
    first = paired_bootstrap_many(
        differences, samples=64, seed=20260714, confidence=0.95, chunk_size=8
    )
    second = paired_bootstrap_many(
        differences, samples=64, seed=20260714, confidence=0.95, chunk_size=8
    )
    assert first == second


def test_preflight_helpers_fail_closed_on_paths_lock_and_output(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="exact repo-relative path"):
        resolve_fixed_cli_path("./" + CONFIG_REL, CONFIG_REL)
    with pytest.raises(ValueError, match="exact repo-relative path"):
        resolve_fixed_cli_path(str(tmp_path / "config.yaml"), CONFIG_REL)

    missing_lock = tmp_path / "missing_lock.json"
    with pytest.raises(FileNotFoundError, match="pre-outcome lock is missing"):
        verify_preoutcome_lock(
            missing_lock,
            {"files": {}, "combined_sha256": "not-used"},
            probe_id="c06_local_hodge_bidirectional_synthetic_v1",
        )

    existing_output = tmp_path / "report.json"
    existing_output.write_text("user-owned\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_exclusive(existing_output, {"should": "not replace"})
    assert existing_output.read_text(encoding="utf-8") == "user-owned\n"
    assert OUTPUT_REL.endswith("synthetic_v1/report.json")
