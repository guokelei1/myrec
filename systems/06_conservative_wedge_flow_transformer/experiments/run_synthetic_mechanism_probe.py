#!/usr/bin/env python3
"""Run the locked, data-free C06 bidirectional mechanism probe.

This program reads only its YAML contract and candidate-local source/protocol
files for hashing.  It does not read repository datasets, qrels, model weights,
dev/test records, or prior experiment outcomes.  Importing this module performs
no experiment; the formal 3 x 4096 run occurs only through ``main``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml


SCRIPT_PATH = Path(__file__).resolve()
C06_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[3]
CONFIG_REL = "systems/06_conservative_wedge_flow_transformer/configs/c06_synthetic_mechanism_probe.yaml"
SCRIPT_REL = "systems/06_conservative_wedge_flow_transformer/experiments/run_synthetic_mechanism_probe.py"
PROTOCOL_REL = "systems/06_conservative_wedge_flow_transformer/notes/synthetic_mechanism_probe_protocol.md"
TEST_REL = "systems/06_conservative_wedge_flow_transformer/tests/test_synthetic_probe.py"
LOCK_REL = "systems/06_conservative_wedge_flow_transformer/notes/c06_synthetic_probe_lock.json"
OUTPUT_REL = "artifacts/c06_conservative_wedge_flow_transformer/synthetic_v1/report.json"
LOCK_ID = "c06_synthetic_probe_preoutcome_v1"
LOCK_STATUS = "locked_before_synthetic_outcome"
MANIFEST_RELATIVE_PATHS = (CONFIG_REL, SCRIPT_REL, PROTOCOL_REL, TEST_REL)


WORLD_NAMES = (
    "reliability_aligned",
    "reliability_decoupled",
    "reliability_adversarial",
)
GATE_NAMES = (
    "local_hodge",
    "global_event",
    "t_one",
    "direct_reliability_oracle",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_exclusive(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically publish a new report and never replace an existing path."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + f".tmp.{os.getpid()}")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {destination}")
    created_by_this_process = False
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            created_by_this_process = True
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        # A same-filesystem hard link is atomic and fails if another process
        # created the destination after the pre-run existence check.
        os.link(temporary, destination)
    finally:
        if created_by_this_process and temporary.exists():
            temporary.unlink()


def _repo_path(relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"path is not a safe repo-relative path: {relative!r}")
    resolved = (REPO_ROOT / raw).resolve()
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise ValueError(f"path resolves outside repository: {relative!r}") from error
    return resolved


def resolve_fixed_cli_path(raw: str | Path, expected_relative: str) -> Path:
    """Interpret CLI paths from repo root, never from the current directory."""

    raw_string = os.fspath(raw)
    supplied = Path(raw_string)
    if supplied.is_absolute() or raw_string != expected_relative:
        raise ValueError(
            f"expected exact repo-relative path {expected_relative!r}; got {raw_string!r}"
        )
    return _repo_path(expected_relative)


def build_pre_run_manifest() -> dict[str, Any]:
    files = {relative: sha256_file(_repo_path(relative)) for relative in MANIFEST_RELATIVE_PATHS}
    combined = hashlib.sha256()
    for relative, digest in sorted(files.items()):
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(digest.encode("ascii"))
        combined.update(b"\n")
    return {
        "files": files,
        "config_sha256": files[CONFIG_REL],
        "source_sha256": files[SCRIPT_REL],
        "protocol_sha256": files[PROTOCOL_REL],
        "test_sha256": files[TEST_REL],
        "combined_sha256": combined.hexdigest(),
    }


def verify_preoutcome_lock(
    lock_path: Path,
    manifest: dict[str, Any],
    *,
    probe_id: str,
) -> dict[str, Any]:
    if not lock_path.is_file():
        raise FileNotFoundError(
            f"independent pre-outcome lock is missing; formal probe is forbidden: {lock_path}"
        )
    with lock_path.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    expected_scalars = {
        "lock_id": LOCK_ID,
        "probe_id": probe_id,
        "status": LOCK_STATUS,
        "outcomes_observed_before_lock": False,
        "output_path": OUTPUT_REL,
        "combined_sha256": manifest["combined_sha256"],
    }
    for key, expected in expected_scalars.items():
        if lock.get(key) != expected:
            raise ValueError(
                f"pre-outcome lock mismatch for {key}: {lock.get(key)!r} != {expected!r}"
            )
    if lock.get("files") != manifest["files"]:
        raise ValueError("pre-outcome lock file hashes do not match the pre-run manifest")
    return {
        "path": LOCK_REL,
        "sha256": sha256_file(lock_path),
        "lock_id": LOCK_ID,
        "status": LOCK_STATUS,
    }


def _nested(config: dict[str, Any], *keys: str) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing frozen configuration key: {'.'.join(keys)}")
        value = value[key]
    return value


def validate_frozen_config(config: dict[str, Any]) -> None:
    """Reject silent changes to the pre-outcome synthetic contract."""

    exact: dict[tuple[str, ...], Any] = {
        ("candidate_id",): "c06",
        ("probe_id",): "c06_local_hodge_bidirectional_synthetic_v1",
        ("status",): "ready_to_lock_not_executed",
        ("paths", "config"): CONFIG_REL,
        ("paths", "script"): SCRIPT_REL,
        ("paths", "protocol"): PROTOCOL_REL,
        ("paths", "test"): TEST_REL,
        ("paths", "lock"): LOCK_REL,
        ("paths", "output"): OUTPUT_REL,
        ("execution", "device"): "cpu",
        ("execution", "dtype"): "float64",
        ("execution", "training"): False,
        ("execution", "repository_data_access"): "forbidden",
        ("execution", "qrels_access"): "forbidden",
        ("execution", "dev_test_access"): "forbidden",
        ("scope", "potential_rms_is_dimensionless_standardized_gauge"): True,
        ("scope", "factor_realizability_tested"): False,
        ("scope", "score_bound_tested"): False,
        ("generation", "seeds"): [20260711, 20260712, 20260713],
        ("generation", "requests_per_seed"): 4096,
        ("generation", "candidates"): 16,
        ("generation", "history_events"): 6,
        ("generation", "request_batch_size"): 128,
        ("generation", "potential_rms"): 1.0,
        ("generation", "candidate_cycle_log_scale_std"): 0.75,
        ("generation", "event_cycle_ratio_log_std"): 0.7,
        ("generation", "noise_scale"): 0.8,
        ("generation", "variance_floor"): 0.05,
        ("generation", "numerical_epsilon"): 1.0e-12,
        ("worlds", "share_true_potential"): True,
        ("worlds", "share_cycle"): True,
        ("worlds", "share_gaussian_draws"): True,
        ("worlds", "share_variance_multiset_per_request"): True,
        ("metrics", "pairwise_true_tie_tolerance"): 1.0e-10,
        ("metrics", "predicted_tie_credit"): 0.5,
        ("metrics", "relevant_candidates"): 4,
        ("metrics", "ndcg_cutoff"): 10,
        ("metrics", "tie_break_salt"): "20260708",
        ("metrics", "paired_bootstrap_samples"): 10000,
        ("metrics", "paired_bootstrap_seed"): 20260714,
        ("metrics", "paired_bootstrap_confidence"): 0.95,
        ("integrity_thresholds", "skew_max_abs"): 1.0e-12,
        ("integrity_thresholds", "divergence_max_abs"): 1.0e-12,
        ("integrity_thresholds", "hodge_recovery_max_abs"): 1.0e-12,
        ("integrity_thresholds", "variance_multiset_max_abs"): 0.0,
        ("integrity_thresholds", "aligned_mean_spearman_min"): 0.999999999,
        ("integrity_thresholds", "decoupled_abs_mean_spearman_max"): 0.02,
        ("integrity_thresholds", "adversarial_mean_spearman_max"): -0.999999999,
        ("stop_rules", "aligned", "local_minus_t_one_pairwise_min"): 0.01,
        ("stop_rules", "aligned", "local_minus_global_pairwise_min"): 0.01,
        (
            "stop_rules",
            "decoupled",
            "local_minus_t_one_pairwise_ci_high_max",
        ): 0.002,
        ("stop_rules", "decoupled", "material_harm_allowed"): True,
        (
            "stop_rules",
            "decoupled",
            "interpretation",
        ): "null_or_harm_allowed_residual_positive_gain_forbidden",
        ("stop_rules", "adversarial", "local_minus_t_one_pairwise_max"): -0.01,
        ("stop_rules", "adversarial", "oracle_minus_local_pairwise_min"): 0.01,
        ("output_contract", "refuse_existing_output"): True,
        (
            "output_contract",
            "reuse_pre_run_manifest_without_post_outcome_rehash",
        ): True,
    }
    for keys, expected in exact.items():
        actual = _nested(config, *keys)
        if actual != expected:
            raise ValueError(
                f"frozen setting {'.'.join(keys)} changed: {actual!r} != {expected!r}"
            )
    event_weight = float(_nested(config, "generation", "event_weight"))
    if event_weight != 1.0 / 6.0:
        raise ValueError("the six event weights must remain exactly uniform")
    declared_worlds = tuple(_nested(config, "worlds", "names"))
    if declared_worlds != WORLD_NAMES:
        raise ValueError(f"world list changed: {declared_worlds!r}")
    declared_gates = tuple(_nested(config, "gates", "names"))
    if declared_gates != GATE_NAMES:
        raise ValueError(f"gate list changed: {declared_gates!r}")
    expected_c06_root = _repo_path("systems/06_conservative_wedge_flow_transformer")
    if C06_ROOT != expected_c06_root or SCRIPT_PATH != _repo_path(SCRIPT_REL):
        raise RuntimeError("executing source is not at the frozen C06 path")


def _center_last(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=-1, keepdims=True)
    # A second subtraction makes the numerical gauge explicit in FP64.
    return centered - centered.mean(axis=-1, keepdims=True)


def _row_spearman(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Spearman correlation per row; fixtures are continuous and tie-free."""

    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("Spearman inputs must share [B, D] shape")
    left_rank = np.argsort(np.argsort(left, axis=1, kind="mergesort"), axis=1)
    right_rank = np.argsort(np.argsort(right, axis=1, kind="mergesort"), axis=1)
    left_rank = left_rank.astype(np.float64)
    right_rank = right_rank.astype(np.float64)
    left_rank -= left_rank.mean(axis=1, keepdims=True)
    right_rank -= right_rank.mean(axis=1, keepdims=True)
    denominator = np.sqrt(
        np.sum(left_rank * left_rank, axis=1)
        * np.sum(right_rank * right_rank, axis=1)
    )
    return np.sum(left_rank * right_rank, axis=1) / denominator


def generate_synthetic_batch(
    rng: np.random.Generator,
    request_count: int,
    candidates: int,
    history_events: int,
    *,
    potential_rms: float = 1.0,
    candidate_cycle_log_scale_std: float = 0.75,
    event_cycle_ratio_log_std: float = 0.7,
    noise_scale: float = 0.8,
    variance_floor: float = 0.05,
    epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Generate paired worlds with identical marginals and Gaussian draws."""

    if request_count <= 0 or candidates < 3 or history_events <= 0:
        raise ValueError("invalid synthetic dimensions")
    shape = (request_count, history_events, candidates)
    raw_potential = rng.normal(size=shape).astype(np.float64)
    true_potential = _center_last(raw_potential)
    rms = np.sqrt(np.mean(true_potential * true_potential, axis=-1, keepdims=True))
    true_potential = potential_rms * true_potential / np.maximum(rms, epsilon)
    true_potential = _center_last(true_potential)

    node_scale = rng.lognormal(
        mean=0.0,
        sigma=candidate_cycle_log_scale_std,
        size=shape,
    ).astype(np.float64)
    raw = rng.normal(
        size=(request_count, history_events, candidates, candidates)
    ).astype(np.float64)
    raw_skew = raw - np.swapaxes(raw, -1, -2)
    raw_skew *= (
        node_scale[..., :, None] * node_scale[..., None, :]
        / math.sqrt(2.0 * candidates)
    )
    projector = np.eye(candidates, dtype=np.float64) - np.full(
        (candidates, candidates), 1.0 / candidates, dtype=np.float64
    )
    cycle = np.einsum(
        "ij,bhjk,kl->bhil", projector, raw_skew, projector, optimize=True
    )
    cycle = 0.5 * (cycle - np.swapaxes(cycle, -1, -2))

    true_gradient = (
        true_potential[..., :, None] - true_potential[..., None, :]
    )
    true_gradient_energy = np.sum(true_gradient * true_gradient, axis=-1)
    cycle_energy = np.sum(cycle * cycle, axis=-1)
    event_ratio = rng.lognormal(
        mean=-0.5 * event_cycle_ratio_log_std**2,
        sigma=event_cycle_ratio_log_std,
        size=(request_count, history_events),
    ).astype(np.float64)
    scale = np.sqrt(
        event_ratio
        * true_gradient_energy.mean(axis=-1)
        / np.maximum(cycle_energy.mean(axis=-1), epsilon)
    )
    cycle *= scale[..., None, None]
    cycle_energy = np.sum(cycle * cycle, axis=-1)

    normalized_energy = cycle_energy / np.maximum(
        cycle_energy.mean(axis=(1, 2), keepdims=True), epsilon
    )
    flat_energy = normalized_energy.reshape(request_count, -1)
    decoupled_flat = np.empty_like(flat_energy)
    adversarial_flat = np.empty_like(flat_energy)
    for row in range(request_count):
        decoupled_flat[row] = flat_energy[row, rng.permutation(flat_energy.shape[1])]
        order = np.argsort(flat_energy[row], kind="mergesort")
        sorted_values = flat_energy[row, order]
        adversarial_flat[row, order] = sorted_values[::-1]
    assignments = {
        "reliability_aligned": normalized_energy,
        "reliability_decoupled": decoupled_flat.reshape(shape),
        "reliability_adversarial": adversarial_flat.reshape(shape),
    }
    gaussian_draws = rng.normal(size=shape).astype(np.float64)
    worlds: dict[str, dict[str, np.ndarray]] = {}
    for world_name in WORLD_NAMES:
        assignment = assignments[world_name]
        noise_variance = (
            noise_scale**2
            * (assignment + variance_floor)
            / (1.0 + variance_floor)
        )
        noise = np.sqrt(noise_variance) * gaussian_draws
        observed_potential = _center_last(true_potential + noise)
        worlds[world_name] = {
            "assignment": assignment,
            "noise_variance": noise_variance,
            "observed_potential": observed_potential,
        }

    return {
        "true_potential": true_potential,
        "cycle": cycle,
        "cycle_energy": cycle_energy,
        "normalized_cycle_energy": normalized_energy,
        "gaussian_draws": gaussian_draws,
        "worlds": worlds,
    }


def compute_gate_scores(
    observed_potential: np.ndarray,
    cycle_energy: np.ndarray,
    planted_noise_variance: np.ndarray,
    *,
    epsilon: float = 1.0e-12,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Apply four fixed gates through the same conservative edge operator."""

    gradient = observed_potential[..., :, None] - observed_potential[..., None, :]
    gradient_energy = np.sum(gradient * gradient, axis=-1)
    local_trust = gradient_energy / (gradient_energy + cycle_energy + epsilon)
    global_fraction = gradient_energy.sum(axis=-1) / (
        gradient_energy.sum(axis=-1) + cycle_energy.sum(axis=-1) + epsilon
    )
    gates = {
        "local_hodge": local_trust,
        "global_event": np.sqrt(global_fraction)[..., None]
        * np.ones_like(local_trust),
        "t_one": np.ones_like(local_trust),
        "direct_reliability_oracle": 1.0 / (1.0 + planted_noise_variance),
    }
    scores: dict[str, np.ndarray] = {}
    for name, gate in gates.items():
        trusted_edge = (
            0.5
            * gate[..., :, None]
            * gate[..., None, :]
            * gradient
        )
        event_divergence = trusted_edge.mean(axis=-1)
        scores[name] = event_divergence.mean(axis=1)
    diagnostics = {
        "gradient_energy": gradient_energy,
        "cycle_energy": cycle_energy,
        **gates,
    }
    return scores, diagnostics


def request_pairwise_accuracy(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    true_tie_tolerance: float,
    predicted_tie_credit: float,
) -> np.ndarray:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("pairwise inputs must share [B, C] shape")
    upper = np.triu_indices(predicted.shape[1], k=1)
    true_difference = target[:, upper[0]] - target[:, upper[1]]
    predicted_difference = predicted[:, upper[0]] - predicted[:, upper[1]]
    valid = np.abs(true_difference) >= true_tie_tolerance
    credit = (
        np.sign(true_difference) == np.sign(predicted_difference)
    ).astype(np.float64)
    credit = np.where(predicted_difference == 0.0, predicted_tie_credit, credit)
    denominator = valid.sum(axis=1)
    if np.any(denominator == 0):
        raise ValueError("a synthetic request has no non-tied true pairs")
    return np.sum(np.where(valid, credit, 0.0), axis=1) / denominator


def _tie_key(request_id: str, candidate: int, salt: str) -> int:
    payload = f"{request_id}\0candidate_{candidate}\0{salt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def request_binary_ndcg(
    predicted: np.ndarray,
    target: np.ndarray,
    request_ids: Iterable[str],
    *,
    relevant_candidates: int,
    cutoff: int,
    tie_break_salt: str,
) -> np.ndarray:
    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("NDCG inputs must share [B, C] shape")
    request_ids = list(request_ids)
    if len(request_ids) != predicted.shape[0]:
        raise ValueError("request ID count mismatch")
    candidates = predicted.shape[1]
    if not 0 < relevant_candidates <= candidates:
        raise ValueError("invalid relevant-candidate count")
    cutoff = min(cutoff, candidates)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    ideal = discounts[: min(relevant_candidates, cutoff)].sum()
    values = np.empty(predicted.shape[0], dtype=np.float64)
    for row, request_id in enumerate(request_ids):
        tie_keys = np.asarray(
            [_tie_key(request_id, candidate, tie_break_salt) for candidate in range(candidates)],
            dtype=np.uint64,
        )
        true_order = np.lexsort((tie_keys, -target[row]))
        relevance = np.zeros(candidates, dtype=np.float64)
        relevance[true_order[:relevant_candidates]] = 1.0
        predicted_order = np.lexsort((tie_keys, -predicted[row]))[:cutoff]
        values[row] = np.sum(relevance[predicted_order] * discounts) / ideal
    return values


def paired_bootstrap_many(
    differences: dict[str, np.ndarray],
    *,
    samples: int,
    seed: int,
    confidence: float,
    chunk_size: int = 32,
) -> dict[str, list[float]]:
    """Use shared request-resampling draws for all registered comparisons."""

    if not differences:
        raise ValueError("no paired differences")
    names = list(differences)
    request_count = len(differences[names[0]])
    if any(len(differences[name]) != request_count for name in names):
        raise ValueError("paired difference lengths do not match")
    matrix = np.column_stack([differences[name] for name in names]).astype(np.float64)
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty((samples, len(names)), dtype=np.float64)
    offset = 0
    while offset < samples:
        count = min(chunk_size, samples - offset)
        indices = rng.integers(0, request_count, size=(count, request_count))
        bootstrap_means[offset : offset + count] = matrix[indices].mean(axis=1)
        offset += count
    alpha = (1.0 - confidence) / 2.0
    lower = np.quantile(bootstrap_means, alpha, axis=0)
    upper = np.quantile(bootstrap_means, 1.0 - alpha, axis=0)
    return {
        name: [float(lower[index]), float(upper[index])]
        for index, name in enumerate(names)
    }


def _empty_metric_store() -> dict[str, dict[str, dict[str, list[np.ndarray]]]]:
    return {
        world: {
            gate: {"pairwise_accuracy": [], "ndcg_at_10": []}
            for gate in GATE_NAMES
        }
        for world in WORLD_NAMES
    }


def _comparison_key(world: str, left: str, right: str) -> str:
    return f"{world}:{left}_minus_{right}"


def _evaluate_stop_rules(
    config: dict[str, Any],
    integrity: dict[str, Any],
    comparisons: dict[str, Any],
) -> dict[str, Any]:
    aligned_t = comparisons[
        _comparison_key("reliability_aligned", "local_hodge", "t_one")
    ]
    aligned_g = comparisons[
        _comparison_key("reliability_aligned", "local_hodge", "global_event")
    ]
    decoupled = comparisons[
        _comparison_key("reliability_decoupled", "local_hodge", "t_one")
    ]
    adversarial = comparisons[
        _comparison_key("reliability_adversarial", "local_hodge", "t_one")
    ]
    oracle = comparisons[
        _comparison_key(
            "reliability_adversarial", "direct_reliability_oracle", "local_hodge"
        )
    ]
    rules = config["stop_rules"]
    checks = {
        "integrity_passed": bool(integrity["passed"]),
        "aligned_local_minus_t_one_effect": aligned_t["pairwise_mean_delta"]
        >= float(rules["aligned"]["local_minus_t_one_pairwise_min"]),
        "aligned_local_minus_t_one_ci": aligned_t["pairwise_bootstrap_95_ci"][0]
        > 0.0,
        "aligned_local_minus_t_one_all_seeds": all(
            value > 0.0 for value in aligned_t["pairwise_per_seed_mean_delta"]
        ),
        "aligned_local_minus_global_effect": aligned_g["pairwise_mean_delta"]
        >= float(rules["aligned"]["local_minus_global_pairwise_min"]),
        "aligned_local_minus_global_ci": aligned_g["pairwise_bootstrap_95_ci"][0]
        > 0.0,
        "aligned_local_minus_global_all_seeds": all(
            value > 0.0 for value in aligned_g["pairwise_per_seed_mean_delta"]
        ),
        "aligned_ndcg_directions": aligned_t["ndcg_mean_delta"] > 0.0
        and aligned_g["ndcg_mean_delta"] > 0.0,
        "decoupled_no_material_gain": decoupled["pairwise_bootstrap_95_ci"][1]
        <= float(
            rules["decoupled"]["local_minus_t_one_pairwise_ci_high_max"]
        ),
        "adversarial_local_harm_effect": adversarial["pairwise_mean_delta"]
        <= float(rules["adversarial"]["local_minus_t_one_pairwise_max"]),
        "adversarial_local_harm_ci": adversarial["pairwise_bootstrap_95_ci"][1]
        < 0.0,
        "adversarial_local_harm_all_seeds": all(
            value < 0.0 for value in adversarial["pairwise_per_seed_mean_delta"]
        ),
        "adversarial_oracle_gap_effect": oracle["pairwise_mean_delta"]
        >= float(rules["adversarial"]["oracle_minus_local_pairwise_min"]),
        "adversarial_oracle_gap_ci": oracle["pairwise_bootstrap_95_ci"][0]
        > 0.0,
    }
    passed = all(checks.values())
    if passed:
        decision = "PASS_SYNTHETIC_CONDITIONAL_MECHANISM_CONTRACT_ONLY"
    elif not checks["integrity_passed"]:
        decision = "STOP_SYNTHETIC_GENERATOR_OR_IMPLEMENTATION_INTEGRITY"
    else:
        decision = "STOP_SYNTHETIC_MECHANISM_CONTRACT_NOT_BIDIRECTIONAL"
    return {
        "passed": passed,
        "decision": decision,
        "checks": checks,
        "authorization_if_passed": "new lock may be considered; no train-internal/dev/test authorization",
    }


def run_formal_probe(config: dict[str, Any]) -> dict[str, Any]:
    validate_frozen_config(config)
    generation = config["generation"]
    metrics_config = config["metrics"]
    thresholds = config["integrity_thresholds"]
    requests_per_seed = int(generation["requests_per_seed"])
    batch_size = int(generation["request_batch_size"])
    all_seed_stores: dict[int, Any] = {}
    per_seed_metrics: dict[str, Any] = {}
    global_integrity = {
        "skew_max_abs": 0.0,
        "divergence_max_abs": 0.0,
        "hodge_recovery_max_abs": 0.0,
        "variance_multiset_max_abs": 0.0,
        "aligned_spearman": [],
        "decoupled_spearman": [],
        "adversarial_spearman": [],
    }

    for seed in generation["seeds"]:
        seed = int(seed)
        rng = np.random.default_rng(seed)
        store = _empty_metric_store()
        processed = 0
        while processed < requests_per_seed:
            current = min(batch_size, requests_per_seed - processed)
            batch = generate_synthetic_batch(
                rng,
                current,
                int(generation["candidates"]),
                int(generation["history_events"]),
                potential_rms=float(generation["potential_rms"]),
                candidate_cycle_log_scale_std=float(
                    generation["candidate_cycle_log_scale_std"]
                ),
                event_cycle_ratio_log_std=float(
                    generation["event_cycle_ratio_log_std"]
                ),
                noise_scale=float(generation["noise_scale"]),
                variance_floor=float(generation["variance_floor"]),
                epsilon=float(generation["numerical_epsilon"]),
            )
            cycle = batch["cycle"]
            cycle_energy = batch["cycle_energy"]
            normalized_energy = batch["normalized_cycle_energy"]
            global_integrity["skew_max_abs"] = max(
                global_integrity["skew_max_abs"],
                float(np.max(np.abs(cycle + np.swapaxes(cycle, -1, -2)))),
            )
            global_integrity["divergence_max_abs"] = max(
                global_integrity["divergence_max_abs"],
                float(np.max(np.abs(cycle.sum(axis=-1)))),
            )
            flat_energy = normalized_energy.reshape(current, -1)
            aligned_variance = batch["worlds"]["reliability_aligned"][
                "noise_variance"
            ].reshape(current, -1)
            decoupled_variance = batch["worlds"]["reliability_decoupled"][
                "noise_variance"
            ].reshape(current, -1)
            adversarial_variance = batch["worlds"]["reliability_adversarial"][
                "noise_variance"
            ].reshape(current, -1)
            global_integrity["aligned_spearman"].append(
                _row_spearman(flat_energy, aligned_variance)
            )
            global_integrity["decoupled_spearman"].append(
                _row_spearman(flat_energy, decoupled_variance)
            )
            global_integrity["adversarial_spearman"].append(
                _row_spearman(flat_energy, adversarial_variance)
            )
            aligned_sorted = np.sort(aligned_variance, axis=1)
            for other in (decoupled_variance, adversarial_variance):
                global_integrity["variance_multiset_max_abs"] = max(
                    global_integrity["variance_multiset_max_abs"],
                    float(np.max(np.abs(aligned_sorted - np.sort(other, axis=1)))),
                )

            target = batch["true_potential"].mean(axis=1)
            request_ids = [
                f"synthetic_s{seed}_r{index}"
                for index in range(processed, processed + current)
            ]
            for world_name in WORLD_NAMES:
                world = batch["worlds"][world_name]
                observed = world["observed_potential"]
                flow = (
                    observed[..., :, None]
                    - observed[..., None, :]
                    + cycle
                )
                recovered = flow.mean(axis=-1)
                global_integrity["hodge_recovery_max_abs"] = max(
                    global_integrity["hodge_recovery_max_abs"],
                    float(np.max(np.abs(recovered - observed))),
                )
                scores, _ = compute_gate_scores(
                    observed,
                    cycle_energy,
                    world["noise_variance"],
                    epsilon=float(generation["numerical_epsilon"]),
                )
                for gate_name, predicted in scores.items():
                    store[world_name][gate_name]["pairwise_accuracy"].append(
                        request_pairwise_accuracy(
                            predicted,
                            target,
                            true_tie_tolerance=float(
                                metrics_config["pairwise_true_tie_tolerance"]
                            ),
                            predicted_tie_credit=float(
                                metrics_config["predicted_tie_credit"]
                            ),
                        )
                    )
                    store[world_name][gate_name]["ndcg_at_10"].append(
                        request_binary_ndcg(
                            predicted,
                            target,
                            request_ids,
                            relevant_candidates=int(
                                metrics_config["relevant_candidates"]
                            ),
                            cutoff=int(metrics_config["ndcg_cutoff"]),
                            tie_break_salt=str(metrics_config["tie_break_salt"]),
                        )
                    )
            processed += current

        concatenated = {
            world: {
                gate: {
                    metric: np.concatenate(chunks)
                    for metric, chunks in store[world][gate].items()
                }
                for gate in GATE_NAMES
            }
            for world in WORLD_NAMES
        }
        all_seed_stores[seed] = concatenated
        per_seed_metrics[str(seed)] = {
            world: {
                gate: {
                    metric: float(values.mean())
                    for metric, values in concatenated[world][gate].items()
                }
                for gate in GATE_NAMES
            }
            for world in WORLD_NAMES
        }

    pooled = {
        world: {
            gate: {
                metric: np.concatenate(
                    [all_seed_stores[int(seed)][world][gate][metric] for seed in generation["seeds"]]
                )
                for metric in ("pairwise_accuracy", "ndcg_at_10")
            }
            for gate in GATE_NAMES
        }
        for world in WORLD_NAMES
    }
    pooled_metrics = {
        world: {
            gate: {
                metric: float(values.mean())
                for metric, values in pooled[world][gate].items()
            }
            for gate in GATE_NAMES
        }
        for world in WORLD_NAMES
    }

    comparison_specs = (
        ("reliability_aligned", "local_hodge", "t_one"),
        ("reliability_aligned", "local_hodge", "global_event"),
        ("reliability_decoupled", "local_hodge", "t_one"),
        ("reliability_adversarial", "local_hodge", "t_one"),
        (
            "reliability_adversarial",
            "direct_reliability_oracle",
            "local_hodge",
        ),
    )
    paired_differences: dict[str, np.ndarray] = {}
    comparisons: dict[str, Any] = {}
    for world, left, right in comparison_specs:
        key = _comparison_key(world, left, right)
        pairwise_difference = (
            pooled[world][left]["pairwise_accuracy"]
            - pooled[world][right]["pairwise_accuracy"]
        )
        ndcg_difference = (
            pooled[world][left]["ndcg_at_10"]
            - pooled[world][right]["ndcg_at_10"]
        )
        paired_differences[key] = pairwise_difference
        comparisons[key] = {
            "world": world,
            "left": left,
            "right": right,
            "pairwise_mean_delta": float(pairwise_difference.mean()),
            "pairwise_per_seed_mean_delta": [
                float(
                    (
                        all_seed_stores[int(seed)][world][left]["pairwise_accuracy"]
                        - all_seed_stores[int(seed)][world][right]["pairwise_accuracy"]
                    ).mean()
                )
                for seed in generation["seeds"]
            ],
            "ndcg_mean_delta": float(ndcg_difference.mean()),
            "ndcg_per_seed_mean_delta": [
                float(
                    (
                        all_seed_stores[int(seed)][world][left]["ndcg_at_10"]
                        - all_seed_stores[int(seed)][world][right]["ndcg_at_10"]
                    ).mean()
                )
                for seed in generation["seeds"]
            ],
        }
    intervals = paired_bootstrap_many(
        paired_differences,
        samples=int(metrics_config["paired_bootstrap_samples"]),
        seed=int(metrics_config["paired_bootstrap_seed"]),
        confidence=float(metrics_config["paired_bootstrap_confidence"]),
    )
    for key, interval in intervals.items():
        comparisons[key]["pairwise_bootstrap_95_ci"] = interval

    aligned_spearman = np.concatenate(global_integrity.pop("aligned_spearman"))
    decoupled_spearman = np.concatenate(global_integrity.pop("decoupled_spearman"))
    adversarial_spearman = np.concatenate(global_integrity.pop("adversarial_spearman"))
    integrity_values = {
        **global_integrity,
        "aligned_mean_spearman": float(aligned_spearman.mean()),
        "decoupled_mean_spearman": float(decoupled_spearman.mean()),
        "adversarial_mean_spearman": float(adversarial_spearman.mean()),
        "worlds_share_true_potential": True,
        "worlds_share_cycle": True,
        "worlds_share_gaussian_draws": True,
    }
    integrity_checks = {
        "skew": integrity_values["skew_max_abs"]
        <= float(thresholds["skew_max_abs"]),
        "divergence_free": integrity_values["divergence_max_abs"]
        <= float(thresholds["divergence_max_abs"]),
        "hodge_recovery": integrity_values["hodge_recovery_max_abs"]
        <= float(thresholds["hodge_recovery_max_abs"]),
        "variance_multiset_shared": integrity_values["variance_multiset_max_abs"]
        <= float(thresholds["variance_multiset_max_abs"]),
        "aligned_correlation": integrity_values["aligned_mean_spearman"]
        >= float(thresholds["aligned_mean_spearman_min"]),
        "decoupled_correlation": abs(integrity_values["decoupled_mean_spearman"])
        <= float(thresholds["decoupled_abs_mean_spearman_max"]),
        "adversarial_correlation": integrity_values["adversarial_mean_spearman"]
        <= float(thresholds["adversarial_mean_spearman_max"]),
    }
    integrity = {
        **integrity_values,
        "checks": integrity_checks,
        "passed": all(integrity_checks.values()),
    }
    verdict = _evaluate_stop_rules(config, integrity, comparisons)
    return {
        "probe_id": config["probe_id"],
        "status": "formal_synthetic_probe_completed",
        "execution": {
            "device": "cpu",
            "dtype": "float64",
            "training": False,
            "seeds": [int(seed) for seed in generation["seeds"]],
            "requests_per_seed": requests_per_seed,
            "total_requests_per_world": requests_per_seed
            * len(generation["seeds"]),
            "repository_data_read": False,
            "qrels_read": False,
            "dev_test_read": False,
        },
        "integrity": integrity,
        "per_seed_metrics": per_seed_metrics,
        "pooled_metrics": pooled_metrics,
        "comparisons": comparisons,
        "stop_rule_verdict": verdict,
        "interpretation_boundary": (
            "An aligned-world pass establishes conditional synthetic competence only; "
            "it does not establish that real recommendation evidence has the planted "
            "cycle-error coupling, novelty, or real-data gain."
        ),
    }


def prepare_formal_preflight(
    config_argument: str | Path,
    output_argument: str | Path,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Fail closed before the first generator or RNG construction."""

    config_path = resolve_fixed_cli_path(config_argument, CONFIG_REL)
    output_path = resolve_fixed_cli_path(output_argument, OUTPUT_REL)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    validate_frozen_config(config)
    manifest = build_pre_run_manifest()
    lock = verify_preoutcome_lock(
        _repo_path(LOCK_REL), manifest, probe_id=str(config["probe_id"])
    )
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output_path}")
    return config, output_path, {
        "pre_run_manifest": manifest,
        "lock": lock,
        "output_path": OUTPUT_REL,
        "manifest_recomputed_after_outcome": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    # This preflight computes and verifies every identity before any call to
    # ``run_formal_probe`` can construct a generator or observe an outcome.
    config, output_path, locked_inputs = prepare_formal_preflight(
        args.config, args.output
    )
    result = run_formal_probe(config)
    # Reuse the exact pre-run object. Do not hash source after outcomes exist.
    result["locked_inputs"] = locked_inputs
    _write_json_exclusive(output_path, result)


if __name__ == "__main__":
    main()
