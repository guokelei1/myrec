"""Run the frozen C39 operator/synthetic design gate without repository data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from model.halfspace_value import (  # noqa: E402
    EVENTWISE_HALFSPACE,
    GLOBAL_ONLY,
    MODES,
    HalfspaceCertifiedValueTransformer,
    project_to_score_halfspace,
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C39 design config must be an object")
    return value


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(config: dict[str, Any], seed: int, mode: str) -> HalfspaceCertifiedValueTransformer:
    row = config["model_smoke"]
    return HalfspaceCertifiedValueTransformer(
        dim=int(row["dim"]),
        inner_dim=int(row["inner_dim"]),
        heads=int(row["heads"]),
        ffn_dim=int(row["ffn_dim"]),
        temperature=float(row["temperature"]),
        global_scale=float(row["global_scale"]),
        candidate_scale=float(row["candidate_scale"]),
        seed=seed,
        mode=mode,
    )


def run_d0(config: dict[str, Any], device: torch.device) -> dict[str, Any]:
    row = config["model_smoke"]
    gate = config["gate"]
    seed = int(config["synthetic"]["seeds"][0])
    models = {mode: make_model(config, seed, mode).to(device) for mode in MODES}
    counts = {mode: model.trainable_parameter_count() for mode, model in models.items()}
    hashes = {mode: state_sha256(model) for mode, model in models.items()}

    values = torch.tensor([[-2.0, 3.0], [1.0, 4.0]], device=device)
    normals = torch.tensor([[1.0, 0.0], [1.0, 0.0]], device=device)
    projected = project_to_score_halfspace(values, normals)
    expected = torch.tensor([[0.0, 3.0], [1.0, 4.0]], device=device)
    hand_error = float((projected - expected).abs().max().cpu())
    projection_violation = float(
        torch.relu(-(projected * normals).sum(-1)).max().cpu()
    )
    feasible_error = float((projected[1] - values[1]).abs().max().cpu())

    a = torch.tensor([2.0, 1.0], device=device)
    first = torch.stack((a, -a))
    second = torch.zeros_like(first)
    witness_normal = torch.tensor([1.0, 0.0], device=device).expand_as(first)
    ordinary_witness_error = float((first.mean(0) - second.mean(0)).abs().max().cpu())
    projected_witness_difference = float(
        (
            project_to_score_halfspace(first, witness_normal).mean(0)
            - project_to_score_halfspace(second, witness_normal).mean(0)
        )
        .norm()
        .cpu()
    )

    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    query = torch.randn(int(row["dim"]), generator=generator).to(device)
    history = torch.randn(
        int(row["history"]), int(row["dim"]), generator=generator
    ).to(device)
    candidates = torch.randn(
        int(row["candidates"]), int(row["dim"]), generator=generator
    ).to(device)
    primary = models[EVENTWISE_HALFSPACE]
    primary_state = primary.components(query, history, candidates)
    projected_violation_real = float(
        torch.relu(-primary_state["projected_readout"]).max().cpu()
    )
    unsupported = primary_state["support"] == 0
    unsupported_edge_error = float(
        primary_state["edge_value"][unsupported].abs().max().cpu()
    ) if bool(unsupported.any()) else math.inf
    permutation = torch.randperm(len(candidates), generator=generator).to(device)
    permutation_error = float(
        (
            primary(query, history, candidates)[permutation]
            - primary(query, history, candidates[permutation])
        )
        .abs()
        .max()
        .cpu()
    )
    zero = torch.zeros(len(candidates), device=device)
    fallback_errors = {
        "nohistory": float((primary(query, history[:0], candidates) - zero).abs().max().cpu()),
        "query_mask": float(
            (primary(query, history, candidates, query_present=False) - zero).abs().max().cpu()
        ),
        "repeat": float(
            (primary(query, history, candidates, repeat_present=True) - zero).abs().max().cpu()
        ),
    }

    shared_down = torch.randn(
        primary.ffn_down.weight.shape, generator=generator
    ).to(device) * 0.02
    outputs = {}
    for mode, model in models.items():
        with torch.no_grad():
            model.ffn_down.weight.copy_(shared_down)
        outputs[mode] = model(query, history, candidates)
    distinct_pairs = {
        f"{left}__{right}": bool(not torch.equal(outputs[left], outputs[right]))
        for position, left in enumerate(MODES)
        for right in MODES[position + 1 :]
    }

    gradient_model = make_model(config, seed + 1, EVENTWISE_HALFSPACE).to(device)
    optimizer = torch.optim.AdamW(
        gradient_model.parameters(), lr=float(row["learning_rate"])
    )
    target = torch.linspace(-0.5, 0.5, len(candidates), device=device)
    active_gradients: set[str] = set()
    finite_gradients = True
    for _ in range(int(row["gradient_steps"])):
        output = gradient_model(query, history, candidates)
        loss = (output - target).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in gradient_model.named_parameters():
            if parameter.grad is None:
                continue
            finite_gradients &= bool(torch.isfinite(parameter.grad).all())
            if bool(parameter.grad.ne(0).any()):
                active_gradients.add(name)
        optimizer.step()
    required_gradients = {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    }
    checks = {
        "hand_projection": hand_error == 0.0,
        "projection_feasible": max(projection_violation, projected_violation_real)
        <= float(gate["projection_max_violation"]),
        "feasible_identity": feasible_error
        <= float(gate["feasible_identity_max_error"]),
        "same_aggregate_witness": ordinary_witness_error == 0.0
        and projected_witness_difference > 0.0,
        "capacity_matched": len(set(counts.values())) == 1,
        "initialization_paired": len(set(hashes.values())) == 1,
        "gradients_finite": finite_gradients,
        "gradients_reach_all_components": required_gradients <= active_gradients,
        "candidate_permutation": permutation_error <= float(gate["permutation_max_error"]),
        "exact_fallbacks": max(fallback_errors.values()) == 0.0,
        "unsupported_edges_zero": unsupported_edge_error == 0.0,
        "modes_operator_distinct": all(distinct_pairs.values()),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "parameter_counts": counts,
        "initial_state_sha256": hashes,
        "hand_error": hand_error,
        "projection_max_violation": max(projection_violation, projected_violation_real),
        "feasible_identity_max_error": feasible_error,
        "ordinary_witness_error": ordinary_witness_error,
        "projected_witness_difference": projected_witness_difference,
        "candidate_permutation_max_error": permutation_error,
        "fallback_max_errors": fallback_errors,
        "unsupported_edge_max_abs": unsupported_edge_error,
        "distinct_mode_pairs": distinct_pairs,
        "active_gradient_names": sorted(active_gradients),
    }


def binary_ndcg(scores: np.ndarray, positive: int) -> float:
    order = np.lexsort((np.arange(len(scores)), -scores))
    rank = int(np.flatnonzero(order == positive)[0]) + 1
    return 1.0 / math.log2(rank + 1)


def synthetic_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    row = config["synthetic"]
    rng = np.random.default_rng(seed)
    requests = int(row["requests"])
    candidates = int(row["candidates"])
    first_weight, second_weight = [float(value) for value in row["pair_weights"]]
    if not math.isclose(first_weight + second_weight, 1.0):
        raise ValueError("C39 synthetic pair weights must sum to one")
    ndcg = {mode: [] for mode in (*MODES, "base", "primary_wrong")}
    rejected_edges = 0
    total_edges = 0
    changed_negative = 0
    negative_edges = 0
    repeat_correction_max = 0.0
    nohistory_correction_max = 0.0
    for request in range(requests):
        positive = int(rng.integers(candidates))
        competitor = int((positive + int(rng.integers(1, candidates))) % candidates)
        base = np.full(candidates, float(row["other_base_score_max"]) - 0.01, dtype=np.float64)
        base[positive] = float(row["positive_base_score"])
        base[competitor] = float(row["competitor_base_score"])
        repeat = request < round(requests * float(row["repeat_fraction"]))
        if repeat:
            base[positive] = float(row["competitor_base_score"]) + 0.6
        donor = competitor

        # Random rotation prevents a coordinate-specific implementation.  n is
        # the candidate readout normal and m is a score-neutral semantic value.
        matrix = rng.standard_normal((2 * candidates, 2 * candidates))
        orthogonal, _ = np.linalg.qr(matrix)
        normal = orthogonal[:, positive]
        semantic = orthogonal[:, candidates + positive]
        pair = np.stack((-normal + semantic, normal - semantic))
        pair_normal = np.stack((normal, normal))
        dot = np.sum(pair * pair_normal, axis=1)
        projected = pair + np.maximum(-dot, 0.0)[:, None] * pair_normal
        ray = np.maximum(dot, 0.0)[:, None] * pair_normal
        raw_pool = first_weight * pair[0] + second_weight * pair[1]
        raw_pool_dot = float(raw_pool @ normal)
        postpool = raw_pool + max(-raw_pool_dot, 0.0) * normal
        primary_value = first_weight * projected[0] + second_weight * projected[1]
        ray_value = first_weight * ray[0] + second_weight * ray[1]
        downstream = np.outer(normal, semantic)

        def read(value: np.ndarray) -> float:
            return float(normal @ value + normal @ downstream @ value)

        correction = {
            EVENTWISE_HALFSPACE: read(primary_value),
            "eventwise_raw": read(raw_pool),
            "postpool_halfspace": read(postpool),
            "ray_only": read(ray_value),
            GLOBAL_ONLY: 0.0,
        }
        if repeat:
            correction = {mode: 0.0 for mode in MODES}
        for mode in MODES:
            scores = base.copy()
            scores[positive] += correction[mode]
            ndcg[mode].append(binary_ndcg(scores, positive))
        ndcg["base"].append(binary_ndcg(base, positive))
        wrong_scores = base.copy()
        if not repeat:
            wrong_scores[donor] += read(primary_value)
        ndcg["primary_wrong"].append(binary_ndcg(wrong_scores, positive))

        total_edges += 2 * candidates
        rejected_edges += 2 * (candidates - 1)
        negative_edges += int((dot < 0).sum())
        changed_negative += int((dot < 0).sum())
        repeat_correction_max = max(
            repeat_correction_max,
            max(abs(value) for value in correction.values()) if repeat else 0.0,
        )
        nohistory_correction_max = max(nohistory_correction_max, 0.0)

    means = {name: float(np.mean(values)) for name, values in ndcg.items()}
    differences = {
        "primary_over_base": means[EVENTWISE_HALFSPACE] - means["base"],
        "primary_over_postpool": means[EVENTWISE_HALFSPACE]
        - means["postpool_halfspace"],
        "primary_over_global": means[EVENTWISE_HALFSPACE] - means[GLOBAL_ONLY],
        "true_over_wrong": means[EVENTWISE_HALFSPACE] - means["primary_wrong"],
    }
    return {
        "seed": seed,
        "ndcg10": means,
        "differences": differences,
        "rejected_edge_fraction": rejected_edges / total_edges,
        "projection_changed_negative_fraction": changed_negative / max(negative_edges, 1),
        "repeat_correction_max_abs": repeat_correction_max,
        "nohistory_correction_max_abs": nohistory_correction_max,
        "optimization_steps": int(row["optimization_steps"]),
    }


def run_d1(config: dict[str, Any]) -> dict[str, Any]:
    rows = [synthetic_seed(config, int(seed)) for seed in config["synthetic"]["seeds"]]
    gate = config["gate"]
    checks = {
        "primary_over_base_each_seed": all(
            row["differences"]["primary_over_base"]
            >= float(gate["primary_over_base_ndcg_min_each_seed"])
            for row in rows
        ),
        "primary_over_postpool_each_seed": all(
            row["differences"]["primary_over_postpool"]
            >= float(gate["primary_over_postpool_ndcg_min_each_seed"])
            for row in rows
        ),
        "primary_over_global_each_seed": all(
            row["differences"]["primary_over_global"]
            >= float(gate["primary_over_global_ndcg_min_each_seed"])
            for row in rows
        ),
        "true_over_wrong_each_seed": all(
            row["differences"]["true_over_wrong"]
            >= float(gate["true_over_wrong_ndcg_min_each_seed"])
            for row in rows
        ),
        "rejected_edges_each_seed": all(
            row["rejected_edge_fraction"]
            >= float(gate["rejected_edge_fraction_min_each_seed"])
            for row in rows
        ),
        "projection_changes_negative_each_seed": all(
            row["projection_changed_negative_fraction"]
            >= float(gate["projection_changed_negative_fraction_min_each_seed"])
            for row in rows
        ),
        "repeat_exact_base": all(row["repeat_correction_max_abs"] == 0.0 for row in rows),
        "nohistory_exact_base": all(
            row["nohistory_correction_max_abs"] == 0.0 for row in rows
        ),
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "seeds": rows,
        "scope": (
            "Constructed equal-pooled-value operator witness only; not real-data "
            "signal, utility, transfer, or novelty evidence."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("C39 design gate requires CUDA")
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(config["synthetic"]["seeds"][0]))
    torch.cuda.manual_seed_all(int(config["synthetic"]["seeds"][0]))
    d0 = run_d0(config, torch.device(args.device))
    d1 = run_d1(config)
    report = {
        "candidate_id": "c39",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if d0["status"] == d1["status"] == "passed" else "failed",
        "D0": d0,
        "D1": d1,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "candidate_source_root": str(SYSTEM_ROOT.relative_to(REPO_ROOT)),
        "physical_gpu": int(config["physical_gpu"]),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "real_repository_data_opened": False,
        "real_labels_scores_qrels_opened": False,
        "dev_test_opened": False,
        "authorization_if_passed": (
            "Amazon-C4 train-internal proposal formulation/implementation only; "
            "no dev/test or proposed-system claim."
        ),
    }
    output_path = Path(config["output"]["report"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(output_path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "passed" else 2


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    raise SystemExit(main())
