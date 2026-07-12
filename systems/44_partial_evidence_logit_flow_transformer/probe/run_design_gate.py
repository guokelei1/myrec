"""Run the locked C44 data-free synthetic mechanism gate on one GPU."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from model.partial_logit_flow import (  # noqa: E402
    FORCED_LOGIT_FLOW,
    GLOBAL_VECTOR_WRITE,
    MODES,
    PARTIAL_LOGIT_FLOW,
    PARTIAL_VECTOR_WRITE,
    PartialEvidenceLogitFlowTransformer,
)
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402


PRIMARY = PARTIAL_LOGIT_FLOW
CONTROLS = (FORCED_LOGIT_FLOW, PARTIAL_VECTOR_WRITE, GLOBAL_VECTOR_WRITE)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict) or value.get("candidate_id") != "c44":
        raise ValueError("unexpected C44 config")
    if value.get("gate_id") != "c44_partial_evidence_logit_flow_design_v1":
        raise ValueError("unexpected C44 gate")
    return value


def verify_lock(config: Mapping[str, Any]) -> str:
    path = Path(config["paths"]["design_lock"])
    if not path.is_file():
        raise PermissionError("C44 design lock missing")
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("lock_id") != "c44_partial_evidence_logit_flow_design_v1":
        raise ValueError("unexpected C44 design lock")
    if lock.get("status") != "locked_before_data_free_outcome":
        raise ValueError("C44 design lock stage differs")
    failures = []
    lines = []
    for relative, expected in sorted(lock["files_sha256"].items()):
        source = SYSTEM_ROOT / relative
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"candidate:{relative}")
        lines.append(f"{expected}  {relative}\n")
    if hashlib.sha256("".join(lines).encode()).hexdigest() != lock["aggregate_sha256"]:
        failures.append("aggregate")
    for raw, expected in lock["external_inputs_sha256"].items():
        source = Path(raw)
        if not source.is_file() or sha256_file(source) != expected:
            failures.append(f"external:{raw}")
    if failures:
        raise RuntimeError(f"C44 design lock mismatch: {failures}")
    if lock["declarations"].get("repository_dataset_read") is not False:
        raise PermissionError("C44 lock is not data-free")
    return sha256_file(path)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def assert_gpu(config: Mapping[str, Any], device: torch.device) -> None:
    physical = int(config["resources"]["physical_gpu"])
    if str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C44 GPU registration mismatch")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C44 deterministic CUBLAS setting missing")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C44 requires exactly one visible GPU")


@dataclass(frozen=True)
class SyntheticRows:
    query: np.ndarray
    clean_history: np.ndarray
    wrong_history: np.ndarray
    candidates: np.ndarray
    base_scores: np.ndarray
    target: np.ndarray
    signal_position: np.ndarray


def normalize(value: np.ndarray, axis: int = -1) -> np.ndarray:
    denominator = np.linalg.norm(value, axis=axis, keepdims=True)
    return value / np.maximum(denominator, 1e-12)


def make_rows(config: Mapping[str, Any], count: int, seed: int) -> SyntheticRows:
    row = config["synthetic"]
    dim = int(config["model"]["embedding_dim"])
    candidates = int(row["candidates"])
    events = int(row["history_events"])
    component = float(row["noise_candidate_component"])
    rng = np.random.default_rng(seed)
    queries = []
    clean_rows = []
    wrong_rows = []
    candidate_rows = []
    base_rows = []
    targets = []
    signal_positions = []
    for _ in range(count):
        query = normalize(rng.normal(size=dim).astype(np.float64))
        raw = rng.normal(size=(dim, candidates + events + 2))
        raw = raw - query[:, None] * (query @ raw)[None, :]
        basis, _ = np.linalg.qr(raw, mode="reduced")
        candidate_directions = basis[:, :candidates].T
        outside = basis[:, candidates : candidates + events + 1].T
        candidate_states = normalize(query[None, :] + candidate_directions)
        target = int(rng.integers(0, candidates))
        noise = []
        for event in range(events):
            coefficients = normalize(rng.normal(size=candidates))
            value = outside[event] + component * coefficients @ candidate_directions
            noise.append(normalize(value))
        clean = np.stack([candidate_directions[target], *noise[: events - 1]])
        order = rng.permutation(events)
        clean = clean[order]
        signal_position = int(np.flatnonzero(order == 0)[0])
        wrong = np.stack(noise[:events])
        queries.append(query.astype(np.float32))
        clean_rows.append(clean.astype(np.float32))
        wrong_rows.append(wrong.astype(np.float32))
        candidate_rows.append(candidate_states.astype(np.float32))
        base_rows.append(
            rng.normal(scale=float(row["base_noise_std"]), size=candidates).astype(np.float32)
        )
        targets.append(target)
        signal_positions.append(signal_position)
    return SyntheticRows(
        query=np.stack(queries),
        clean_history=np.stack(clean_rows),
        wrong_history=np.stack(wrong_rows),
        candidates=np.stack(candidate_rows),
        base_scores=np.stack(base_rows),
        target=np.asarray(targets, dtype=np.int64),
        signal_position=np.asarray(signal_positions, dtype=np.int64),
    )


def make_model(
    config: Mapping[str, Any], seed: int, mode: str
) -> PartialEvidenceLogitFlowTransformer:
    row = config["model"]
    return PartialEvidenceLogitFlowTransformer(
        dim=int(row["embedding_dim"]),
        heads=int(row["heads"]),
        rank=int(row["rank"]),
        temperature=float(row["temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        seed=seed,
        mode=mode,
        init_std=float(row["init_std"]),
    )


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def tensors(rows: SyntheticRows, index: int, device: torch.device, source: str):
    history = rows.clean_history if source == "clean" else rows.wrong_history
    return (
        torch.from_numpy(rows.query[index]).to(device),
        torch.from_numpy(history[index]).to(device),
        torch.from_numpy(rows.candidates[index]).to(device),
        torch.from_numpy(rows.base_scores[index]).to(device),
    )


def train_mode(
    model: PartialEvidenceLogitFlowTransformer,
    rows: SyntheticRows,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    row = config["training"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(row["learning_rate"]),
        weight_decay=float(row["weight_decay"]),
    )
    model.to(device).train()
    losses = []
    gradients: set[str] = set()
    for epoch in range(int(row["epochs"])):
        order = np.random.default_rng(seed + epoch * 1009).permutation(len(rows.target))
        for start in range(0, len(order), int(row["batch_size"])):
            request_losses = []
            for raw in order[start : start + int(row["batch_size"])]:
                index = int(raw)
                query, history, candidates, base = tensors(rows, index, device, "clean")
                score = base + model(query, history, candidates)
                target = int(rows.target[index])
                request_losses.append(torch.logsumexp(score, dim=0) - score[target])
            loss = torch.stack(request_losses).mean()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("nonfinite C44 loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None:
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise RuntimeError(f"nonfinite C44 gradient: {name}")
                    if bool(parameter.grad.ne(0).any()):
                        gradients.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(row["gradient_clip_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {
        "steps": len(losses),
        "finite": bool(np.isfinite(losses).all()),
        "loss_first_20": float(np.mean(losses[:20])),
        "loss_last_20": float(np.mean(losses[-20:])),
        "gradient_parameter_names": sorted(gradients),
    }


def ndcg_row(scores: np.ndarray, target: int, request_id: str) -> float:
    item_ids = [f"item_{index}" for index in range(len(scores))]
    ranked = [
        row.item_id
        for row in sort_candidates(
            request_id,
            [ScoredCandidate(item, float(score)) for item, score in zip(item_ids, scores)],
        )
    ]
    return ndcg_at_k(ranked, {item_ids[int(target)]}, 10)


def evaluate(
    model: PartialEvidenceLogitFlowTransformer,
    rows: SyntheticRows,
    device: torch.device,
    source: str,
) -> tuple[np.ndarray, list[np.ndarray]]:
    model.eval()
    metrics = []
    corrections = []
    with torch.inference_mode():
        for index in range(len(rows.target)):
            query, history, candidates, base = tensors(rows, index, device, source)
            correction = model(query, history, candidates)
            score = (base + correction).cpu().numpy()
            metrics.append(ndcg_row(score, int(rows.target[index]), f"c44_{source}_{index}"))
            corrections.append(correction.cpu().numpy())
    return np.asarray(metrics, dtype=np.float64), corrections


def base_metrics(rows: SyntheticRows) -> np.ndarray:
    return np.asarray(
        [
            ndcg_row(scores, int(target), f"c44_base_{index}")
            for index, (scores, target) in enumerate(zip(rows.base_scores, rows.target))
        ],
        dtype=np.float64,
    )


def diagnostics(
    model: PartialEvidenceLogitFlowTransformer,
    rows: SyntheticRows,
    device: torch.device,
) -> dict[str, float | list[int] | bool]:
    signal_mass = []
    irrelevant_null = []
    wrong_null = []
    correction_sum = 0.0
    model.eval()
    with torch.inference_mode():
        for index in range(len(rows.target)):
            query, clean, candidates, _ = tensors(rows, index, device, "clean")
            state = model.components(query, clean, candidates)
            target = int(rows.target[index])
            signal = int(rows.signal_position[index])
            signal_mass.append(float(state["candidate_mass"][:, target, signal].mean().cpu()))
            mask = [event for event in range(clean.shape[0]) if event != signal]
            irrelevant_null.append(float(state["null_mass"][:, mask].mean().cpu()))
            correction_sum = max(
                correction_sum, float(state["correction"].sum().abs().cpu())
            )
            _, wrong, _, _ = tensors(rows, index, device, "wrong")
            wrong_state = model.components(query, wrong, candidates)
            wrong_null.append(float(wrong_state["null_mass"].mean().cpu()))
    return {
        "signal_candidate_mass_mean": float(np.mean(signal_mass)),
        "irrelevant_null_mass_mean": float(np.mean(irrelevant_null)),
        "wrong_null_mass_mean": float(np.mean(wrong_null)),
        "correction_sum_max_abs": correction_sum,
    }


def structural_checks(
    models: Mapping[str, PartialEvidenceLogitFlowTransformer],
    rows: SyntheticRows,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    gate = config["gate"]
    index = 0
    query, history, candidates, _ = tensors(rows, index, device, "clean")
    checks: dict[str, bool] = {}
    diagnostics_by_mode = {}
    expected_parameters = 2 * int(config["model"]["heads"]) * int(config["model"]["rank"]) * int(config["model"]["embedding_dim"])
    checks["equal_capacity"] = {model.trainable_parameter_count() for model in models.values()} == {expected_parameters}
    for mode, model in models.items():
        model.eval()
        with torch.inference_mode():
            first = model(query, history, candidates)
            second = model(query, history, candidates)
            order = torch.tensor([7, 1, 6, 3, 4, 5, 2, 0], device=device)
            permuted = model(query, history, candidates[order])
            state = model.components(query, history, candidates)
            deterministic = float((first - second).abs().max().cpu())
            permutation = float((first[order] - permuted).abs().max().cpu())
            nohistory = float(model(query, history[:0], candidates).abs().max().cpu())
            noquery = float(
                model(query, history, candidates, query_present=False).abs().max().cpu()
            )
            repeat = float(
                model(query, history, candidates, repeat_present=True).abs().max().cpu()
            )
            finite = all(
                bool(torch.isfinite(value).all())
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        diagnostics_by_mode[mode] = {
            "deterministic_max_abs": deterministic,
            "candidate_permutation_max_abs": permutation,
            "nohistory_max_abs": nohistory,
            "query_absent_max_abs": noquery,
            "repeat_max_abs": repeat,
            "states_finite": finite,
        }
        checks[f"{mode}_contracts"] = (
            deterministic <= float(gate["deterministic_max_abs"])
            and permutation <= float(gate["candidate_permutation_max_abs"])
            and nohistory == 0.0
            and noquery == 0.0
            and repeat == 0.0
            and finite
        )
    primary_state = models[PRIMARY].components(query, history, candidates)
    checks["primary_mass_conservation"] = bool(
        torch.allclose(
            primary_state["candidate_mass"].sum(dim=1) + primary_state["null_mass"],
            torch.ones_like(primary_state["null_mass"]),
            atol=1e-6,
            rtol=0,
        )
    )
    checks["primary_zero_sum"] = float(primary_state["correction"].sum().abs().cpu()) <= float(
        gate["correction_sum_max_abs"]
    )
    checks["forced_removes_null_only"] = bool(
        models[FORCED_LOGIT_FLOW].components(query, history, candidates)["null_mass"].eq(0).all()
    )
    checks["vector_keeps_partial_mass"] = (
        models[PARTIAL_VECTOR_WRITE].components(query, history, candidates)["candidate_mass"]
        is not None
    )
    checks["global_removes_candidate_event_allocation"] = (
        models[GLOBAL_VECTOR_WRITE].components(query, history, candidates)["candidate_mass"]
        is None
    )
    return {"checks": checks, "diagnostics": diagnostics_by_mode, "expected_parameters": expected_parameters}


def run(config_path: str | Path, device: torch.device) -> dict[str, Any]:
    config = load_config(config_path)
    lock_hash = verify_lock(config)
    assert_gpu(config, device)
    output = Path(config["paths"]["report"])
    if output.exists():
        raise FileExistsError(output)
    started = time.time()
    synthetic = config["synthetic"]
    train = make_rows(
        config,
        int(synthetic["train_requests"]),
        int(synthetic["generator_seed"]),
    )
    evaluation = make_rows(
        config,
        int(synthetic["eval_requests"]),
        int(synthetic["generator_seed"]) + 1,
    )
    base = base_metrics(evaluation)
    seeds = [int(value) for value in config["training"]["seeds"]]
    seed_reports = {}
    all_models = {}
    initial_primary = []
    for seed in seeds:
        mode_reports = {}
        models = {}
        initial_hashes = {}
        for mode in MODES:
            seed_all(seed)
            model = make_model(config, seed, mode)
            initial = state_sha256(model)
            initial_hashes[mode] = initial
            training = train_mode(model, train, config, seed, device)
            final = state_sha256(model)
            clean, corrections = evaluate(model, evaluation, device, "clean")
            wrong, _ = evaluate(model, evaluation, device, "wrong")
            models[mode] = model
            mode_reports[mode] = {
                "parameters": model.trainable_parameter_count(),
                "initial_state_sha256": initial,
                "final_state_sha256": final,
                "parameters_updated": initial != final,
                "training": training,
                "clean_ndcg10": float(clean.mean()),
                "wrong_ndcg10": float(wrong.mean()),
                "clean_minus_base": float((clean - base).mean()),
                "clean_minus_wrong": float((clean - wrong).mean()),
                "correction_sum_max_abs": max(
                    float(abs(np.asarray(row, dtype=np.float64).sum())) for row in corrections
                ),
            }
        primary_diagnostics = diagnostics(models[PRIMARY], evaluation, device)
        structural = structural_checks(models, evaluation, config, device)
        initial_primary.append(initial_hashes[PRIMARY])
        seed_reports[str(seed)] = {
            "paired_initialization": len(set(initial_hashes.values())) == 1,
            "mode_reports": mode_reports,
            "primary_diagnostics": primary_diagnostics,
            "structural": structural,
        }
        all_models[seed] = models

    d0_checks = {
        "paired_initialization": all(row["paired_initialization"] for row in seed_reports.values()),
        "seed_specific_initialization": len(set(initial_primary)) == len(seeds),
        "all_parameters_updated": all(
            row["mode_reports"][mode]["parameters_updated"]
            for row in seed_reports.values()
            for mode in MODES
        ),
        "all_gradients_active": all(
            set(row["mode_reports"][mode]["training"]["gradient_parameter_names"])
            == {"down", "up"}
            for row in seed_reports.values()
            for mode in MODES
        ),
        "all_training_finite": all(
            row["mode_reports"][mode]["training"]["finite"]
            for row in seed_reports.values()
            for mode in MODES
        ),
        "all_structural_checks": all(
            all(row["structural"]["checks"].values()) for row in seed_reports.values()
        ),
        "repository_dataset_read": False,
        "train_labels_read": False,
        "dev_test_qrels_read": False,
    }
    gate = config["gate"]
    d1_checks = {}
    for seed in seeds:
        row = seed_reports[str(seed)]
        primary = row["mode_reports"][PRIMARY]
        d1_checks[f"seed_{seed}_clean_ndcg"] = primary["clean_ndcg10"] >= float(
            gate["clean_ndcg_min"]
        )
        d1_checks[f"seed_{seed}_over_base"] = primary["clean_minus_base"] >= float(
            gate["over_base_min"]
        )
        d1_checks[f"seed_{seed}_clean_wrong"] = primary["clean_minus_wrong"] >= float(
            gate["clean_minus_wrong_min"]
        )
        for control in CONTROLS:
            d1_checks[f"seed_{seed}_over_{control}"] = (
                primary["clean_ndcg10"] - row["mode_reports"][control]["clean_ndcg10"]
                >= float(gate["over_each_control_min"])
            )
        diagnostics_row = row["primary_diagnostics"]
        d1_checks[f"seed_{seed}_signal_mass"] = diagnostics_row[
            "signal_candidate_mass_mean"
        ] >= float(gate["signal_candidate_mass_min"])
        d1_checks[f"seed_{seed}_irrelevant_null"] = diagnostics_row[
            "irrelevant_null_mass_mean"
        ] >= float(gate["irrelevant_null_mass_min"])
        d1_checks[f"seed_{seed}_wrong_null"] = diagnostics_row[
            "wrong_null_mass_mean"
        ] >= float(gate["wrong_null_mass_min"])
        d1_checks[f"seed_{seed}_zero_sum"] = diagnostics_row[
            "correction_sum_max_abs"
        ] <= float(gate["correction_sum_max_abs"])
    report = {
        "candidate_id": "c44",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "physical_gpu": int(config["resources"]["physical_gpu"]),
        "design_lock_sha256": lock_hash,
        "D0": {"status": "passed" if all(d0_checks.values()) else "failed", "checks": d0_checks},
        "D1": {"status": "passed" if all(d1_checks.values()) else "failed", "checks": d1_checks},
        "base_ndcg10": float(base.mean()),
        "seed_reports": seed_reports,
        "status": "passed_design_gate" if all(d0_checks.values()) and all(d1_checks.values()) else "failed_D1_terminal",
        "repository_dataset_read": False,
        "train_labels_read": False,
        "dev_test_qrels_read": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    temporary.replace(output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    report = run(args.config, torch.device(args.device))
    print(json.dumps({"candidate_id": "c44", "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
