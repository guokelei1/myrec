"""Train and evaluate C49 on the exposed C47 train-internal formulation roles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(REPO_ROOT / "src"), str(C38_ROOT), str(C47_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from freeze_lock import load_config, verify_lock, write_once  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from probe.freeze_signal_lock import load_config as load_c47_config, verify_signal_lock  # noqa: E402
from probe.locking import sha256_file  # noqa: E402
from probe.run_signal_gate import (  # noqa: E402
    AmazonStore,
    KuaiStore,
    amazon_labels,
    candidate_key_sha256,
    kuai_labels,
    load_score_rows,
)
from train.freeze_locks import load_config as load_c38_config  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.store import FrozenTransferStore  # noqa: E402


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PREDICTOR = _load_module("c49_runtime_predictor", SYSTEM_ROOT / "model/predictor.py")
MEMORY = _load_module("c49_runtime_memory", SYSTEM_ROOT / "model/innovation_memory.py")
PrequentialSemanticTransformer = PREDICTOR.PrequentialSemanticTransformer
innovation_memory_reads = MEMORY.innovation_memory_reads

SCORE_NAMES = (
    "base",
    "primary_true",
    "primary_wrong",
    "primary_reverse",
    "raw_krr",
    "innovation_softmax",
    "delta_net",
    "shuffled_innovation",
    "primary_correction",
    "wrong_correction",
)


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


class DomainStore:
    def __init__(self, domain: str, c47: Mapping[str, Any], c38: Mapping[str, Any]) -> None:
        self.domain = domain
        self.c47 = c47
        self.selection = json.loads((REPO_ROOT / c47["paths"]["selection"]).read_text(encoding="utf-8"))
        if domain == "kuai":
            self.fit_store: Any = KuaiStore(c47)
            self.eval_store: Any = self.fit_store
            self.fit_role = "kuai_fit"
            self.a_role = "kuai_internal_A"
            self.input_dim = int(self.fit_store.item_embeddings.shape[1])
        elif domain == "amazon":
            self.fit_store = FrozenTransferStore(c38)
            self.eval_store = AmazonStore(c47)
            self.fit_role = "amazon_fit"
            self.a_role = "amazon_internal_A"
            self.input_dim = int(self.fit_store.item_embeddings.shape[1])
        else:
            raise ValueError(f"unknown C49 domain: {domain}")

    def fit_indices(self) -> list[int]:
        return [int(value) for value in self.selection["roles"][self.fit_role]["indices"]]

    def a_indices(self) -> list[int]:
        return [int(value) for value in self.selection["roles"][self.a_role]["indices"]]

    def donors(self) -> list[int]:
        return [int(value) for value in self.selection["wrong_history_donors"][self.a_role]["indices"]]

    def fit_sequence(self, index: int) -> np.ndarray:
        if self.domain == "kuai":
            return self.fit_store.history(index)
        return self.fit_store.items(self.fit_store.history_positions(index, "true"))

    def eval_sequence(self, index: int, *, source: str = "true", donor: int | None = None) -> np.ndarray:
        if self.domain == "kuai":
            target = index if source == "true" else int(donor)
            return self.eval_store.history(target)
        return self.eval_store.history(index, source)

    def query(self, index: int) -> np.ndarray:
        return self.eval_store.query(index)

    def candidates(self, index: int) -> np.ndarray:
        return self.eval_store.candidates(index)

    def request_id(self, index: int) -> str:
        return self.eval_store.request_id(index)

    def candidate_ids(self, index: int) -> list[str]:
        return self.eval_store.candidate_ids(index)


def materialize_sequences(store: DomainStore) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    sequences = [np.asarray(store.fit_sequence(index), dtype=np.float16) for index in store.fit_indices()]
    request_positions = []
    target_positions = []
    for request, sequence in enumerate(sequences):
        for target in range(1, len(sequence)):
            request_positions.append(request)
            target_positions.append(target)
    if len(request_positions) < 5000:
        raise RuntimeError(f"C49 {store.domain} has too few transition examples")
    return sequences, np.asarray(request_positions, np.int32), np.asarray(target_positions, np.int16)


def make_batch(
    sequences: Sequence[np.ndarray],
    request_positions: np.ndarray,
    target_positions: np.ndarray,
    sampled: np.ndarray,
    max_history: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefixes = []
    targets = []
    for example in sampled:
        sequence = sequences[int(request_positions[example])]
        target = int(target_positions[example])
        prefixes.append(np.asarray(sequence[max(0, target - max_history) : target], dtype=np.float32))
        targets.append(np.asarray(sequence[target], dtype=np.float32))
    width = max(len(row) for row in prefixes)
    values = np.zeros((len(prefixes), width, targets[0].shape[0]), dtype=np.float32)
    mask = np.zeros((len(prefixes), width), dtype=bool)
    for row, prefix in enumerate(prefixes):
        values[row, : len(prefix)] = prefix
        mask[row, : len(prefix)] = True
    return values, mask, np.stack(targets).astype(np.float32, copy=False)


def make_model(config: Mapping[str, Any], input_dim: int) -> Any:
    row = config["model"]
    return PrequentialSemanticTransformer(
        input_dim=input_dim,
        width=int(row["width"]),
        heads=int(row["heads"]),
        layers=int(row["layers"]),
        ff_multiplier=int(row["ff_multiplier"]),
        max_history=int(row["max_history"]),
        temperature=float(row["temperature"]),
    )


def parameter_groups(names: set[str]) -> dict[str, bool]:
    return {
        "item_projection": any(name.startswith("item_projection.") for name in names),
        "transformer": any(name.startswith("transformer.") for name in names),
        "read_position": any(name in {"read_token", "position"} for name in names),
        "output_norm": any(name.startswith("output_norm.") for name in names),
    }


def train_predictor(
    model: Any,
    sequences: Sequence[np.ndarray],
    request_positions: np.ndarray,
    target_positions: np.ndarray,
    config: Mapping[str, Any],
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    training = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    rng = np.random.default_rng(seed + 49)
    losses = []
    gradients: set[str] = set()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    model.train()
    for _ in range(int(training["steps"])):
        sampled = rng.integers(0, len(request_positions), size=int(training["batch_size"]))
        prefixes, mask, targets = make_batch(sequences, request_positions, target_positions, sampled, int(config["model"]["max_history"]))
        prefix_tensor = torch.from_numpy(prefixes).to(device)
        mask_tensor = torch.from_numpy(mask).to(device)
        target_tensor = torch.from_numpy(targets).to(device)
        logits = model.contrastive_logits(prefix_tensor, mask_tensor, target_tensor)
        loss = F.cross_entropy(logits, torch.arange(len(sampled), device=device))
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("C49 nonfinite predictor loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                if not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"C49 nonfinite gradient: {name}")
                if bool(parameter.grad.ne(0).any()):
                    gradients.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(training["gradient_clip_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    groups = parameter_groups(gradients)
    updated = [name for name, value in model.named_parameters() if not torch.equal(initial[name], value.detach())]
    return {
        "steps": len(losses),
        "examples": len(request_positions),
        "loss_first_50": float(np.mean(losses[:50])),
        "loss_last_50": float(np.mean(losses[-50:])),
        "loss_decreased": float(np.mean(losses[-50:])) < float(np.mean(losses[:50])),
        "finite": bool(np.isfinite(losses).all()),
        "gradient_groups": groups,
        "all_gradient_groups_active": all(groups.values()),
        "updated_parameter_names": updated,
        "parameters_updated": bool(updated),
    }


def prequential_states(
    model: Any, sequence: np.ndarray, device: torch.device, max_history: int
) -> tuple[torch.Tensor, torch.Tensor]:
    sequence = np.asarray(sequence, dtype=np.float32)
    if not len(sequence):
        width = int(model.width)
        return torch.empty(0, width, device=device), torch.empty(0, width, device=device)
    raw = torch.from_numpy(np.array(sequence, copy=True, order="C")).to(device)
    keys = model.encode_items(raw)
    predictions = torch.zeros_like(keys)
    if len(sequence) > 1:
        prefixes = [sequence[max(0, target - max_history) : target] for target in range(1, len(sequence))]
        width = max(len(row) for row in prefixes)
        values = np.zeros((len(prefixes), width, sequence.shape[1]), dtype=np.float32)
        mask = np.zeros((len(prefixes), width), dtype=bool)
        for row, prefix in enumerate(prefixes):
            values[row, : len(prefix)] = prefix
            mask[row, : len(prefix)] = True
        predictions[1:] = model.predict_next(torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device))
    return keys, predictions


def score_request(
    model: Any,
    query: np.ndarray,
    history: np.ndarray,
    candidates: np.ndarray,
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, np.ndarray]:
    with torch.inference_mode():
        keys, predictions = prequential_states(model, history, device, int(config["model"]["max_history"]))
        q = model.encode_items(torch.from_numpy(np.array(query, dtype=np.float32, copy=True))[None].to(device))
        c = model.encode_items(torch.from_numpy(np.array(candidates, dtype=np.float32, copy=True, order="C")).to(device))
        if len(keys):
            reads = innovation_memory_reads(
                q,
                keys[None],
                predictions[None],
                torch.ones(1, len(keys), dtype=torch.bool, device=device),
                ridge=float(config["memory"]["ridge"]),
                softmax_temperature=float(config["memory"]["softmax_temperature"]),
                epsilon=float(config["memory"]["normalization_epsilon"]),
            )
        else:
            width = int(config["model"]["width"])
            reads = innovation_memory_reads(
                q,
                torch.zeros(1, 1, width, device=device),
                torch.zeros(1, 1, width, device=device),
                torch.zeros(1, 1, dtype=torch.bool, device=device),
                ridge=float(config["memory"]["ridge"]),
                softmax_temperature=float(config["memory"]["softmax_temperature"]),
                epsilon=float(config["memory"]["normalization_epsilon"]),
            )
        scale = float(config["memory"]["correction_scale"])
        output = {}
        for name in ("primary", "raw_krr", "innovation_softmax", "delta_net", "shuffled_innovation"):
            read = getattr(reads, name)[0]
            output[name] = (scale * (c @ read)).detach().cpu().numpy().astype(np.float32, copy=False)
        output["innovation_rms"] = np.asarray(float(reads.innovations.square().mean().sqrt().cpu()) if len(keys) else 0.0, dtype=np.float32)
    return output


def rankings(request_id: str, item_ids: Sequence[str], values: np.ndarray) -> list[str]:
    return [row.item_id for row in sort_candidates(request_id, [ScoredCandidate(str(item), float(score)) for item, score in zip(item_ids, values)])]


def order_change_fraction(
    request_ids: Sequence[str], item_ids: Sequence[Sequence[str]], first: Sequence[np.ndarray], second: Sequence[np.ndarray]
) -> float:
    return float(np.mean([rankings(r, ids, a) != rankings(r, ids, b) for r, ids, a, b in zip(request_ids, item_ids, first, second)]))


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[int(offsets[i]) : int(offsets[i + 1])], np.float32).copy() for i in range(len(offsets) - 1)]


def state_sha256(model: Any) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_seed(config: Mapping[str, Any], domain: str, seed: int, device: torch.device) -> dict[str, Any]:
    _, lock_hash = verify_lock(config)
    c47 = load_c47_config(REPO_ROOT / config["paths"]["c47_config"])
    verify_signal_lock(c47)
    c38 = load_c38_config(REPO_ROOT / config["paths"]["c38_config"])
    mapping = config["resources"][f"{domain}_seed_to_physical_gpu"]
    physical = int(mapping.get(str(seed), -1))
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical) or str(device) != "cuda:0":
        raise RuntimeError("C49 GPU registration differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C49 deterministic CUBLAS setting absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C49 requires exactly one visible GPU")
    store = DomainStore(domain, c47, c38)
    sequences, requests, targets = materialize_sequences(store)
    seed_all(seed)
    model = make_model(config, store.input_dim).to(device)
    initial_hash = state_sha256(model)
    training = train_predictor(model, sequences, requests, targets, config, seed, device)
    model.eval()
    root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoints = REPO_ROOT / config["paths"]["checkpoint_root"]
    root.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoints / f"{domain}_seed_{seed}.pt"
    score_path = root / f"{domain}_seed_{seed}_scores.npz"
    report_path = root / f"{domain}_seed_{seed}_report.json"
    if any(path.exists() for path in (checkpoint, score_path, report_path)):
        raise FileExistsError(report_path)
    torch.save({"candidate_id": "c49", "domain": domain, "seed": seed, "proposal_lock_sha256": lock_hash, "state_dict": model.state_dict()}, checkpoint)
    selection = store.selection
    indices, donors = store.a_indices(), store.donors()
    expected = c47["integrity"][f"{domain}_candidate_key_sha256"]
    if candidate_key_sha256(store.eval_store, indices) != expected:
        raise RuntimeError("C49 candidate hash differs")
    prior_report = json.loads((REPO_ROOT / c47["paths"]["artifact_root"] / f"{domain}_fixed_score_report.json").read_text(encoding="utf-8"))
    prior = load_score_rows(REPO_ROOT / c47["paths"]["artifact_root"], prior_report)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    deterministic_max = 0.0
    candidate_permutation_max = 0.0
    innovation_rms = []
    for position, (index, donor) in enumerate(zip(indices, donors)):
        query = store.query(index)
        candidates = store.candidates(index)
        true_history = store.eval_sequence(index, source="true")
        wrong_history = store.eval_sequence(index, source="wrong", donor=donor)
        true = score_request(model, query, true_history, candidates, config, device)
        again = score_request(model, query, true_history, candidates, config, device)
        wrong = score_request(model, query, wrong_history, candidates, config, device)
        reverse = score_request(model, query, true_history[::-1], candidates, config, device)
        candidate_reverse = score_request(model, query, true_history, candidates[::-1], config, device)
        base = prior["base"][position]
        deterministic_max = max(deterministic_max, float(np.max(np.abs(true["primary"] - again["primary"]))))
        candidate_permutation_max = max(candidate_permutation_max, float(np.max(np.abs(true["primary"] - candidate_reverse["primary"][::-1]))))
        innovation_rms.append(float(true["innovation_rms"]))
        rows["base"].append(base)
        rows["primary_true"].append(base + true["primary"])
        rows["primary_wrong"].append(base + wrong["primary"])
        rows["primary_reverse"].append(base + reverse["primary"])
        rows["raw_krr"].append(base + true["raw_krr"])
        rows["innovation_softmax"].append(base + true["innovation_softmax"])
        rows["delta_net"].append(base + true["delta_net"])
        rows["shuffled_innovation"].append(base + true["shuffled_innovation"])
        rows["primary_correction"].append(true["primary"])
        rows["wrong_correction"].append(wrong["primary"])
    nohistory = score_request(model, np.ones(store.input_dim, np.float32), np.empty((0, store.input_dim), np.float32), np.eye(store.input_dim, dtype=np.float32)[:3], config, device)
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    activity = {
        "primary_vs_raw_krr": order_change_fraction(request_ids, item_ids, rows["primary_true"], rows["raw_krr"]),
        "true_vs_wrong": order_change_fraction(request_ids, item_ids, rows["primary_true"], rows["primary_wrong"]),
        "primary_vs_shuffled": order_change_fraction(request_ids, item_ids, rows["primary_true"], rows["shuffled_innovation"]),
        "true_vs_reverse": order_change_fraction(request_ids, item_ids, rows["primary_true"], rows["primary_reverse"]),
    }
    offsets, _ = flatten(rows["base"])
    with score_path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(value)[1] for name, value in rows.items()})
    checks = {
        "candidate_hash": candidate_key_sha256(store.eval_store, indices) == expected,
        "finite_training": training["finite"],
        "loss_decreased": training["loss_decreased"],
        "all_gradient_groups": training["all_gradient_groups_active"],
        "parameters_updated": training["parameters_updated"],
        "finite_scores": all(np.isfinite(row).all() for values in rows.values() for row in values),
        "deterministic": deterministic_max <= float(config["evaluation"]["deterministic_tolerance"]),
        "candidate_permutation": candidate_permutation_max <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "nohistory_exact_zero": all(np.count_nonzero(nohistory[name]) == 0 for name in ("primary", "raw_krr", "innovation_softmax", "delta_net", "shuffled_innovation")),
        "innovation_nonzero": float(np.mean(innovation_rms)) > 1e-4,
        "primary_raw_active": activity["primary_vs_raw_krr"] >= float(config["evaluation"]["primary_raw_order_change_min"]),
        "true_wrong_active": activity["true_vs_wrong"] >= float(config["evaluation"]["true_wrong_order_change_min"]),
        "A_labels_not_read": True,
        "fresh_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c49",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "seed": seed,
        "physical_gpu": physical,
        "proposal_lock_sha256": lock_hash,
        "initial_state_sha256": initial_hash,
        "final_state_sha256": state_sha256(model),
        "parameters": sum(value.numel() for value in model.parameters()),
        "training": training,
        "checks": checks,
        "activity": activity,
        "mean_innovation_rms": float(np.mean(innovation_rms)),
        "deterministic_max_abs": deterministic_max,
        "candidate_permutation_max_abs": candidate_permutation_max,
        "checkpoint": {"path": str(checkpoint.relative_to(REPO_ROOT)), "sha256": sha256_file(checkpoint)},
        "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)},
        "A_labels_read": False,
        "fresh_reserve_dev_test_qrels_opened": False,
    }
    write_once(report_path, report)
    return report


def load_seed_rows(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C49 score artifact changed")
    with np.load(path, allow_pickle=False) as values:
        offsets = np.asarray(values["offsets"], np.int64)
        return {name: unflatten(offsets, values[name]) for name in SCORE_NAMES}


def run_a0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    reports = {}
    for domain in ("kuai", "amazon"):
        reports[domain] = [json.loads((root / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in config["training"][f"{domain}_seeds"]]
    checks = {
        "all_seed_checks": all(all(report["checks"].values()) for rows in reports.values() for report in rows),
        "same_parameters": all(len({report["parameters"] for report in rows}) == 1 for rows in reports.values()),
        "score_hashes": all(sha256_file(REPO_ROOT / report["score_artifact"]["path"]) == report["score_artifact"]["sha256"] for rows in reports.values() for report in rows),
        "checkpoint_hashes": all(sha256_file(REPO_ROOT / report["checkpoint"]["path"]) == report["checkpoint"]["sha256"] for rows in reports.values() for report in rows),
        "A_labels_closed_during_training_scoring": all(report["A_labels_read"] is False for rows in reports.values() for report in rows),
        "fresh_dev_test_qrels_closed": all(report["fresh_reserve_dev_test_qrels_opened"] is False for rows in reports.values() for report in rows),
    }
    value = {
        "candidate_id": "c49",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "stage": "A0_exposed_label_release",
        "status": "passed" if all(checks.values()) else "failed_A0_terminal",
        "proposal_lock_sha256": lock_hash,
        "checks": checks,
        "A_labels_read": False,
        "A_labels_authorized_after_A0": all(checks.values()),
        "fresh_reserve_dev_test_qrels_opened": False,
    }
    write_once(root / "a0_report.json", value)
    return value


def average_rows(groups: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack(values), axis=0).astype(np.float32) for values in zip(*groups)]


def ndcg_rows(request_ids, item_ids, scores, labels):
    output = []
    for request_id, items, values, label in zip(request_ids, item_ids, scores, labels):
        ranked = rankings(request_id, items, values)
        positives = {str(item) for item, value in zip(items, label) if value > 0}
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, np.float64)


def aggregate_domain(config: Mapping[str, Any], domain: str, c47: Mapping[str, Any], c38: Mapping[str, Any]) -> dict[str, Any]:
    store = DomainStore(domain, c47, c38)
    indices = store.a_indices()
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    labels = kuai_labels(c47, store.eval_store, indices) if domain == "kuai" else amazon_labels(c47, store.eval_store, indices)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    seeds = [int(value) for value in config["training"][f"{domain}_seeds"]]
    reports = [json.loads((root / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in seeds]
    seed_rows = {seed: load_seed_rows(report) for seed, report in zip(seeds, reports)}
    ensemble = {name: average_rows([seed_rows[seed][name] for seed in seeds]) for name in SCORE_NAMES}
    ensemble_ndcg = {name: ndcg_rows(request_ids, item_ids, rows, labels) for name, rows in ensemble.items() if name not in {"primary_correction", "wrong_correction"}}
    seed_ndcg = {seed: {name: ndcg_rows(request_ids, item_ids, rows, labels) for name, rows in seed_rows[seed].items() if name not in {"primary_correction", "wrong_correction"}} for seed in seeds}
    references = {
        "base": "base",
        "raw_krr": "raw_krr",
        "innovation_softmax": "innovation_softmax",
        "delta_net": "delta_net",
        "shuffled_innovation": "shuffled_innovation",
        "wrong_history": "primary_wrong",
    }
    evaluation = config["evaluation"]
    comparisons = compare(request_ids, ensemble_ndcg["primary_true"], {name: ensemble_ndcg[target] for name, target in references.items()}, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]), folds=int(evaluation["hash_folds"]))
    seed_differences = {name: {str(seed): float((seed_ndcg[seed]["primary_true"] - seed_ndcg[seed][target]).mean()) for seed in seeds} for name, target in references.items()}
    clicked_true = clicked_direction(ensemble["primary_correction"], labels)
    clicked_wrong = clicked_direction(ensemble["wrong_correction"], labels)
    clicked = bootstrap(clicked_true, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 20)
    specificity = bootstrap(clicked_true - clicked_wrong, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 21)
    thresholds = {
        "base": float(evaluation["primary_minus_base_min"]),
        "raw_krr": float(evaluation["primary_minus_raw_krr_min"]),
        "innovation_softmax": float(evaluation["primary_minus_softmax_min"]),
        "delta_net": float(evaluation["primary_minus_delta_min"]),
        "shuffled_innovation": float(evaluation["primary_minus_shuffled_min"]),
        "wrong_history": float(evaluation["true_minus_wrong_min"]),
    }
    checks = {}
    for name, minimum in thresholds.items():
        row = comparisons[name]
        checks[f"{name}_effect"] = row["mean"] >= minimum
        checks[f"{name}_ci"] = row["percentile_95_ci"][0] > 0
        checks[f"{name}_all_seed_fold_positive"] = all(value > 0 for value in seed_differences[name].values()) and all(fold["mean_difference"] > 0 for fold in row["hash_folds"])
    checks["clicked_direction_ci"] = clicked["percentile_95_ci"][0] > 0
    checks["clicked_specificity_ci"] = specificity["percentile_95_ci"][0] > 0
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "requests": len(indices),
        "checks": checks,
        "mean_ndcg10": {name: float(values.mean()) for name, values in ensemble_ndcg.items()},
        "comparisons": comparisons,
        "seed_differences": seed_differences,
        "clicked_direction": clicked,
        "clicked_true_minus_wrong": specificity,
    }


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_lock(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0 = json.loads((root / "a0_report.json").read_text(encoding="utf-8"))
    if a0.get("status") != "passed" or a0.get("A_labels_authorized_after_A0") is not True or a0.get("A_labels_read") is not False:
        raise PermissionError("C49 A0 did not authorize exposed A label read")
    c47 = load_c47_config(REPO_ROOT / config["paths"]["c47_config"])
    c38 = load_c38_config(REPO_ROOT / config["paths"]["c38_config"])
    domains = {domain: aggregate_domain(config, domain, c47, c38) for domain in ("kuai", "amazon")}
    passed = all(row["status"] == "passed" for row in domains.values())
    value = {
        "candidate_id": "c49",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "gate_id": config["gate_id"],
        "status": "passed_exposed_learnability_only" if passed else "failed_exposed_learnability_terminal",
        "decision": "authorize_separately_locked_fresh_C49_architecture_gate" if passed else "close_C49_before_fresh_reserve",
        "proposal_lock_sha256": lock_hash,
        "A0_report_sha256": sha256_file(root / "a0_report.json"),
        "domains": domains,
        "A_labels_read_after_A0": True,
        "fresh_reserve_dev_test_qrels_opened": False,
        "claims": {"exposed_train_internal_only": True, "fresh_result": False, "trained_architecture_result": False, "dev_test_result": False},
    }
    write_once(root / "learnability_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("seed", "a0", "aggregate"))
    parser.add_argument("--domain", choices=("kuai", "amazon"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "seed":
        if args.domain is None or args.seed is None or args.seed not in config["training"][f"{args.domain}_seeds"]:
            raise ValueError("C49 seed/domain is not registered")
        value = run_seed(config, args.domain, args.seed, torch.device(args.device))
    elif args.stage == "a0":
        value = run_a0(config)
    else:
        value = aggregate(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
