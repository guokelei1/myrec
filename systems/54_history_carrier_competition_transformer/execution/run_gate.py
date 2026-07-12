"""Run C54 data-free D0 and exposed-label-free mechanics gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(C38_ROOT), str(REPO_ROOT / "src"), str(SYSTEM_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from execution.freeze_locks import (  # noqa: E402
    load_config, sha256_file, verify_execution, verify_proposal, write_once,
)
from model.history_carrier import HistoryCarrierCompetitionTransformer  # noqa: E402
from myrec.analysis.supervised_diagnostics import PackedRequestData  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, sort_candidates  # noqa: E402
from train.freeze_locks import load_config as load_c38_config  # noqa: E402
from train.store import FrozenTransferStore  # noqa: E402


SCORE_NAMES = (
    "base", "primary", "independent", "factual", "raw", "wrong",
    "correction", "wrong_correction",
)


class LocalCompactLabels:
    def __init__(self, root: Path, prefix: str) -> None:
        self.indices = np.load(root / f"{prefix}_fit_indices.npy", mmap_mode="r")
        self.offsets = np.load(root / f"{prefix}_fit_label_offsets.npy", mmap_mode="r")
        self.values = np.load(root / f"{prefix}_fit_labels.npy", mmap_mode="r")
        self.positions = {int(value): row for row, value in enumerate(self.indices)}

    def row(self, index: int, count: int) -> np.ndarray:
        position = self.positions[int(index)]
        start, stop = int(self.offsets[position]), int(self.offsets[position + 1])
        if stop - start != count:
            raise ValueError("C54 compact label count differs")
        return np.asarray(self.values[start:stop], np.float32).copy()


class DomainData:
    def __init__(self, domain: str, config: Mapping[str, Any]) -> None:
        self.domain = domain
        paths = config["paths"]
        self.selection = json.loads((REPO_ROOT / paths["c47_selection"]).read_text(encoding="utf-8"))
        self.root = REPO_ROOT / paths["c53_artifact_root"]
        self.labels = LocalCompactLabels(self.root, domain)
        if domain == "kuai":
            self.data = PackedRequestData.load(
                (REPO_ROOT / paths["kuai_packed_root"]).parent, "train"
            )
            feature = np.load(self.root / "kuai_feature_indices.npy", mmap_mode="r")
            self.feature_position = {int(index): row for row, index in enumerate(feature)}
            self.item_embeddings = np.load(self.root / "kuai_item_states.npy", mmap_mode="r")
            self.query_embeddings = np.load(self.root / "kuai_query_states.npy", mmap_mode="r")
            offsets = np.load(self.root / "kuai_base_offsets.npy", mmap_mode="r")
            scores = np.load(self.root / "kuai_base_scores.npy", mmap_mode="r")
            self.base_rows = {
                int(index): np.asarray(
                    scores[int(offsets[row]):int(offsets[row + 1])], np.float32,
                ).copy()
                for row, index in enumerate(feature)
            }
            self.fit_store = self.eval_store = None
            self.fit_indices = [int(value) for value in self.labels.indices]
            self.a_indices = [int(value) for value in self.selection["roles"]["kuai_internal_A"]["indices"]]
            donors = self.selection["wrong_history_donors"]["kuai_internal_A"]["indices"]
            self.donor = dict(zip(self.a_indices, map(int, donors)))
            self.input_dim = int(self.item_embeddings.shape[1])
        elif domain == "amazon":
            self.fit_store = FrozenTransferStore(load_c38_config(REPO_ROOT / paths["c38_config"]))
            self.eval_store = FrozenTransferStore({
                "paths": {
                    "selection": str(REPO_ROOT / paths["c47_amazon_adapter_selection"]),
                    "feature_root": str(REPO_ROOT / paths["c47_amazon_feature_root"]),
                },
                "model": {"embedding_dim": 384},
            })
            self.fit_indices = [int(value) for value in self.labels.indices]
            self.a_indices = [int(value) for value in self.selection["roles"]["amazon_internal_A"]["indices"]]
            self.input_dim = 384
        else:
            raise ValueError("unknown C54 domain")

    def _store(self, index: int) -> FrozenTransferStore:
        if self.domain != "amazon":
            raise ValueError("C54 Kuai has no transfer store")
        return self.fit_store if index in self.fit_store.feature_position else self.eval_store

    def request_id(self, index: int) -> str:
        return self.data.request_ids[index] if self.domain == "kuai" else self._store(index).request_id(index)

    def candidate_ids(self, index: int) -> list[str]:
        if self.domain == "kuai":
            start, stop = int(self.data.candidate_offsets[index]), int(self.data.candidate_offsets[index + 1])
            return [str(value) for value in self.data.candidate_item_ids[start:stop].tolist()]
        return self._store(index).candidate_ids(index)

    def candidate_positions(self, index: int) -> np.ndarray:
        if self.domain == "kuai":
            start, stop = int(self.data.candidate_offsets[index]), int(self.data.candidate_offsets[index + 1])
            return np.asarray(self.data.candidate_embedding_indices[start:stop], np.int64)
        return self._store(index).candidate_positions(index)

    def history_positions(self, index: int, source: str) -> np.ndarray:
        if self.domain == "kuai":
            target = index if source == "true" else self.donor[index]
            start, stop = int(self.data.history_offsets[target]), int(self.data.history_offsets[target + 1])
            return np.asarray(self.data.history_embedding_indices[start:stop], np.int64)
        return self._store(index).history_positions(index, source)

    def query(self, index: int) -> np.ndarray:
        if self.domain == "kuai":
            return np.asarray(self.query_embeddings[self.feature_position[index]], np.float32)
        return self._store(index).query(index)

    def items(self, positions: np.ndarray, index: int) -> np.ndarray:
        if self.domain == "kuai":
            return np.asarray(self.item_embeddings[positions], np.float32)
        return self._store(index).items(positions)

    def base(self, index: int) -> np.ndarray:
        return self.base_rows[index].copy() if self.domain == "kuai" else self._store(index).base_row(index)

    def fit_label(self, index: int) -> np.ndarray:
        return self.labels.row(index, len(self.candidate_positions(index)))

    def sizes(self, index: int) -> tuple[int, int]:
        return len(self.history_positions(index, "true")), len(self.candidate_positions(index))


def candidate_hash(data: DomainData, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(json.dumps(
            [data.request_id(index), *data.candidate_ids(index)], separators=(",", ":"),
        ).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def batches(
    data: DomainData, indices: Sequence[int], config: Mapping[str, Any],
    *, seed: int, shuffle: bool,
) -> list[list[int]]:
    values = list(map(int, indices))
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    maximum_requests = int(config["training"]["max_requests_per_batch"])
    maximum_tokens = int(config["training"]["max_sequence_tokens_per_batch"])
    output: list[list[int]] = []
    current: list[int] = []
    max_length = 0
    for index in values:
        h, c = data.sizes(index); length = 1 + h + c
        next_max = max(max_length, length); next_count = len(current) + 1
        if current and (next_count > maximum_requests or next_count * next_max > maximum_tokens):
            output.append(current); current = []; max_length = 0
        current.append(index); max_length = max(max_length, length)
    if current:
        output.append(current)
    return output


def collate(
    data: DomainData, indices: Sequence[int], *, source: str,
    labels: bool, device: torch.device,
) -> dict[str, torch.Tensor]:
    history_rows = [data.history_positions(index, source) for index in indices]
    candidate_rows = [data.candidate_positions(index) for index in indices]
    max_h = max(1, max(map(len, history_rows))); max_c = max(map(len, candidate_rows))
    query = np.stack([data.query(index) for index in indices]).astype(np.float32)
    history = np.zeros((len(indices), max_h, data.input_dim), np.float32)
    history_mask = np.zeros((len(indices), max_h), bool)
    candidates = np.zeros((len(indices), max_c, data.input_dim), np.float32)
    candidate_mask = np.zeros((len(indices), max_c), bool)
    base = np.zeros((len(indices), max_c), np.float32)
    target = np.zeros((len(indices), max_c), np.float32)
    for row, index in enumerate(indices):
        hp, cp = history_rows[row], candidate_rows[row]
        if len(hp):
            history[row, :len(hp)] = data.items(hp, index)
            history_mask[row, :len(hp)] = True
        candidates[row, :len(cp)] = data.items(cp, index)
        candidate_mask[row, :len(cp)] = True
        base[row, :len(cp)] = data.base(index)
        if labels:
            target[row, :len(cp)] = data.fit_label(index)
    result = {
        "query": torch.from_numpy(query).to(device),
        "history": torch.from_numpy(history).to(device),
        "history_mask": torch.from_numpy(history_mask).to(device),
        "candidates": torch.from_numpy(candidates).to(device),
        "candidate_mask": torch.from_numpy(candidate_mask).to(device),
        "base_scores": torch.from_numpy(base).to(device),
    }
    if labels:
        result["labels"] = torch.from_numpy(target).to(device)
    return result


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positive = labels.gt(0) & mask
    if not bool(positive.any(dim=-1).all()):
        raise ValueError("C54 training row lacks positive")
    negative = -torch.finfo(scores.dtype).max
    return (
        torch.logsumexp(scores.masked_fill(~mask, negative), dim=-1)
        - torch.logsumexp(scores.masked_fill(~positive, negative), dim=-1)
    ).mean()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)


def required_gradient_groups(names: Sequence[str]) -> dict[str, bool]:
    return {
        prefix: any(name.startswith(prefix) for name in names)
        for prefix in (
            "content_projection", "base_projection", "context_attention",
            "list_attention", "ffn", "output",
        )
    }


def make_model(config: Mapping[str, Any], input_dim: int) -> HistoryCarrierCompetitionTransformer:
    row = config["model"]
    return HistoryCarrierCompetitionTransformer(
        input_dim=input_dim, hidden_dim=int(row["hidden_dim"]),
        heads=int(row["heads"]), ffn_dim=int(row["ffn_dim"]),
        dropout=float(row["dropout"]), max_history=int(row["max_history"]),
    )


def assert_gpu(config: Mapping[str, Any], device_name: str, physical: int) -> None:
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C54 GPU binding differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C54 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C54 requires one visible GPU")


def d0_inputs(device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(5401)
    return {
        "query": torch.randn(2, 16, generator=generator).to(device),
        "history": torch.randn(2, 4, 16, generator=generator).to(device),
        "history_mask": torch.tensor([[True, True, True, False], [True] * 4], device=device),
        "candidates": torch.randn(2, 7, 16, generator=generator).to(device),
        "candidate_mask": torch.ones(2, 7, dtype=torch.bool, device=device),
        "base_scores": torch.randn(2, 7, generator=generator).to(device),
    }


def run_d0(config: Mapping[str, Any], device_name: str) -> dict[str, Any]:
    _, proposal_hash = verify_proposal(config)
    assert_gpu(config, device_name, int(config["resources"]["d0_physical_gpu"]))
    seed_all(5400); device = torch.device(device_name)
    model = HistoryCarrierCompetitionTransformer(
        input_dim=16, hidden_dim=32, heads=4, ffn_dim=64,
        dropout=0.0, max_history=8,
    ).to(device).eval()
    values = d0_inputs(device)
    with torch.inference_mode():
        primary = model(**values)
        repeated = model(**values)
        independent = model(**values, mode="independent_carrier")
        factual = model(**values, mode="factual_carrier")
        changed_history = dict(values); changed_history["history"] = values["history"].roll(1, 0)
        wrong = model(**changed_history)
        raw = model(**values, mode="raw_candidate")
        raw_wrong = model(**changed_history, mode="raw_candidate")
        empty = dict(values); empty["history_mask"] = torch.zeros_like(values["history_mask"])
        nohistory = model(**empty)
        permutation = torch.arange(values["candidates"].shape[1] - 1, -1, -1, device=device)
        permuted_input = dict(values)
        for name in ("candidates", "candidate_mask", "base_scores"):
            permuted_input[name] = values[name][:, permutation]
        permuted = model(**permuted_input).scores[:, torch.argsort(permutation)]
        one = {name: value[:1].clone() for name, value in values.items()}
        distractor = dict(one); distractor["candidates"] = one["candidates"].clone()
        distractor["candidates"][:, 1] += 3.0
        one_primary = model(**one); distractor_primary = model(**distractor)
        one_independent = model(**one, mode="independent_carrier")
        distractor_independent = model(**distractor, mode="independent_carrier")
    model.train(); train_values = d0_inputs(device)
    weighted = torch.arange(7, dtype=torch.float32, device=device)
    loss = (model(**train_values).scores * weighted).sum(); loss.backward()
    gradient_names = [
        name for name, value in model.named_parameters()
        if value.grad is not None and bool(value.grad.ne(0).any())
    ]
    gradient_groups = required_gradient_groups(gradient_names)
    tolerance = float(config["evaluation"]["candidate_permutation_tolerance"])
    zero_tolerance = float(config["evaluation"]["zero_carrier_tolerance"])
    expected_base = values["base_scores"].masked_fill(~values["candidate_mask"], 0.0)
    checks = {
        "finite": all(torch.isfinite(value).all().item() for value in (primary.scores, primary.carrier, primary.list_message)),
        "deterministic": torch.equal(primary.scores, repeated.scores),
        "candidate_permutation": float((primary.scores - permuted).abs().max().cpu()) <= tolerance,
        "nohistory_exact_base": torch.equal(nohistory.scores, expected_base),
        "nohistory_zero_carrier": float(nohistory.carrier.abs().max().cpu()) <= zero_tolerance,
        "nohistory_zero_list_message": float(nohistory.list_message.abs().max().cpu()) <= zero_tolerance,
        "cross_candidate_functional": not torch.allclose(primary.scores, independent.scores),
        "null_contrast_functional": not torch.allclose(primary.scores, factual.scores),
        "wrong_history_functional": not torch.allclose(primary.correction, wrong.correction),
        "raw_control_history_invariant": torch.equal(raw.correction, raw_wrong.correction),
        "distractor_changes_primary_target": not torch.allclose(one_primary.raw_correction[:, 0], distractor_primary.raw_correction[:, 0]),
        "distractor_preserves_independent_target": torch.allclose(one_independent.raw_correction[:, 0], distractor_independent.raw_correction[:, 0], atol=2e-6),
        "gradients_finite_active": bool(gradient_names) and all(
            torch.isfinite(value.grad).all().item()
            for value in model.parameters() if value.grad is not None
        ),
        "load_bearing_gradient_groups_active": all(gradient_groups.values()),
        "repository_labels_closed": True,
    }
    report = {
        "candidate_id": "c54", "stage": "D0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed_D0_terminal",
        "proposal_lock_sha256": proposal_hash, "checks": checks,
        "parameters": model.parameter_count(), "gradient_groups": gradient_groups,
        "repository_labels_opened": False,
        "fresh_reserve_dev_test_qrels_opened": False,
    }
    path = REPO_ROOT / config["paths"]["artifact_root"] / "d0_report.json"
    write_once(path, report); return report


def rankings(request_id: str, item_ids: Sequence[str], values: np.ndarray) -> list[str]:
    scored = [ScoredCandidate(str(item), float(score)) for item, score in zip(item_ids, values)]
    return [row.item_id for row in sort_candidates(request_id, scored)]


def change(
    data: DomainData, indices: Sequence[int],
    first: Sequence[np.ndarray], second: Sequence[np.ndarray],
) -> dict[str, Any]:
    a = [rankings(data.request_id(index), data.candidate_ids(index), row) for index, row in zip(indices, first)]
    b = [rankings(data.request_id(index), data.candidate_ids(index), row) for index, row in zip(indices, second)]
    any_count = sum(int(x != y) for x, y in zip(a, b))
    top_count = sum(int(set(x[:10]) != set(y[:10])) for x, y in zip(a, b))
    return {
        "requests": len(a), "any_count": any_count,
        "any_fraction": any_count / len(a), "top10_count": top_count,
        "top10_fraction": top_count / len(a),
    }


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def correlation(first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> float:
    a, b = np.concatenate(first), np.concatenate(second)
    if float(a.std()) <= 1e-12 or float(b.std()) <= 1e-12:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def run_seed(
    config: Mapping[str, Any], domain: str, seed: int, device_name: str,
) -> dict[str, Any]:
    _, execution_hash = verify_execution(config)
    physical = int(config["resources"][f"{domain}_seed_to_physical_gpu"][str(seed)])
    assert_gpu(config, device_name, physical); seed_all(seed)
    device = torch.device(device_name); data = DomainData(domain, config)
    expected_hash = config["integrity"][f"{domain}_candidate_key_sha256"]
    actual_hash = candidate_hash(data, data.a_indices)
    if actual_hash != expected_hash:
        raise RuntimeError(f"C54 {domain} candidate key differs")
    model = make_model(config, data.input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epoch_losses: list[list[float]] = []
    gradients: set[str] = set(); model.train()
    for epoch in range(int(config["training"]["epochs"])):
        losses = []
        for request_batch in batches(data, data.fit_indices, config, seed=seed + epoch, shuffle=True):
            batch = collate(data, request_batch, source="true", labels=True, device=device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**{name: value for name, value in batch.items() if name != "labels"})
            loss = listwise_loss(output.scores, batch["labels"], batch["candidate_mask"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("C54 nonfinite loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and bool(parameter.grad.ne(0).any()):
                    gradients.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
            optimizer.step(); losses.append(float(loss.detach().cpu()))
        epoch_losses.append(losses)

    model.eval(); artifact_root = REPO_ROOT / config["paths"]["artifact_root"]
    checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]
    artifact_root.mkdir(parents=True, exist_ok=True); checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / f"{domain}_seed_{seed}.pt"
    score_path = artifact_root / f"{domain}_seed_{seed}_scores.npz"
    report_path = artifact_root / f"{domain}_seed_{seed}_report.json"
    if checkpoint.exists() or score_path.exists() or report_path.exists():
        raise FileExistsError(report_path)
    torch.save({
        "candidate_id": "c54", "domain": domain, "seed": seed,
        "state_dict": model.state_dict(), "execution_lock_sha256": execution_hash,
    }, checkpoint)

    rows: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    deterministic = 0.0; nohistory_exact = True; zero_carrier = 0.0; zero_message = 0.0
    with torch.inference_mode():
        for request_batch in batches(data, data.a_indices, config, seed=0, shuffle=False):
            true = collate(data, request_batch, source="true", labels=False, device=device)
            wrong_input = collate(data, request_batch, source="wrong", labels=False, device=device)
            primary = model(**true); repeated = model(**true)
            independent = model(**true, mode="independent_carrier")
            factual = model(**true, mode="factual_carrier")
            raw = model(**true, mode="raw_candidate")
            wrong = model(**wrong_input)
            empty = dict(true); empty["history_mask"] = torch.zeros_like(true["history_mask"])
            nohistory = model(**empty)
            deterministic = max(deterministic, float((primary.scores - repeated.scores).abs().max().cpu()))
            nohistory_exact = nohistory_exact and bool(torch.equal(
                nohistory.scores, true["base_scores"].masked_fill(~true["candidate_mask"], 0.0),
            ))
            zero_carrier = max(zero_carrier, float(nohistory.carrier.abs().max().cpu()))
            zero_message = max(zero_message, float(nohistory.list_message.abs().max().cpu()))
            mask = true["candidate_mask"].cpu().numpy()
            arrays = {
                "base": true["base_scores"].cpu().numpy(),
                "primary": primary.scores.cpu().numpy(),
                "independent": independent.scores.cpu().numpy(),
                "factual": factual.scores.cpu().numpy(),
                "raw": raw.scores.cpu().numpy(),
                "wrong": wrong.scores.cpu().numpy(),
                "correction": primary.correction.cpu().numpy(),
                "wrong_correction": wrong.correction.cpu().numpy(),
            }
            for row in range(len(request_batch)):
                count = int(mask[row].sum())
                for name in SCORE_NAMES:
                    rows[name].append(np.asarray(arrays[name][row, :count], np.float32).copy())

    permutation_error = 0.0
    with torch.inference_mode():
        for index in data.a_indices[:8]:
            batch = collate(data, [index], source="true", labels=False, device=device)
            original = model(**batch).scores
            count = int(batch["candidate_mask"].sum())
            permutation = torch.arange(count - 1, -1, -1, device=device)
            changed = dict(batch)
            for name in ("candidates", "candidate_mask", "base_scores"):
                changed[name] = changed[name][:, permutation]
            permuted = model(**changed).scores[:, torch.argsort(permutation)]
            permutation_error = max(permutation_error, float((original - permuted).abs().max().cpu()))

    edge_change = change(data, data.a_indices, rows["primary"], rows["independent"])
    contrast_change = change(data, data.a_indices, rows["primary"], rows["factual"])
    wrong_change = change(data, data.a_indices, rows["primary"], rows["wrong"])
    raw_change = change(data, data.a_indices, rows["primary"], rows["raw"])
    offsets, _ = flatten(rows["base"])
    with score_path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(value)[1] for name, value in rows.items()})
    means = [float(np.mean(values)) for values in epoch_losses]
    ev = config["evaluation"]
    checks = {
        "finite_training": all(np.isfinite(value).all() for value in epoch_losses),
        "epoch_loss_decreased": len(means) == 2 and means[1] < means[0],
        "load_bearing_gradient_groups_active": all(required_gradient_groups(sorted(gradients)).values()),
        "finite_scores": all(np.isfinite(row).all() for values in rows.values() for row in values),
        "deterministic": deterministic <= float(ev["deterministic_tolerance"]),
        "candidate_permutation": permutation_error <= float(ev["candidate_permutation_tolerance"]),
        "nohistory_exact": nohistory_exact,
        "nohistory_zero_carrier": zero_carrier <= float(ev["zero_carrier_tolerance"]),
        "nohistory_zero_list_message": zero_message <= float(ev["zero_carrier_tolerance"]),
        "cross_edge_order_active": edge_change["any_fraction"] >= float(ev["cross_edge_order_change_min"]),
        "cross_edge_top10_active": edge_change["top10_fraction"] >= float(ev["cross_edge_top10_change_min"]),
        "null_contrast_order_active": contrast_change["any_fraction"] >= float(ev["contrast_order_change_min"]),
        "null_contrast_top10_active": contrast_change["top10_fraction"] >= float(ev["contrast_top10_change_min"]),
        "wrong_order_active": wrong_change["any_fraction"] >= float(ev["wrong_order_change_min"]),
        "wrong_top10_active": wrong_change["top10_fraction"] >= float(ev["wrong_top10_change_min"]),
        "candidate_key_asserted": actual_hash == expected_hash,
        "A_labels_closed": True,
        "fresh_reserve_dev_test_qrels_closed": True,
    }
    report = {
        "candidate_id": "c54", "domain": domain, "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": execution_hash, "candidate_key_sha256": actual_hash,
        "parameters": model.parameter_count(), "epoch_loss_means": means,
        "active_gradient_parameters": sorted(gradients), "checks": checks,
        "edge_change": edge_change, "null_contrast_change": contrast_change,
        "wrong_change": wrong_change, "raw_change": raw_change,
        "true_wrong_correction_correlation": correlation(rows["correction"], rows["wrong_correction"]),
        "deterministic_max_abs": deterministic,
        "candidate_permutation_max_abs": permutation_error,
        "nohistory_carrier_max_abs": zero_carrier,
        "nohistory_list_message_max_abs": zero_message,
        "checkpoint": {"path": str(checkpoint.relative_to(REPO_ROOT)), "sha256": sha256_file(checkpoint)},
        "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)},
        "fit_labels_read": True, "A_labels_read": False,
        "fresh_reserve_dev_test_qrels_opened": False,
    }
    write_once(report_path, report); return report


def run_a0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, execution_hash = verify_execution(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    reports = {
        domain: [
            json.loads((root / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8"))
            for seed in config["training"][f"{domain}_seeds"]
        ]
        for domain in ("kuai", "amazon")
    }
    checks = {
        "all_seed_checks": all(all(row["checks"].values()) for values in reports.values() for row in values),
        "same_parameters_per_domain": all(len({row["parameters"] for row in values}) == 1 for values in reports.values()),
        "candidate_hashes_match": all(
            row["candidate_key_sha256"] == config["integrity"][f"{domain}_candidate_key_sha256"]
            for domain, values in reports.items() for row in values
        ),
        "artifacts_match": all(
            sha256_file(REPO_ROOT / row["score_artifact"]["path"]) == row["score_artifact"]["sha256"]
            and sha256_file(REPO_ROOT / row["checkpoint"]["path"]) == row["checkpoint"]["sha256"]
            for values in reports.values() for row in values
        ),
        "A_labels_closed": all(row["A_labels_read"] is False for values in reports.values() for row in values),
        "fresh_reserve_dev_test_qrels_closed": True,
    }
    passed = all(checks.values())
    value = {
        "candidate_id": "c54", "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_mechanics_authorize_fresh_design" if passed else "failed_mechanics_terminal",
        "decision": "freeze_fresh_matched_control_gate" if passed else "close_history_carrier_competition",
        "execution_lock_sha256": execution_hash, "checks": checks,
        "domains": {
            domain: [{
                "seed": row["seed"], "checks": row["checks"],
                "epoch_loss_means": row["epoch_loss_means"],
                "edge_change": row["edge_change"],
                "null_contrast_change": row["null_contrast_change"],
                "wrong_change": row["wrong_change"],
                "raw_change": row["raw_change"],
                "true_wrong_correction_correlation": row["true_wrong_correction_correlation"],
            } for row in values]
            for domain, values in reports.items()
        },
        "claims": {
            "mechanics_only": True, "utility_result": False,
            "novelty_established": False, "fresh_result": False,
        },
        "A_labels_read": False, "fresh_reserve_dev_test_qrels_opened": False,
    }
    write_once(root / "a0_report.json", value)
    write_once(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("d0", "seed", "a0"))
    parser.add_argument("--domain", choices=("kuai", "amazon"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(); config = load_config(args.config)
    if args.stage == "d0":
        value = run_d0(config, args.device)
    elif args.stage == "seed":
        if args.domain is None or args.seed is None:
            raise ValueError("C54 seed requires domain and seed")
        value = run_seed(config, args.domain, args.seed, args.device)
    else:
        value = run_a0(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
