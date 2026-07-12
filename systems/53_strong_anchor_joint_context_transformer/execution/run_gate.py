"""Materialize, train, and evaluate the locked C53 foundation gate."""

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
from torch.nn import functional as F
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(C38_ROOT), str(REPO_ROOT / "src"), str(SYSTEM_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from execution.freeze_locks import (  # noqa: E402
    load_config, sha256_file, verify_execution, verify_proposal, write_once,
)
from model.joint_context import StrongAnchorJointContextTransformer  # noqa: E402
from myrec.analysis.finetuned_query_tower import build_model as build_d2_model, iter_query_batches  # noqa: E402
from myrec.analysis.supervised_diagnostics import PackedRequestData  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.freeze_locks import load_config as load_c38_config  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.store import CompactLabels, FrozenTransferStore, open_role_labels  # noqa: E402


SCORE_NAMES = ("base", "joint", "independent", "wrong", "correction", "wrong_correction")


def save_array(root: Path, name: str, value: np.ndarray) -> dict[str, Any]:
    path = root / name
    if path.exists():
        raise FileExistsError(path)
    np.save(path, value)
    return {"path": str(path.relative_to(REPO_ROOT)), "sha256": sha256_file(path), "shape": list(value.shape), "dtype": str(value.dtype)}


def zscore(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    scale = float(value.std())
    return np.zeros_like(value, dtype=np.float32) if scale <= 1e-8 else ((value - value.mean()) / scale).astype(np.float32)


def compact_rows(indices: Sequence[int], rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(indices, np.int64), np.asarray(offsets, np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def assert_materialization_cuda(config: Mapping[str, Any], device: str) -> None:
    expected = str(config["resources"]["materialization_physical_gpu"])
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError("C53 materialization GPU differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C53 deterministic CUBLAS workspace absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C53 requires one visible GPU")


def materialize(config: Mapping[str, Any], device_name: str) -> dict[str, Any]:
    _, proposal_hash = verify_proposal(config)
    assert_materialization_cuda(config, device_name)
    paths = config["paths"]
    root = REPO_ROOT / paths["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    selection = json.loads((REPO_ROOT / paths["c47_selection"]).read_text(encoding="utf-8"))
    d2_config = yaml.safe_load((REPO_ROOT / paths["d2_config"]).read_text(encoding="utf-8"))
    d2_final = yaml.safe_load((REPO_ROOT / paths["d2_final_config"]).read_text(encoding="utf-8"))
    data = PackedRequestData.load(d2_config["packed_data_dir"], "train")
    fit = [int(value) for value in selection["roles"]["kuai_fit"]["indices"]]
    a_indices = [int(value) for value in selection["roles"]["kuai_internal_A"]["indices"]]
    feature_indices = fit + a_indices
    token_root = REPO_ROOT / paths["kuai_query_tokens"]
    token_ids = np.load(token_root / "train_input_ids.npy", mmap_mode="r")
    token_mask = np.load(token_root / "train_attention_mask.npy", mmap_mode="r")
    model = build_d2_model(d2_config, device_name)
    checkpoint = torch.load(REPO_ROOT / paths["d2_checkpoint"], map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    popularity = np.load(REPO_ROOT / paths["kuai_popularity"], mmap_mode="r")
    alpha = float(d2_final["final_training"]["d2p_alpha"])
    base_rows: dict[int, np.ndarray] = {}
    query_rows: dict[int, np.ndarray] = {}
    with torch.inference_mode():
        for batch in iter_query_batches(data, np.asarray(feature_indices, np.int64), 128, 65536, 0, False):
            indices = batch["request_indices"]
            input_ids = torch.from_numpy(np.asarray(token_ids[indices], np.int64)).to(device_name)
            attention_mask = torch.from_numpy(np.asarray(token_mask[indices], np.int64)).to(device_name)
            candidate_indices = torch.from_numpy(batch["candidate_indices"]).to(device_name)
            encoded = model.encoder(input_ids=input_ids, attention_mask=attention_mask)
            query_state = F.normalize(
                encoded.last_hidden_state[:, 0, :].float(), dim=-1, eps=1e-6
            )
            candidate_state = F.normalize(
                model.item_adapter(model.item_embeddings[candidate_indices].float()),
                dim=-1, eps=1e-6,
            )
            lower, upper = model.logit_scale_bounds
            scale = model.logit_scale.exp().clamp(min=lower, max=upper)
            raw = (scale * torch.einsum("bd,bcd->bc", query_state, candidate_state)).cpu().numpy()
            query_state = query_state.cpu().numpy()
            for row, raw_index in enumerate(indices):
                index = int(raw_index); count = int(batch["candidate_mask"][row].sum())
                base_rows[index] = alpha * zscore(raw[row, :count]) + (1.0 - alpha) * zscore(popularity[batch["candidate_indices"][row, :count]])
                query_rows[index] = np.asarray(query_state[row], np.float32).copy()
        item_chunks = []
        for start in range(0, len(model.item_embeddings), 4096):
            item_chunks.append(
                F.normalize(
                    model.item_adapter(model.item_embeddings[start : start + 4096].float()),
                    dim=-1, eps=1e-6,
                ).cpu().numpy().astype(np.float32, copy=False)
            )
        item_states = np.concatenate(item_chunks)
    query_states = np.stack([query_rows[index] for index in feature_indices]).astype(np.float32)
    representation_checks = {
        "query_states_finite": bool(np.isfinite(query_states).all()),
        "item_states_finite": bool(np.isfinite(item_states).all()),
        "base_scores_finite": all(np.isfinite(row).all() for row in base_rows.values()),
        "query_states_unit_normalized": float(
            np.max(np.abs(np.linalg.norm(query_states, axis=-1) - 1.0))
        ) <= 1e-5,
        "item_states_unit_normalized": float(
            np.max(np.abs(np.linalg.norm(item_states, axis=-1) - 1.0))
        ) <= 1e-5,
    }
    if not all(representation_checks.values()):
        raise RuntimeError(f"C53 D2 representation contract failed: {representation_checks}")
    del model
    base_offsets = [0]
    for index in feature_indices:
        base_offsets.append(base_offsets[-1] + len(base_rows[index]))
    kuai_fit_labels = []
    source_labels = np.load(REPO_ROOT / paths["kuai_candidate_labels"], mmap_mode="r")
    for index in fit:
        start, stop = int(data.candidate_offsets[index]), int(data.candidate_offsets[index + 1])
        kuai_fit_labels.append(np.asarray(source_labels[start:stop], np.float32).copy())
    c38 = load_c38_config(REPO_ROOT / paths["c38_config"])
    amazon_store = FrozenTransferStore(c38)
    amazon_all = open_role_labels(
        records_train_path=REPO_ROOT / paths["amazon_records_train"],
        records_train_sha256=config["integrity"]["amazon_records_train_sha256"],
        selection_path=REPO_ROOT / paths["c38_selection"],
        selection_sha256=config["integrity"]["c38_selection_sha256"],
        store=amazon_store, role="fit",
    )
    amazon_fit = [int(value) for value in selection["roles"]["amazon_fit"]["indices"] if not amazon_store.has_repeat(int(value))]
    amazon_fit_labels = [amazon_all.row(index, amazon_store.candidate_count(index)) for index in amazon_fit]
    kfi, kfo, kfv = compact_rows(fit, kuai_fit_labels)
    afi, afo, afv = compact_rows(amazon_fit, amazon_fit_labels)
    outputs = {
        "kuai_feature_indices": save_array(root, "kuai_feature_indices.npy", np.asarray(feature_indices, np.int64)),
        "kuai_query_states": save_array(root, "kuai_query_states.npy", query_states),
        "kuai_item_states": save_array(root, "kuai_item_states.npy", item_states),
        "kuai_base_offsets": save_array(root, "kuai_base_offsets.npy", np.asarray(base_offsets, np.int64)),
        "kuai_base_scores": save_array(root, "kuai_base_scores.npy", np.concatenate([base_rows[index] for index in feature_indices]).astype(np.float32)),
        "kuai_fit_indices": save_array(root, "kuai_fit_indices.npy", kfi),
        "kuai_fit_label_offsets": save_array(root, "kuai_fit_label_offsets.npy", kfo),
        "kuai_fit_labels": save_array(root, "kuai_fit_labels.npy", kfv),
        "amazon_fit_indices": save_array(root, "amazon_fit_indices.npy", afi),
        "amazon_fit_label_offsets": save_array(root, "amazon_fit_label_offsets.npy", afo),
        "amazon_fit_labels": save_array(root, "amazon_fit_labels.npy", afv),
    }
    report = {
        "candidate_id": "c53", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed", "proposal_lock_sha256": proposal_hash,
        "kuai_fit_requests": len(fit), "kuai_A_requests": len(a_indices),
        "amazon_fit_requests": len(amazon_fit),
        "amazon_repeat_fit_excluded": len(selection["roles"]["amazon_fit"]["indices"]) - len(amazon_fit),
        "kuai_representation_checks": representation_checks,
        "outputs": outputs, "fit_labels_read": True, "A_labels_read": False,
        "fresh_reserve_dev_test_qrels_opened": False,
    }
    write_once(REPO_ROOT / paths["materialization_report"], report)
    return report


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
            raise ValueError("C53 compact label count differs")
        return np.asarray(self.values[start:stop], np.float32).copy()


class DomainData:
    def __init__(self, domain: str, config: Mapping[str, Any]) -> None:
        self.domain = domain; self.config = config; paths = config["paths"]
        self.selection = json.loads((REPO_ROOT / paths["c47_selection"]).read_text(encoding="utf-8"))
        self.root = REPO_ROOT / paths["artifact_root"]
        self.labels = LocalCompactLabels(self.root, domain)
        if domain == "kuai":
            # PackedRequestData.load appends the split name.  The frozen C53
            # path names the split directory itself, so load from its parent.
            self.data = PackedRequestData.load(
                (REPO_ROOT / paths["kuai_packed_root"]).parent, "train"
            )
            feature = np.load(self.root / "kuai_feature_indices.npy", mmap_mode="r")
            self.feature_position = {
                int(index): position for position, index in enumerate(feature)
            }
            self.item_embeddings = np.load(self.root / "kuai_item_states.npy", mmap_mode="r")
            self.query_embeddings = np.load(self.root / "kuai_query_states.npy", mmap_mode="r")
            offsets = np.load(self.root / "kuai_base_offsets.npy", mmap_mode="r")
            scores = np.load(self.root / "kuai_base_scores.npy", mmap_mode="r")
            self.base_rows = {int(index): np.asarray(scores[int(offsets[row]) : int(offsets[row + 1])], np.float32).copy() for row, index in enumerate(feature)}
            self.fit_store = self.eval_store = None
            self.fit_indices = [int(value) for value in self.labels.indices]
            self.a_indices = [int(value) for value in self.selection["roles"]["kuai_internal_A"]["indices"]]
            self.donor = dict(zip(self.a_indices, [int(value) for value in self.selection["wrong_history_donors"]["kuai_internal_A"]["indices"]]))
            self.input_dim = int(self.item_embeddings.shape[1])
        elif domain == "amazon":
            c38 = load_c38_config(REPO_ROOT / paths["c38_config"])
            self.fit_store = FrozenTransferStore(c38)
            self.eval_store = FrozenTransferStore({"paths": {"selection": str(REPO_ROOT / paths["c47_amazon_adapter_selection"]), "feature_root": str(REPO_ROOT / paths["c47_amazon_feature_root"])}, "model": {"embedding_dim": 384}})
            self.fit_indices = [int(value) for value in self.labels.indices]
            self.a_indices = [int(value) for value in self.selection["roles"]["amazon_internal_A"]["indices"]]
            self.input_dim = 384
        else:
            raise ValueError("unknown C53 domain")

    def _store(self, index: int) -> Any:
        return self.fit_store if self.domain == "amazon" and index in self.fit_store.feature_position else self.eval_store

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
            return np.asarray(self.query_embeddings[self.feature_position[int(index)]], np.float32)
        return self._store(index).query(index)

    def items(self, positions: np.ndarray, index: int) -> np.ndarray:
        if self.domain == "kuai":
            return np.asarray(self.item_embeddings[positions], np.float32)
        return self._store(index).items(positions)

    def base(self, index: int) -> np.ndarray:
        return self.base_rows[index].copy() if self.domain == "kuai" else self._store(index).base_row(index)

    def fit_label(self, index: int) -> np.ndarray:
        return self.labels.row(index, len(self.candidate_positions(index)))

    def sizes(self, index: int, source: str = "true") -> tuple[int, int]:
        return len(self.history_positions(index, source)), len(self.candidate_positions(index))


def candidate_hash(data: DomainData, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(json.dumps([data.request_id(index), *data.candidate_ids(index)], separators=(",", ":")).encode()); digest.update(b"\n")
    return digest.hexdigest()


def batches(data: DomainData, indices: Sequence[int], config: Mapping[str, Any], *, seed: int, shuffle: bool) -> list[list[int]]:
    values = list(int(value) for value in indices)
    if shuffle:
        np.random.default_rng(seed).shuffle(values)
    maximum_requests = int(config["training"]["max_requests_per_batch"])
    maximum_tokens = int(config["training"]["max_sequence_tokens_per_batch"])
    output: list[list[int]] = []; current: list[int] = []; max_length = 0
    for index in values:
        h, c = data.sizes(index); length = 1 + h + c
        next_max = max(max_length, length); next_count = len(current) + 1
        if current and (next_count > maximum_requests or next_count * next_max > maximum_tokens):
            output.append(current); current = []; max_length = 0
        current.append(index); max_length = max(max_length, length)
    if current:
        output.append(current)
    return output


def collate(data: DomainData, indices: Sequence[int], *, source: str, labels: bool, device: torch.device) -> dict[str, torch.Tensor]:
    history_rows = [data.history_positions(index, source) for index in indices]
    candidate_rows = [data.candidate_positions(index) for index in indices]
    max_h = max(1, max(len(row) for row in history_rows)); max_c = max(len(row) for row in candidate_rows)
    query = np.stack([data.query(index) for index in indices]).astype(np.float32)
    history = np.zeros((len(indices), max_h, data.input_dim), np.float32); history_mask = np.zeros((len(indices), max_h), bool)
    candidates = np.zeros((len(indices), max_c, data.input_dim), np.float32); candidate_mask = np.zeros((len(indices), max_c), bool)
    base = np.zeros((len(indices), max_c), np.float32); target = np.zeros((len(indices), max_c), np.float32)
    for row, index in enumerate(indices):
        hp, cp = history_rows[row], candidate_rows[row]
        if len(hp): history[row, : len(hp)] = data.items(hp, index); history_mask[row, : len(hp)] = True
        candidates[row, : len(cp)] = data.items(cp, index); candidate_mask[row, : len(cp)] = True; base[row, : len(cp)] = data.base(index)
        if labels: target[row, : len(cp)] = data.fit_label(index)
    result = {"query": torch.from_numpy(query).to(device), "history": torch.from_numpy(history).to(device), "history_mask": torch.from_numpy(history_mask).to(device), "candidates": torch.from_numpy(candidates).to(device), "candidate_mask": torch.from_numpy(candidate_mask).to(device), "base_scores": torch.from_numpy(base).to(device)}
    if labels: result["labels"] = torch.from_numpy(target).to(device)
    return result


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positive = labels.gt(0) & mask; negative = -torch.finfo(scores.dtype).max
    if not bool(positive.any(dim=-1).all()): raise ValueError("C53 training row lacks positive")
    return (torch.logsumexp(scores.masked_fill(~mask, negative), dim=-1) - torch.logsumexp(scores.masked_fill(~positive, negative), dim=-1)).mean()


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.use_deterministic_algorithms(True)


def make_model(config: Mapping[str, Any], input_dim: int) -> StrongAnchorJointContextTransformer:
    row = config["model"]
    return StrongAnchorJointContextTransformer(input_dim=input_dim, hidden_dim=int(row["hidden_dim"]), heads=int(row["heads"]), layers=int(row["layers"]), ffn_dim=int(row["ffn_dim"]), dropout=float(row["dropout"]), max_history=int(row["max_history"]))


def rankings(request_id: str, item_ids: Sequence[str], values: np.ndarray) -> list[str]:
    return [row.item_id for row in sort_candidates(request_id, [ScoredCandidate(str(item), float(score)) for item, score in zip(item_ids, values)])]


def change(data: DomainData, indices: Sequence[int], first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> dict[str, Any]:
    a = [rankings(data.request_id(index), data.candidate_ids(index), row) for index, row in zip(indices, first)]
    b = [rankings(data.request_id(index), data.candidate_ids(index), row) for index, row in zip(indices, second)]
    any_count = sum(int(x != y) for x, y in zip(a, b)); top = sum(int(set(x[:10]) != set(y[:10])) for x, y in zip(a, b))
    return {"requests": len(a), "any_count": any_count, "any_fraction": any_count / len(a), "top10_count": top, "top10_fraction": top / len(a)}


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows: offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[int(offsets[i]) : int(offsets[i + 1])], np.float32).copy() for i in range(len(offsets) - 1)]


def run_seed(config: Mapping[str, Any], domain: str, seed: int, device_name: str) -> dict[str, Any]:
    _, lock_hash = verify_execution(config)
    physical = int(config["resources"][f"{domain}_seed_to_physical_gpu"][str(seed)])
    if device_name != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical): raise RuntimeError("C53 seed GPU differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}: raise RuntimeError("C53 CUBLAS workspace absent")
    seed_all(seed); device = torch.device(device_name); data = DomainData(domain, config)
    expected_candidate_hash = config["integrity"][f"{domain}_candidate_key_sha256"]
    actual_candidate_hash = candidate_hash(data, data.a_indices)
    if actual_candidate_hash != expected_candidate_hash:
        raise RuntimeError(f"C53 {domain} candidate key differs before scoring")
    model = make_model(config, data.input_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))
    losses = []; gradients: set[str] = set(); model.train()
    for epoch in range(int(config["training"]["epochs"])):
        for request_batch in batches(data, data.fit_indices, config, seed=seed + epoch, shuffle=True):
            batch = collate(data, request_batch, source="true", labels=True, device=device); optimizer.zero_grad(set_to_none=True)
            output = model(**{name: value for name, value in batch.items() if name != "labels"}); loss = listwise_loss(output.scores, batch["labels"], batch["candidate_mask"])
            if not bool(torch.isfinite(loss)): raise RuntimeError("C53 nonfinite loss")
            loss.backward()
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and bool(parameter.grad.ne(0).any()): gradients.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"])); optimizer.step(); losses.append(float(loss.detach().cpu()))
    model.eval(); root = REPO_ROOT / config["paths"]["artifact_root"]; checkpoint_root = REPO_ROOT / config["paths"]["checkpoint_root"]; checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_root / f"{domain}_seed_{seed}.pt"; score_path = root / f"{domain}_seed_{seed}_scores.npz"; report_path = root / f"{domain}_seed_{seed}_report.json"
    if checkpoint.exists() or score_path.exists() or report_path.exists(): raise FileExistsError(report_path)
    torch.save({"candidate_id": "c53", "domain": domain, "seed": seed, "state_dict": model.state_dict(), "execution_lock_sha256": lock_hash}, checkpoint)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}; deterministic = 0.0; nohistory_exact = True
    with torch.inference_mode():
        for request_batch in batches(data, data.a_indices, config, seed=0, shuffle=False):
            true = collate(data, request_batch, source="true", labels=False, device=device); wrong = collate(data, request_batch, source="wrong", labels=False, device=device)
            joint = model(**true); repeated = model(**true); independent = model(**true, independent_candidates=True); wrong_out = model(**wrong)
            empty = dict(true); empty["history_mask"] = torch.zeros_like(true["history_mask"]); nohistory = model(**empty)
            deterministic = max(deterministic, float((joint.scores - repeated.scores).abs().max().cpu()))
            nohistory_exact = nohistory_exact and bool(torch.equal(nohistory.scores, true["base_scores"].masked_fill(~true["candidate_mask"], 0.0)))
            mask = true["candidate_mask"].cpu().numpy()
            arrays = {"base": true["base_scores"].cpu().numpy(), "joint": joint.scores.cpu().numpy(), "independent": independent.scores.cpu().numpy(), "wrong": wrong_out.scores.cpu().numpy(), "correction": joint.correction.cpu().numpy(), "wrong_correction": wrong_out.correction.cpu().numpy()}
            for row in range(len(request_batch)):
                count = int(mask[row].sum())
                for name in SCORE_NAMES: rows[name].append(np.asarray(arrays[name][row, :count], np.float32).copy())
    # Exact caller-order audit on the first eight requests, one at a time.
    permutation_error = 0.0
    with torch.inference_mode():
        for index in data.a_indices[:8]:
            batch = collate(data, [index], source="true", labels=False, device=device); original = model(**batch).scores
            count = int(batch["candidate_mask"].sum()); permutation = torch.arange(count - 1, -1, -1, device=device); changed = dict(batch)
            for name in ("candidates", "candidate_mask", "base_scores"): changed[name] = changed[name][:, permutation]
            permuted = model(**changed).scores[:, torch.argsort(permutation)]
            permutation_error = max(permutation_error, float((original - permuted).abs().max().cpu()))
    edge_change = change(data, data.a_indices, rows["joint"], rows["independent"]); wrong_change = change(data, data.a_indices, rows["joint"], rows["wrong"])
    offsets, _ = flatten(rows["base"])
    with score_path.open("wb") as handle: np.savez(handle, offsets=offsets, **{name: flatten(value)[1] for name, value in rows.items()})
    ev = config["evaluation"]
    checks = {
        "finite_training": bool(losses) and bool(np.isfinite(losses).all()), "loss_decreased": float(np.mean(losses[-50:])) < float(np.mean(losses[:50])),
        "gradients_active": bool(gradients), "finite_scores": all(np.isfinite(row).all() for values in rows.values() for row in values),
        "deterministic": deterministic <= float(ev["deterministic_tolerance"]), "candidate_permutation": permutation_error <= float(ev["candidate_permutation_tolerance"]),
        "nohistory_exact": nohistory_exact, "cross_edge_order_active": edge_change["any_fraction"] >= float(ev["cross_edge_order_change_min"]),
        "cross_edge_top10_active": edge_change["top10_fraction"] >= float(ev["cross_edge_top10_change_min"]), "wrong_order_active": wrong_change["any_fraction"] >= float(ev["wrong_order_change_min"]),
        "wrong_top10_active": wrong_change["top10_fraction"] >= float(ev["wrong_top10_change_min"]),
        "candidate_key_asserted": actual_candidate_hash == expected_candidate_hash,
        "A_labels_closed": True, "fresh_reserve_dev_test_qrels_closed": True,
    }
    report = {"candidate_id": "c53", "domain": domain, "seed": seed, "created_at": datetime.now(timezone.utc).isoformat(), "execution_lock_sha256": lock_hash, "candidate_key_sha256": actual_candidate_hash, "parameters": model.parameter_count(), "training": {"steps": len(losses), "loss_first_50": float(np.mean(losses[:50])), "loss_last_50": float(np.mean(losses[-50:])), "active_gradient_parameters": sorted(gradients)}, "checks": checks, "edge_change": edge_change, "wrong_change": wrong_change, "deterministic_max_abs": deterministic, "candidate_permutation_max_abs": permutation_error, "checkpoint": {"path": str(checkpoint.relative_to(REPO_ROOT)), "sha256": sha256_file(checkpoint)}, "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)}, "A_labels_read": False, "fresh_reserve_dev_test_qrels_opened": False}
    write_once(report_path, report); return report


def load_rows(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]: raise RuntimeError("C53 score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], np.int64); return {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def run_a0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_execution(config); root = REPO_ROOT / config["paths"]["artifact_root"]
    reports = {domain: [json.loads((root / f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in config["training"][f"{domain}_seeds"]] for domain in ("kuai", "amazon")}
    checks = {"all_seed_checks": all(all(row["checks"].values()) for values in reports.values() for row in values), "same_parameters_per_domain": all(len({row["parameters"] for row in values}) == 1 for values in reports.values()), "candidate_hashes_match": all(row["candidate_key_sha256"] == config["integrity"][f"{domain}_candidate_key_sha256"] for domain, values in reports.items() for row in values), "artifacts_match": all(sha256_file(REPO_ROOT / row["score_artifact"]["path"]) == row["score_artifact"]["sha256"] and sha256_file(REPO_ROOT / row["checkpoint"]["path"]) == row["checkpoint"]["sha256"] for values in reports.values() for row in values), "A_labels_closed": all(row["A_labels_read"] is False for values in reports.values() for row in values), "fresh_reserve_dev_test_qrels_closed": True}
    value = {"candidate_id": "c53", "stage": "A0", "created_at": datetime.now(timezone.utc).isoformat(), "status": "passed" if all(checks.values()) else "failed_A0_terminal", "execution_lock_sha256": lock_hash, "checks": checks, "A_labels_read": False, "A_labels_authorized_after_A0": all(checks.values()), "fresh_reserve_dev_test_qrels_opened": False}
    write_once(root / "a0_report.json", value); return value


def average(groups: Sequence[Sequence[np.ndarray]]) -> list[np.ndarray]:
    return [np.mean(np.stack(values), axis=0).astype(np.float32) for values in zip(*groups)]


def a_labels(data: DomainData, config: Mapping[str, Any]) -> list[np.ndarray]:
    if data.domain == "kuai":
        source = np.load(REPO_ROOT / config["paths"]["kuai_candidate_labels"], mmap_mode="r")
        return [np.asarray(source[int(data.data.candidate_offsets[index]) : int(data.data.candidate_offsets[index + 1])], np.float32).copy() for index in data.a_indices]
    compact = open_role_labels(records_train_path=REPO_ROOT / config["paths"]["amazon_records_train"], records_train_sha256=config["integrity"]["amazon_records_train_sha256"], selection_path=REPO_ROOT / config["paths"]["c47_amazon_adapter_selection"], selection_sha256=config["integrity"]["c47_amazon_adapter_selection_sha256"], store=data.eval_store, role="internal_A")
    return [compact.row(index, len(data.candidate_positions(index))) for index in data.a_indices]


def ndcg_rows(data: DomainData, rows: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> np.ndarray:
    output=[]
    for index, scores, target in zip(data.a_indices, rows, labels):
        items=data.candidate_ids(index); positive={item for item,label in zip(items,target) if label>0}; output.append(ndcg_at_k(rankings(data.request_id(index),items,scores),positive,10))
    return np.asarray(output,np.float64)


def aggregate_domain(config: Mapping[str, Any], domain: str) -> dict[str, Any]:
    data=DomainData(domain,config); root=REPO_ROOT/config["paths"]["artifact_root"]; seeds=[int(value) for value in config["training"][f"{domain}_seeds"]]
    expected_candidate_hash=config["integrity"][f"{domain}_candidate_key_sha256"]; actual_candidate_hash=candidate_hash(data,data.a_indices)
    if actual_candidate_hash != expected_candidate_hash: raise RuntimeError(f"C53 {domain} candidate key differs before evaluation")
    reports=[json.loads((root/f"{domain}_seed_{seed}_report.json").read_text(encoding="utf-8")) for seed in seeds]; seed_rows={seed:load_rows(report) for seed,report in zip(seeds,reports)}; ensemble={name:average([seed_rows[seed][name] for seed in seeds]) for name in SCORE_NAMES}; labels=a_labels(data,config)
    ndcg={name:ndcg_rows(data,rows,labels) for name,rows in ensemble.items() if name not in {"correction","wrong_correction"}}; request_ids=[data.request_id(index) for index in data.a_indices]; ev=config["evaluation"]
    comparisons=compare(request_ids,ndcg["joint"],{"base":ndcg["base"],"independent":ndcg["independent"],"wrong":ndcg["wrong"]},samples=int(ev["bootstrap_samples"]),seed=int(ev["bootstrap_seed"]),folds=int(ev["hash_folds"]))
    seed_ndcg={seed:{name:ndcg_rows(data,seed_rows[seed][name],labels) for name in ("base","joint","independent","wrong")} for seed in seeds}; seed_diff={name:{str(seed):float((seed_ndcg[seed]["joint"]-seed_ndcg[seed][name]).mean()) for seed in seeds} for name in ("base","independent","wrong")}
    clicked_true=clicked_direction(ensemble["correction"],labels); clicked_wrong=clicked_direction(ensemble["wrong_correction"],labels); direction=bootstrap(clicked_true,samples=int(ev["bootstrap_samples"]),seed=int(ev["bootstrap_seed"])+20); specificity=bootstrap(clicked_true-clicked_wrong,samples=int(ev["bootstrap_samples"]),seed=int(ev["bootstrap_seed"])+21)
    thresholds={"base":float(ev["primary_minus_base_min"]),"independent":float(ev["primary_minus_independent_min"]),"wrong":float(ev["true_minus_wrong_min"])}; checks={}
    for name,minimum in thresholds.items():
        row=comparisons[name]; checks[f"{name}_effect"]=row["mean"]>=minimum; checks[f"{name}_ci"]=row["percentile_95_ci"][0]>0; checks[f"{name}_all_seed_fold_positive"]=all(value>0 for value in seed_diff[name].values()) and all(fold["mean_difference"]>0 for fold in row["hash_folds"])
    checks["clicked_direction_ci"]=direction["percentile_95_ci"][0]>0; checks["clicked_specificity_ci"]=specificity["percentile_95_ci"][0]>0
    return {"status":"passed" if all(checks.values()) else "failed","requests":len(data.a_indices),"candidate_key_sha256":actual_candidate_hash,"checks":checks,"mean_ndcg10":{name:float(value.mean()) for name,value in ndcg.items()},"comparisons":comparisons,"seed_differences":seed_diff,"clicked_direction":direction,"clicked_true_minus_wrong":specificity}


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _,lock_hash=verify_execution(config); root=REPO_ROOT/config["paths"]["artifact_root"]; a0=json.loads((root/"a0_report.json").read_text(encoding="utf-8"))
    if a0.get("status")!="passed" or a0.get("A_labels_authorized_after_A0") is not True: raise PermissionError("C53 A0 did not authorize A labels")
    domains={domain:aggregate_domain(config,domain) for domain in ("kuai","amazon")}; passed=all(row["status"]=="passed" for row in domains.values())
    value={"candidate_id":"c53","gate_id":config["gate_id"],"created_at":datetime.now(timezone.utc).isoformat(),"status":"passed_foundation_only" if passed else "failed_foundation_terminal","decision":"authorize_separate_innovation_design" if passed else "close_joint_context_as_immediate_foundation","execution_lock_sha256":lock_hash,"A0_report_sha256":sha256_file(root/"a0_report.json"),"domains":domains,"claims":{"known_foundation_only":True,"architecture_innovation":False,"exposed_train_internal":True,"fresh_result":False,"dev_test_result":False},"fresh_reserve_dev_test_qrels_opened":False}
    write_once(root/"foundation_gate_report.json",value); write_once(REPO_ROOT/config["paths"]["promoted_report"],value); return value


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); parser.add_argument("--stage",required=True,choices=("materialize","seed","a0","aggregate")); parser.add_argument("--domain",choices=("kuai","amazon")); parser.add_argument("--seed",type=int); parser.add_argument("--device",default="cuda:0"); args=parser.parse_args(); config=load_config(args.config)
    if args.stage=="materialize": value=materialize(config,args.device)
    elif args.stage=="seed":
        if args.domain is None or args.seed is None: raise ValueError("C53 seed requires domain/seed")
        value=run_seed(config,args.domain,args.seed,args.device)
    elif args.stage=="a0": value=run_a0(config)
    else: value=aggregate(config)
    print(json.dumps(value,indent=2,sort_keys=True))


if __name__=="__main__": main()
