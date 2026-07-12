from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(REPO_ROOT / "src"), str(C47_ROOT), str(C38_ROOT)):
    if value not in sys.path:
        # C69's own root is inserted first by each executable.  Dependencies
        # are search fallbacks: prepending C47 would shadow C69's top-level
        # ``model`` package when the formal seed entry point is used.
        sys.path.append(value)

from probe.run_signal_gate import AmazonStore, KuaiStore, candidate_key_sha256  # noqa: E402
from train.store import FrozenTransferStore  # noqa: E402


class DomainStore:
    def __init__(self, domain: str, c47: Mapping[str, Any], c38: Mapping[str, Any]) -> None:
        self.domain = domain
        self.selection = json.loads((REPO_ROOT / c47["paths"]["selection"]).read_text())
        if domain == "kuai":
            self.fit_store: Any = KuaiStore(c47)
            self.eval_store: Any = self.fit_store
            self.fit_role = "kuai_fit"
            self.a_role = "kuai_internal_A"
            self.expected_hash = c47["integrity"]["kuai_candidate_key_sha256"]
            self.input_dim = int(self.fit_store.item_embeddings.shape[1])
        elif domain == "amazon":
            self.fit_store = FrozenTransferStore(c38)
            self.eval_store = AmazonStore(c47)
            self.fit_role = "amazon_fit"
            self.a_role = "amazon_internal_A"
            self.expected_hash = c47["integrity"]["amazon_candidate_key_sha256"]
            self.input_dim = int(self.fit_store.item_embeddings.shape[1])
        else:
            raise ValueError(domain)

    def fit_indices(self) -> list[int]:
        return [int(v) for v in self.selection["roles"][self.fit_role]["indices"]]

    def a_indices(self) -> list[int]:
        return [int(v) for v in self.selection["roles"][self.a_role]["indices"]]

    def donors(self) -> list[int]:
        return [int(v) for v in self.selection["wrong_history_donors"][self.a_role]["indices"]]

    def fit_sequence(self, index: int) -> np.ndarray:
        if self.domain == "kuai":
            return self.fit_store.history(index)
        return self.fit_store.items(self.fit_store.history_positions(index, "true"))

    def query(self, index: int) -> np.ndarray:
        return self.eval_store.query(index)

    def candidates(self, index: int) -> np.ndarray:
        return self.eval_store.candidates(index)

    def history(self, index: int, donor: int | None = None) -> np.ndarray:
        if self.domain == "kuai":
            return self.eval_store.history(index if donor is None else donor)
        return self.eval_store.history(index, "true" if donor is None else "wrong")

    def request_id(self, index: int) -> str:
        return self.eval_store.request_id(index)

    def candidate_ids(self, index: int) -> list[str]:
        return self.eval_store.candidate_ids(index)

    def assert_candidate_hash(self) -> None:
        actual = candidate_key_sha256(self.eval_store, self.a_indices())
        if actual != self.expected_hash:
            raise RuntimeError(f"C69 {self.domain} candidate hash changed")


def materialize_pairs(store: DomainStore) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sequences = [np.asarray(store.fit_sequence(i), dtype=np.float16) for i in store.fit_indices()]
    request_rows, target_rows = [], []
    for request, sequence in enumerate(sequences):
        for target in range(1, len(sequence)):
            request_rows.append(request)
            target_rows.append(target)
    if len(request_rows) < 5000:
        raise RuntimeError(f"C69 {store.domain} insufficient transition pairs")
    boxed = np.empty(len(sequences), dtype=object)
    boxed[:] = sequences
    return boxed, np.asarray(request_rows, dtype=np.int32), np.asarray(target_rows, dtype=np.int16)


def batch_pairs(
    sequences: np.ndarray,
    request_rows: np.ndarray,
    target_rows: np.ndarray,
    sampled: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sources, targets, requests = [], [], []
    for example in sampled:
        request = int(request_rows[example])
        target = int(target_rows[example])
        sequence = sequences[request]
        sources.append(np.asarray(sequence[target - 1], dtype=np.float32))
        targets.append(np.asarray(sequence[target], dtype=np.float32))
        requests.append(request)
    return np.stack(sources), np.stack(targets), np.asarray(requests, dtype=np.int64)


def choose_negative_indices(
    sources: Tensor,
    targets: Tensor,
    requests: Tensor,
    *,
    mode: str,
    target_similarity_weight: float,
) -> tuple[Tensor, dict[str, float]]:
    count = len(sources)
    same_request = requests[:, None].eq(requests[None, :])
    eye = torch.eye(count, dtype=torch.bool, device=sources.device)
    forbidden = same_request | eye
    source_n = F.normalize(sources, dim=-1)
    target_n = F.normalize(targets, dim=-1)
    pair = source_n @ target_n.T
    positive_pair = pair.diag()
    if mode == "semantic_matched_negative":
        target_similarity = target_n @ target_n.T
        cost = (pair - positive_pair[:, None]).abs() + target_similarity_weight * (1.0 - target_similarity)
        cost = cost.masked_fill(forbidden, torch.inf)
        chosen = cost.argmin(dim=1)
    elif mode == "random_negative":
        chosen = torch.empty(count, dtype=torch.long, device=sources.device)
        for row in range(count):
            for shift in range(1, count + 1):
                candidate = (row + shift) % count
                if not bool(forbidden[row, candidate]):
                    chosen[row] = candidate
                    break
            else:
                raise RuntimeError("C69 batch has no cross-request negative")
        target_similarity = target_n @ target_n.T
    else:
        raise ValueError(mode)
    negative_pair = pair[torch.arange(count, device=sources.device), chosen]
    selected_target_similarity = target_similarity[torch.arange(count, device=sources.device), chosen]
    return chosen, {
        "pair_cosine_abs_gap": float((negative_pair - positive_pair).abs().mean().detach().cpu()),
        "target_cosine": float(selected_target_similarity.mean().detach().cpu()),
    }


def relation_matrix(model: Any, history: np.ndarray, candidates: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    history = np.asarray(history, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if not len(history):
        return np.zeros((0, len(candidates)), dtype=np.float32)
    h = torch.from_numpy(np.ascontiguousarray(history)).to(device)
    c = torch.from_numpy(np.ascontiguousarray(candidates)).to(device)
    hi = torch.arange(len(h), device=device).repeat_interleave(len(c))
    ci = torch.arange(len(c), device=device).repeat(len(h))
    values = []
    with torch.inference_mode():
        for start in range(0, len(hi), batch_size):
            stop = min(len(hi), start + batch_size)
            values.append(model.anchored_score(h[hi[start:stop]], c[ci[start:stop]]).cpu())
    return torch.cat(values).numpy().reshape(len(h), len(c)).astype(np.float32, copy=False)


def score_request(model: Any, query: np.ndarray, history: np.ndarray, candidates: np.ndarray, *, temperature: float, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(query, dtype=np.float32)
    history = np.asarray(history, dtype=np.float32)
    candidates = np.asarray(candidates, dtype=np.float32)
    if not len(history):
        zero = np.zeros(len(candidates), dtype=np.float32)
        return zero, zero.copy()
    q = query / max(float(np.linalg.norm(query)), 1e-8)
    h = history / np.maximum(np.linalg.norm(history, axis=1, keepdims=True), 1e-8)
    c = candidates / np.maximum(np.linalg.norm(candidates, axis=1, keepdims=True), 1e-8)
    logits = (h @ q) / temperature
    logits -= logits.max()
    weights = np.exp(logits)
    weights /= weights.sum()
    relation = relation_matrix(model, history, candidates, batch_size=batch_size, device=device)
    return (weights @ relation).astype(np.float32), (weights @ (h @ c.T)).astype(np.float32)


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, row in enumerate(rows):
        offsets[i + 1] = offsets[i] + len(row)
    return offsets, np.concatenate(rows).astype(np.float32, copy=False)


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [np.asarray(values[offsets[i] : offsets[i + 1]], dtype=np.float32).copy() for i in range(len(offsets) - 1)]
