"""Execute C52's locked exposed dual-domain formulation gate."""

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
from transformers import AutoModel, AutoTokenizer


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
for value in (str(C38_ROOT), str(REPO_ROOT / "src"), str(SYSTEM_ROOT)):
    if value not in sys.path:
        sys.path.insert(0, value)

from model.concept_attention import history_supported_concept_scores  # noqa: E402
from probe.freeze_lock import load_config, sha256_file, verify  # noqa: E402
from myrec.eval.metrics import ScoredCandidate, ndcg_at_k, sort_candidates  # noqa: E402
from train.gate_metrics import bootstrap, clicked_direction, compare  # noqa: E402
from train.store import FrozenTransferStore, open_role_labels  # noqa: E402


SCORE_NAMES = (
    "base",
    "primary",
    "linearized_token_krr",
    "token_softmax",
    "pooled_plain_krr",
    "pooled_softmax",
    "posterior",
    "wrong_primary",
    "correction",
    "wrong_correction",
)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def flatten(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    values = np.concatenate(rows).astype(np.float32, copy=False) if rows else np.empty(0, dtype=np.float32)
    return np.asarray(offsets, dtype=np.int64), values


def unflatten(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32).copy()
        for row in range(len(offsets) - 1)
    ]


class PackedTrain:
    def __init__(self, root: str | Path) -> None:
        root = Path(root)
        self.request_ids = [
            str(json.loads(line)["request_id"])
            for line in (root / "request_ids.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.candidate_offsets = np.load(root / "candidate_offsets.npy", mmap_mode="r")
        self.candidate_indices = np.load(root / "candidate_embedding_indices.npy", mmap_mode="r")
        self.candidate_item_ids = np.load(root / "candidate_item_ids.npy", mmap_mode="r")
        self.history_offsets = np.load(root / "history_offsets.npy", mmap_mode="r")
        self.history_indices = np.load(root / "history_embedding_indices.npy", mmap_mode="r")

    def candidates(self, index: int) -> np.ndarray:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return np.asarray(self.candidate_indices[start:stop], dtype=np.int64)

    def candidate_ids(self, index: int) -> list[str]:
        start, stop = int(self.candidate_offsets[index]), int(self.candidate_offsets[index + 1])
        return [str(value) for value in self.candidate_item_ids[start:stop].tolist()]

    def history(self, index: int) -> np.ndarray:
        start, stop = int(self.history_offsets[index]), int(self.history_offsets[index + 1])
        return np.asarray(self.history_indices[start:stop], dtype=np.int64)


class KuaiStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        paths = config["paths"]
        root = REPO_ROOT / paths["kuai_packed_root"]
        self.data = PackedTrain(root)
        self.query_indices = np.load(root / "query_indices.npy", mmap_mode="r")
        self.query_embeddings = np.load(REPO_ROOT / paths["kuai_query_embeddings"], mmap_mode="r")
        self.item_embeddings = np.load(REPO_ROOT / paths["kuai_item_embeddings"], mmap_mode="r")

    def query(self, index: int) -> np.ndarray:
        return np.asarray(self.query_embeddings[int(self.query_indices[index])], dtype=np.float32)

    def candidates(self, index: int) -> np.ndarray:
        return np.asarray(self.item_embeddings[self.data.candidates(index)], dtype=np.float32)

    def candidate_positions(self, index: int) -> np.ndarray:
        return self.data.candidates(index)

    def history(self, index: int) -> np.ndarray:
        return np.asarray(self.item_embeddings[self.data.history(index)], dtype=np.float32)

    def request_id(self, index: int) -> str:
        return str(self.data.request_ids[index])

    def candidate_ids(self, index: int) -> list[str]:
        return self.data.candidate_ids(index)


class AmazonStore:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.store = FrozenTransferStore(
            {
                "paths": {
                    "selection": str(REPO_ROOT / config["paths"]["amazon_adapter_selection"]),
                    "feature_root": str(REPO_ROOT / config["paths"]["amazon_feature_root"]),
                },
                "model": {"embedding_dim": int(config["encoding"]["hidden_dim"])},
            }
        )

    def query(self, index: int) -> np.ndarray:
        return self.store.query(index)

    def candidates(self, index: int) -> np.ndarray:
        return self.store.items(self.store.candidate_positions(index))

    def candidate_positions(self, index: int) -> np.ndarray:
        return self.store.candidate_positions(index)

    def history(self, index: int, source: str = "true") -> np.ndarray:
        return self.store.items(self.store.history_positions(index, source))

    def request_id(self, index: int) -> str:
        return self.store.request_id(index)

    def candidate_ids(self, index: int) -> list[str]:
        return self.store.candidate_ids(index)


def candidate_key_sha256(store: Any, indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(
            json.dumps([store.request_id(index), *store.candidate_ids(index)], separators=(",", ":")).encode()
        )
        digest.update(b"\n")
    return digest.hexdigest()


def role(config: Mapping[str, Any], domain: str) -> tuple[Any, list[int], list[int], str]:
    selection = json.loads((REPO_ROOT / config["paths"]["c47_selection"]).read_text(encoding="utf-8"))
    role_name = f"{domain}_internal_A"
    store: Any = KuaiStore(config) if domain == "kuai" else AmazonStore(config)
    indices = [int(value) for value in selection["roles"][role_name]["indices"]]
    donors = [int(value) for value in selection["wrong_history_donors"][role_name]["indices"]]
    expected = config["integrity"][f"{domain}_candidate_key_sha256"]
    if candidate_key_sha256(store, indices) != expected:
        raise RuntimeError(f"C52 {domain} candidate key differs")
    return store, indices, donors, expected


def load_c47_rows(config: Mapping[str, Any], domain: str) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / config["paths"][f"c47_{domain}_scores"]
    expected = config["integrity"][f"c47_{domain}_scores_sha256"]
    if sha256_file(path) != expected:
        raise RuntimeError(f"C52 C47 {domain} score source changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return {
            name: unflatten(offsets, source[name])
            for name in ("base", "plain_ridge", "softmax_attention", "posterior_supported")
        }


def assert_cuda(config: Mapping[str, Any], shard_id: int, device: str) -> None:
    expected = str(config["resources"]["physical_gpus"][shard_id])
    if device != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != expected:
        raise RuntimeError(f"C52 shard {shard_id} requires physical GPU {expected}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C52 requires exactly one visible CUDA GPU")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C52 deterministic CUBLAS workspace is absent")


def seed_all(seed: int = 20263610) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def kuai_item_texts(config: Mapping[str, Any], positions: Sequence[int]) -> dict[int, str]:
    targets = sorted(set(int(value) for value in positions))
    output: dict[int, str] = {}
    pointer = 0
    with (REPO_ROOT / config["paths"]["kuai_corpus"]).open("r", encoding="utf-8") as handle:
        for row, line in enumerate(handle):
            if pointer >= len(targets):
                break
            if row < targets[pointer]:
                continue
            if row != targets[pointer]:
                raise RuntimeError("C52 Kuai corpus row coverage differs")
            output[row] = str(json.loads(line).get("item_title", ""))
            pointer += 1
    if len(output) != len(targets):
        raise RuntimeError("C52 Kuai item text coverage differs")
    return output


def amazon_item_texts(config: Mapping[str, Any], positions: Sequence[int]) -> dict[int, str]:
    targets = set(int(value) for value in positions)
    output: dict[int, str] = {}
    path = REPO_ROOT / config["paths"]["amazon_feature_root"] / "items.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            position = int(row["position"])
            if position in targets:
                output[position] = str(row["text"])
    if output.keys() != targets:
        raise RuntimeError("C52 Amazon item text coverage differs")
    return output


def amazon_query_texts(config: Mapping[str, Any], indices: Sequence[int]) -> dict[int, str]:
    targets = set(int(value) for value in indices)
    output: dict[int, str] = {}
    path = REPO_ROOT / config["paths"]["amazon_feature_root"] / "requests.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            index = int(row["record_index"])
            if index in targets:
                output[index] = str(row["text"])
    if output.keys() != targets:
        raise RuntimeError("C52 Amazon query text coverage differs")
    return output


def encode_batch(
    model: Any,
    tokenizer: Any,
    *,
    texts: Sequence[str] | None,
    input_ids: np.ndarray | None,
    attention_mask: np.ndarray | None,
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> list[np.ndarray]:
    if (texts is None) == (input_ids is None):
        raise ValueError("C52 encoding requires exactly one input representation")
    if texts is not None:
        encoded = tokenizer(
            list(texts), padding="max_length", truncation=True, max_length=max_length,
            add_special_tokens=True, return_tensors="np",
        )
        ids = np.asarray(encoded["input_ids"], dtype=np.int64)
        mask = np.asarray(encoded["attention_mask"], dtype=bool)
    else:
        ids = np.asarray(input_ids[:, :max_length], dtype=np.int64)
        mask = np.asarray(attention_mask[:, :max_length], dtype=bool)
    special = np.asarray(sorted(set(int(value) for value in tokenizer.all_special_ids)), dtype=np.int64)
    content = mask & ~np.isin(ids, special)
    output: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(ids), batch_size):
            stop = min(len(ids), start + batch_size)
            result = model(
                input_ids=torch.from_numpy(ids[start:stop]).to(device),
                attention_mask=torch.from_numpy(mask[start:stop].astype(np.int64)).to(device),
            ).last_hidden_state.float().cpu().numpy()
            for row, values in enumerate(result):
                selected = values[content[start + row]]
                if not len(selected):
                    selected = values[:1]
                denominator = np.maximum(np.linalg.norm(selected, axis=-1, keepdims=True), 1e-6)
                output.append((selected / denominator).astype(np.float16))
    return output


def padded_candidates(rows: Sequence[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = rows[0].shape[-1]
    length = max(len(row) for row in rows)
    values = np.zeros((len(rows), length, width), dtype=np.float32)
    mask = np.zeros((len(rows), length), dtype=bool)
    for index, row in enumerate(rows):
        values[index, : len(row)] = np.asarray(row, dtype=np.float32)
        mask[index, : len(row)] = True
    return torch.from_numpy(values).to(device), torch.from_numpy(mask).to(device)


def settings(config: Mapping[str, Any]) -> dict[str, float]:
    row = config["operator"]
    return {
        "ridge": float(row["ridge"]),
        "candidate_token_temperature": float(row["candidate_token_temperature"]),
        "query_concept_temperature": float(row["query_concept_temperature"]),
        "history_softmax_temperature": float(row["history_softmax_temperature"]),
        "epsilon": float(row["normalization_epsilon"]),
    }


def score_shard(config: Mapping[str, Any], domain: str, shard_id: int, device_name: str) -> dict[str, Any]:
    _, lock_hash = verify(config)
    assert_cuda(config, shard_id, device_name)
    seed_all()
    device = torch.device(device_name)
    store, indices, donors, expected_hash = role(config, domain)
    positions = [row for row in range(len(indices)) if row % int(config["resources"]["num_shards"]) == shard_id]
    request_indices = [indices[row] for row in positions]
    candidate_positions = [
        int(value) for index in request_indices for value in store.candidate_positions(index).tolist()
    ]
    texts = kuai_item_texts(config, candidate_positions) if domain == "kuai" else amazon_item_texts(config, candidate_positions)
    snapshot = REPO_ROOT / config["paths"][f"{domain}_bge_snapshot"]
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True).to(device)
    ordered_item_positions = sorted(texts)
    item_states = encode_batch(
        model, tokenizer, texts=[texts[value] for value in ordered_item_positions],
        input_ids=None, attention_mask=None,
        max_length=int(config["encoding"]["candidate_max_length"]),
        batch_size=int(config["encoding"]["batch_size"]), device=device,
    )
    item_cache = dict(zip(ordered_item_positions, item_states))
    if domain == "kuai":
        root = REPO_ROOT / config["paths"]["kuai_query_tokens"]
        source_ids = np.load(root / "train_input_ids.npy", mmap_mode="r")
        source_mask = np.load(root / "train_attention_mask.npy", mmap_mode="r")
        query_states = encode_batch(
            model, tokenizer, texts=None,
            input_ids=np.asarray(source_ids[request_indices]),
            attention_mask=np.asarray(source_mask[request_indices]),
            max_length=int(config["encoding"]["query_max_length"]),
            batch_size=int(config["encoding"]["batch_size"]), device=device,
        )
    else:
        query_text = amazon_query_texts(config, request_indices)
        prefix = str(config["encoding"]["amazon_query_prefix"])
        query_states = encode_batch(
            model, tokenizer, texts=[prefix + query_text[index] for index in request_indices],
            input_ids=None, attention_mask=None,
            max_length=int(config["encoding"]["query_max_length"]),
            batch_size=int(config["encoding"]["batch_size"]), device=device,
        )
    del model
    torch.cuda.empty_cache()
    c47 = load_c47_rows(config, domain)
    rows: dict[str, list[np.ndarray]] = {name: [] for name in SCORE_NAMES}
    deterministic_max = 0.0
    candidate_permutation_max = 0.0
    history_permutation_max = 0.0
    nohistory_exact = True
    finite = True
    attention_change_fraction: list[float] = []
    operator_settings = settings(config)
    for local, (position, index) in enumerate(zip(positions, request_indices)):
        candidate_position = store.candidate_positions(index)
        candidate_rows = [item_cache[int(value)] for value in candidate_position]
        candidate_tensor, candidate_mask = padded_candidates(candidate_rows, device)
        query_tensor = torch.from_numpy(np.asarray(query_states[local], dtype=np.float32)).to(device)
        query_mask = torch.ones(len(query_tensor), dtype=torch.bool, device=device)
        true_history = store.history(index) if domain == "kuai" else store.history(index, "true")
        wrong_history = store.history(donors[position]) if domain == "kuai" else store.history(index, "wrong")
        true_tensor = torch.from_numpy(np.asarray(true_history, dtype=np.float32)).to(device)
        wrong_tensor = torch.from_numpy(np.asarray(wrong_history, dtype=np.float32)).to(device)
        with torch.inference_mode():
            primary = history_supported_concept_scores(
                query_tensor, query_mask, candidate_tensor, candidate_mask, true_tensor,
                **operator_settings,
            )
            repeated = history_supported_concept_scores(
                query_tensor, query_mask, candidate_tensor, candidate_mask, true_tensor,
                **operator_settings,
            )
            wrong = history_supported_concept_scores(
                query_tensor, query_mask, candidate_tensor, candidate_mask, wrong_tensor,
                **operator_settings,
            )
            reversed_candidate = history_supported_concept_scores(
                query_tensor, query_mask, candidate_tensor.flip(0), candidate_mask.flip(0), true_tensor,
                **operator_settings,
            )
            reversed_history = history_supported_concept_scores(
                query_tensor, query_mask, candidate_tensor, candidate_mask, true_tensor.flip(0),
                **operator_settings,
            )
            empty = history_supported_concept_scores(
                query_tensor, query_mask, candidate_tensor, candidate_mask,
                true_tensor.new_empty((0, true_tensor.shape[-1])), **operator_settings,
            )
        values = {
            "primary": primary.primary_correction.cpu().numpy(),
            "linearized": primary.linearized_correction.cpu().numpy(),
            "softmax": primary.softmax_correction.cpu().numpy(),
            "wrong": wrong.primary_correction.cpu().numpy(),
        }
        base = c47["base"][position]
        rows["base"].append(base)
        rows["primary"].append((base + values["primary"]).astype(np.float32))
        rows["linearized_token_krr"].append((base + values["linearized"]).astype(np.float32))
        rows["token_softmax"].append((base + values["softmax"]).astype(np.float32))
        rows["pooled_plain_krr"].append(c47["plain_ridge"][position])
        rows["pooled_softmax"].append(c47["softmax_attention"][position])
        rows["posterior"].append(c47["posterior_supported"][position])
        rows["wrong_primary"].append((base + values["wrong"]).astype(np.float32))
        rows["correction"].append(values["primary"].astype(np.float32))
        rows["wrong_correction"].append(values["wrong"].astype(np.float32))
        deterministic_max = max(
            deterministic_max,
            float((primary.primary_correction - repeated.primary_correction).abs().max().cpu()),
        )
        candidate_permutation_max = max(
            candidate_permutation_max,
            float((primary.primary_correction - reversed_candidate.primary_correction.flip(0)).abs().max().cpu()),
        )
        history_permutation_max = max(
            history_permutation_max,
            float((primary.primary_correction - reversed_history.primary_correction).abs().max().cpu()),
        )
        nohistory_exact = nohistory_exact and bool(
            torch.count_nonzero(empty.primary_correction) == 0
            and torch.equal(empty.base_concept_attention, empty.factual_concept_attention)
        )
        finite = finite and all(np.isfinite(value).all() for value in values.values())
        attention_change_fraction.append(
            float((primary.base_concept_attention - primary.factual_concept_attention).abs().gt(1e-7).any(dim=-1).float().mean().cpu())
        )
    root = REPO_ROOT / config["paths"]["artifact_root"]
    root.mkdir(parents=True, exist_ok=True)
    score_path = root / f"{domain}_shard_{shard_id}.npz"
    report_path = root / f"{domain}_shard_{shard_id}.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(score_path if score_path.exists() else report_path)
    offsets, _ = flatten(rows["base"])
    with score_path.open("wb") as handle:
        np.savez(
            handle, request_positions=np.asarray(positions, dtype=np.int64), offsets=offsets,
            **{name: flatten(value)[1] for name, value in rows.items()},
        )
    report = {
        "candidate_id": "c52", "domain": domain, "shard_id": shard_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "execution_lock_sha256": lock_hash, "candidate_key_sha256": expected_hash,
        "requests": len(positions), "unique_candidate_texts": len(item_cache),
        "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)},
        "checks": {
            "finite": finite,
            "deterministic": deterministic_max <= float(config["evaluation"]["deterministic_tolerance"]),
            "candidate_permutation": candidate_permutation_max <= float(config["evaluation"]["candidate_permutation_tolerance"]),
            "history_permutation": history_permutation_max <= float(config["evaluation"]["history_permutation_tolerance"]),
            "nohistory_exact": nohistory_exact,
            "fresh_reserve_closed": True,
            "dev_test_qrels_closed": True,
        },
        "diagnostics": {
            "deterministic_max_abs": deterministic_max,
            "candidate_permutation_max_abs": candidate_permutation_max,
            "history_permutation_max_abs": history_permutation_max,
            "mean_candidate_attention_change_fraction": float(np.mean(attention_change_fraction)),
        },
    }
    atomic_json(report_path, report)
    return report


def load_shard(path: Path) -> tuple[np.ndarray, dict[str, list[np.ndarray]]]:
    with np.load(path, allow_pickle=False) as source:
        positions = np.asarray(source["request_positions"], dtype=np.int64)
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return positions, {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def rankings(request_ids: Sequence[str], item_ids: Sequence[Sequence[str]], scores: Sequence[np.ndarray]) -> list[list[str]]:
    return [
        [row.item_id for row in sort_candidates(request_id, [ScoredCandidate(str(item), float(score)) for item, score in zip(items, values)])]
        for request_id, items, values in zip(request_ids, item_ids, scores)
    ]


def order_changes(request_ids: Sequence[str], item_ids: Sequence[Sequence[str]], first: Sequence[np.ndarray], second: Sequence[np.ndarray]) -> dict[str, Any]:
    a, b = rankings(request_ids, item_ids, first), rankings(request_ids, item_ids, second)
    any_count = sum(int(x != y) for x, y in zip(a, b))
    top_count = sum(int(set(x[:10]) != set(y[:10])) for x, y in zip(a, b))
    return {"requests": len(a), "any_count": any_count, "any_fraction": any_count / len(a), "top10_count": top_count, "top10_fraction": top_count / len(a)}


def merge_domain(config: Mapping[str, Any], domain: str) -> dict[str, Any]:
    _, lock_hash = verify(config)
    store, indices, _, expected_hash = role(config, domain)
    count = len(indices)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    combined: dict[str, list[np.ndarray | None]] = {name: [None] * count for name in SCORE_NAMES}
    shard_reports = []
    for shard in range(int(config["resources"]["num_shards"])):
        report_path = root / f"{domain}_shard_{shard}.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        shard_reports.append(report)
        score_path = REPO_ROOT / report["score_artifact"]["path"]
        if sha256_file(score_path) != report["score_artifact"]["sha256"]:
            raise RuntimeError("C52 shard artifact changed")
        positions, rows = load_shard(score_path)
        for local, position in enumerate(positions):
            for name in SCORE_NAMES:
                if combined[name][int(position)] is not None:
                    raise RuntimeError("C52 duplicate shard position")
                combined[name][int(position)] = rows[name][local]
    if any(value is None for rows in combined.values() for value in rows):
        raise RuntimeError("C52 shard coverage incomplete")
    typed = {name: [np.asarray(value, dtype=np.float32) for value in rows] for name, rows in combined.items()}
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    base_change = order_changes(request_ids, item_ids, typed["base"], typed["primary"])
    wrong_change = order_changes(request_ids, item_ids, typed["primary"], typed["wrong_primary"])
    active_fraction = float(np.mean([np.max(np.abs(row)) > 1e-7 for row in typed["correction"]]))
    evaluation = config["evaluation"]
    structural = {
        "all_shard_checks": all(all(report["checks"].values()) for report in shard_reports),
        "candidate_key": candidate_key_sha256(store, indices) == expected_hash,
        "request_coverage": count == (600 if domain == "kuai" else 300),
        "active_correction": active_fraction >= float(evaluation["active_correction_fraction_min"]),
        "base_order_activity": base_change["any_fraction"] >= float(evaluation["base_order_change_fraction_min"]),
        "base_top10_activity": base_change["top10_fraction"] >= float(evaluation["base_top10_change_fraction_min"]),
        "wrong_order_activity": wrong_change["any_fraction"] >= float(evaluation["wrong_order_change_fraction_min"]),
        "fresh_reserve_closed": True,
        "dev_test_qrels_closed": True,
    }
    score_path = root / f"{domain}_scores.npz"
    report_path = root / f"{domain}_score_report.json"
    if score_path.exists() or report_path.exists():
        raise FileExistsError(score_path if score_path.exists() else report_path)
    offsets, _ = flatten(typed["base"])
    with score_path.open("wb") as handle:
        np.savez(handle, offsets=offsets, **{name: flatten(rows)[1] for name, rows in typed.items()})
    report = {
        "candidate_id": "c52", "domain": domain,
        "created_at": datetime.now(timezone.utc).isoformat(), "execution_lock_sha256": lock_hash,
        "status": "passed" if all(structural.values()) else "failed_terminal",
        "requests": count, "candidate_key_sha256": expected_hash, "checks": structural,
        "diagnostics": {"active_correction_fraction": active_fraction, "base_changes": base_change, "wrong_changes": wrong_change},
        "score_artifact": {"path": str(score_path.relative_to(REPO_ROOT)), "sha256": sha256_file(score_path)},
        "shard_reports_sha256": {str(row["shard_id"]): sha256_file(root / f"{domain}_shard_{row['shard_id']}.json") for row in shard_reports},
        "labels_used": False,
    }
    atomic_json(report_path, report)
    return report


def run_a0(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    reports = {domain: json.loads((root / f"{domain}_score_report.json").read_text(encoding="utf-8")) for domain in ("kuai", "amazon")}
    checks = {
        "both_domains_passed": all(report["status"] == "passed" for report in reports.values()),
        "score_hashes": all(sha256_file(REPO_ROOT / report["score_artifact"]["path"]) == report["score_artifact"]["sha256"] for report in reports.values()),
        "same_lock": all(report["execution_lock_sha256"] == lock_hash for report in reports.values()),
        "no_c52_label_used": all(report["labels_used"] is False for report in reports.values()),
        "fresh_reserve_closed": True,
        "dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c52", "stage": "A0", "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed_terminal", "checks": checks,
        "execution_lock_sha256": lock_hash, "exposed_A_metric_authorized": all(checks.values()),
        "fresh_reserve_opened": False, "dev_test_qrels_opened": False,
    }
    atomic_json(root / "a0_report.json", value)
    return value


def load_scores(report: Mapping[str, Any]) -> dict[str, list[np.ndarray]]:
    path = REPO_ROOT / report["score_artifact"]["path"]
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C52 merged score artifact changed")
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        return {name: unflatten(offsets, source[name]) for name in SCORE_NAMES}


def kuai_labels(config: Mapping[str, Any], store: KuaiStore, indices: Sequence[int]) -> list[np.ndarray]:
    path = REPO_ROOT / config["paths"]["kuai_candidate_labels"]
    if sha256_file(path) != config["integrity"]["kuai_candidate_labels_sha256"]:
        raise RuntimeError("C52 Kuai labels changed")
    source = np.load(path, mmap_mode="r")
    return [
        np.asarray(source[int(store.data.candidate_offsets[index]) : int(store.data.candidate_offsets[index + 1])], dtype=np.float32).copy()
        for index in indices
    ]


def amazon_labels(config: Mapping[str, Any], store: AmazonStore, indices: Sequence[int]) -> list[np.ndarray]:
    compact = open_role_labels(
        records_train_path=REPO_ROOT / config["paths"]["amazon_records_train"],
        records_train_sha256=config["integrity"]["amazon_records_train_sha256"],
        selection_path=REPO_ROOT / config["paths"]["amazon_adapter_selection"],
        selection_sha256=sha256_file(REPO_ROOT / config["paths"]["amazon_adapter_selection"]),
        store=store.store, role="internal_A",
    )
    return [compact.row(index, len(store.candidate_ids(index))) for index in indices]


def ndcg_rows(request_ids: Sequence[str], item_ids: Sequence[Sequence[str]], scores: Sequence[np.ndarray], labels: Sequence[np.ndarray]) -> np.ndarray:
    output = []
    for request_id, items, values, target in zip(request_ids, item_ids, scores, labels):
        ranked = [row.item_id for row in sort_candidates(request_id, [ScoredCandidate(str(item), float(score)) for item, score in zip(items, values)])]
        positives = {str(item) for item, label in zip(items, target) if label > 0}
        output.append(ndcg_at_k(ranked, positives, 10))
    return np.asarray(output, dtype=np.float64)


def evaluate_domain(config: Mapping[str, Any], domain: str, report: Mapping[str, Any]) -> dict[str, Any]:
    store, indices, _, expected_hash = role(config, domain)
    if candidate_key_sha256(store, indices) != expected_hash:
        raise RuntimeError("C52 candidate key changed before metric")
    scores = load_scores(report)
    request_ids = [store.request_id(index) for index in indices]
    item_ids = [store.candidate_ids(index) for index in indices]
    labels = kuai_labels(config, store, indices) if domain == "kuai" else amazon_labels(config, store, indices)
    metric_names = ("base", "primary", "linearized_token_krr", "token_softmax", "pooled_plain_krr", "pooled_softmax", "posterior", "wrong_primary")
    ndcg = {name: ndcg_rows(request_ids, item_ids, scores[name], labels) for name in metric_names}
    controls = {name: ndcg[name] for name in metric_names if name not in {"primary"}}
    evaluation = config["evaluation"]
    comparisons = compare(
        request_ids, ndcg["primary"], controls,
        samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]),
        folds=int(evaluation["hash_folds"]),
    )
    clicked_true = clicked_direction(scores["correction"], labels)
    clicked_wrong = clicked_direction(scores["wrong_correction"], labels)
    direction = bootstrap(clicked_true, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 20)
    specificity = bootstrap(clicked_true - clicked_wrong, samples=int(evaluation["bootstrap_samples"]), seed=int(evaluation["bootstrap_seed"]) + 21)
    thresholds = {name: float(evaluation["primary_minus_each_control_min"]) for name in controls}
    thresholds["base"] = float(evaluation["primary_minus_base_min"])
    thresholds["wrong_primary"] = float(evaluation["true_minus_wrong_min"])
    checks: dict[str, bool] = {"candidate_hash": True}
    for name, minimum in thresholds.items():
        row = comparisons[name]
        checks[f"{name}_effect"] = row["mean"] >= minimum
        checks[f"{name}_ci"] = row["percentile_95_ci"][0] > 0
        checks[f"{name}_all_folds_positive"] = all(fold["mean_difference"] > 0 for fold in row["hash_folds"])
    checks["clicked_direction_ci"] = direction["percentile_95_ci"][0] > 0
    checks["clicked_specificity_ci"] = specificity["percentile_95_ci"][0] > 0
    return {
        "status": "passed" if all(checks.values()) else "failed", "requests": len(indices),
        "candidate_key_sha256": expected_hash, "checks": checks,
        "mean_ndcg10": {name: float(value.mean()) for name, value in ndcg.items()},
        "comparisons": comparisons, "clicked_direction": direction,
        "clicked_true_minus_wrong": specificity,
    }


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify(config)
    root = REPO_ROOT / config["paths"]["artifact_root"]
    a0_path = root / "a0_report.json"
    a0 = json.loads(a0_path.read_text(encoding="utf-8"))
    if a0.get("status") != "passed" or a0.get("exposed_A_metric_authorized") is not True:
        raise PermissionError("C52 A0 did not authorize exposed-A metric use")
    reports = {domain: json.loads((root / f"{domain}_score_report.json").read_text(encoding="utf-8")) for domain in ("kuai", "amazon")}
    domains = {domain: evaluate_domain(config, domain, reports[domain]) for domain in ("kuai", "amazon")}
    passed = all(row["status"] == "passed" for row in domains.values())
    value = {
        "candidate_id": "c52", "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed_exposed_formulation" if passed else "failed_formulation_terminal",
        "decision": "authorize_trainable_internal_attention_gate" if passed else "close_C52_before_training_or_fresh_reserve",
        "execution_lock_sha256": lock_hash, "A0_report_sha256": sha256_file(a0_path),
        "domains": domains,
        "claims": {"exposed_formulation_only": True, "fresh_result": False, "trained_result": False, "dev_test_result": False},
        "fresh_reserve_opened": False, "dev_test_records_labels_qrels_opened": False,
    }
    atomic_json(root / "formulation_gate_report.json", value)
    atomic_json(REPO_ROOT / config["paths"]["promoted_report"], value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=("score-shard", "merge", "a0", "aggregate"))
    parser.add_argument("--domain", choices=("kuai", "amazon"))
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    verify(config)
    if args.stage == "score-shard":
        if args.domain is None or args.shard_id is None:
            raise ValueError("C52 score-shard requires domain and shard")
        value = score_shard(config, args.domain, args.shard_id, args.device)
    elif args.stage == "merge":
        if args.domain is None:
            raise ValueError("C52 merge requires domain")
        value = merge_domain(config, args.domain)
    elif args.stage == "a0":
        value = run_a0(config)
    else:
        value = aggregate(config)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
