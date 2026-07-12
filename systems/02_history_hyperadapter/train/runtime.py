"""Shared, label-safe runtime helpers for the frozen C02 screen."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.nn import functional as F

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
SRC_ROOT = REPO_ROOT / "src"
for entry in (SYSTEM_ROOT, SRC_ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from model.chht import CHHTRanker, masked_zscore


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config.get("candidate_id") != "c02":
        raise ValueError("configuration is not C02")
    if int(config.get("seed", -1)) != 20260708:
        raise ValueError("C02 seed must remain 20260708")
    if int(config.get("physical_gpu", -1)) != 1:
        raise ValueError("C02 physical GPU must remain 1")
    if config.get("environment") != "myrec-c02":
        raise ValueError("C02 environment must remain myrec-c02")
    if not str(config.get("run_id", "")).startswith("20260710_kuaisearch_c02_"):
        raise ValueError("invalid C02 run prefix")
    if int(config["training"]["max_implementation_attempts"]) > 2:
        raise ValueError("implementation-attempt budget exceeds prompt")
    if float(config["training"]["max_gpu_hours"]) > 8.0:
        raise ValueError("GPU budget exceeds prompt")


def validate_gpu(device: str) -> None:
    if device != "cuda:0":
        raise ValueError("C02 program device must be cuda:0")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != "1":
        raise RuntimeError(f"C02 requires CUDA_VISIBLE_DEVICES=1, got {visible!r}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C02 must see exactly one CUDA GPU")
    name = torch.cuda.get_device_name(0)
    if "A40" not in name:
        raise RuntimeError(f"C02 expected an NVIDIA A40, got {name}")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def assert_candidate_hash(config: dict[str, Any]) -> str:
    path = Path(config["paths"]["candidate_manifest"])
    actual = sha256_file(path)
    expected = str(config["integrity"]["candidate_manifest_sha256"])
    if actual != expected:
        raise ValueError(f"candidate manifest hash mismatch: {actual} != {expected}")
    return actual


def assert_proposal_lock(config: dict[str, Any]) -> dict[str, Any]:
    lock_path = Path(config["paths"]["candidate_source_root"]) / "notes/proposal_lock.json"
    lock = read_json(lock_path)
    if lock.get("status") != "locked_before_c02_gpu_outcome":
        raise ValueError("proposal is not frozen before C02 outcome")
    if lock.get("candidate_manifest_sha256") != config["integrity"]["candidate_manifest_sha256"]:
        raise ValueError("proposal/candidate manifest hash mismatch")
    if int(lock.get("seed", -1)) != int(config["seed"]):
        raise ValueError("proposal/config seed mismatch")
    for relative, expected in lock["file_sha256"].items():
        path = Path(config["paths"]["candidate_source_root"]) / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"post-lock source mutation: {relative}")
    return lock


def build_model(config: dict[str, Any], variant: str, device: str) -> CHHTRanker:
    values = config["model"]
    model = CHHTRanker(
        input_dim=int(values["input_dim"]),
        hidden_dim=int(values["hidden_dim"]),
        heads=int(values["heads"]),
        ffn_dim=int(values["ffn_dim"]),
        rank=int(values["rank"]),
        history_layers=int(values["history_layers"]),
        pair_layers=int(values["pair_layers"]),
        dropout=float(values["dropout"]),
        max_history=int(config["data"]["history_limit"]),
        max_skew_norm=float(values["max_skew_norm"]),
        max_score_residual=float(values["max_score_residual"]),
        variant=variant,
    )
    return model.to(device)


class FrozenFeatureStore:
    """Memory-mapped frozen states plus GPU-side D2p composition."""

    def __init__(self, config: dict[str, Any], split: str) -> None:
        paths = config["paths"]
        root = Path(paths["feature_root"])
        self.query = np.load(root / split / "query_embeddings.npy", mmap_mode="r")
        self.items = np.load(paths["shared_item_embeddings"], mmap_mode="r")
        self.item_adapter = np.load(root / "item_adapter_weight.npy", mmap_mode="r")
        self.popularity = np.load(paths["shared_item_popularity"], mmap_mode="r")
        self.category_centroids = np.load(root / "category_centroids.npy", mmap_mode="r")
        self.logit_scale = float(read_json(root / "base_parameters.json")["logit_scale"])
        self.d2p_alpha = float(config["base"]["d2p_alpha"])
        self.item_teacher_beta = float(config["base"]["item_teacher_beta"])
        self._adapter_by_device: dict[str, torch.Tensor] = {}

    def tensors(
        self,
        batch: dict[str, np.ndarray | list[str]],
        device: str,
        *,
        include_corruptions: bool,
    ) -> dict[str, torch.Tensor]:
        request_indices = np.asarray(batch["request_indices"], dtype=np.int64)
        candidate_indices = np.asarray(batch["candidate_indices"], dtype=np.int64)
        history_indices = np.asarray(batch["history_indices"], dtype=np.int64)
        query = _tensor(self.query[request_indices], device)
        candidates = self._adapt(self.items[candidate_indices], device)
        history = self._adapt(self.items[history_indices], device)
        candidate_mask = torch.from_numpy(np.asarray(batch["candidate_mask"])).to(device)
        history_mask = torch.from_numpy(np.asarray(batch["history_mask"])).to(device)
        popularity = _tensor(self.popularity[candidate_indices], device)
        text = self.logit_scale * torch.einsum("bd,bcd->bc", query, candidates)
        base = self.d2p_alpha * masked_zscore(text, candidate_mask)
        base = base + (1.0 - self.d2p_alpha) * masked_zscore(popularity, candidate_mask)
        result = {
            "query_embeddings": query,
            "candidate_embeddings": candidates,
            "history_embeddings": history,
            "base_scores": base,
            "candidate_mask": candidate_mask,
            "history_mask": history_mask,
            "history_event_weight": _tensor(batch["history_event_weights"], device),
            "repeat_mask": torch.from_numpy(np.asarray(batch["repeat_mask"])).to(device),
            "candidate_labels": _tensor(batch["candidate_labels"], device),
        }
        item_component = 3.0 * (
            result["repeat_mask"].to(base.dtype)
            * result["history_event_weight"][:, None, :]
        ).sum(dim=-1)
        result["item_teacher_scores"] = (
            self.item_teacher_beta * masked_zscore(base, candidate_mask)
            + (1.0 - self.item_teacher_beta)
            * masked_zscore(item_component, candidate_mask)
        )
        if include_corruptions:
            wrong_indices = np.asarray(batch["wrong_history_indices"], dtype=np.int64)
            shuffled_indices = np.asarray(batch["shuffled_history_indices"], dtype=np.int64)
            category_ids = np.asarray(batch["history_category_ids"], dtype=np.int64)
            result.update(
                {
                    "wrong_history_embeddings": self._adapt(
                        self.items[wrong_indices], device
                    ),
                    "wrong_history_mask": torch.from_numpy(
                        np.asarray(batch["wrong_history_mask"])
                    ).to(device),
                    "wrong_history_event_weight": _tensor(
                        batch["wrong_history_event_weights"], device
                    ),
                    "wrong_repeat_mask": torch.from_numpy(
                        np.asarray(batch["wrong_repeat_mask"])
                    ).to(device),
                    "shuffled_history_embeddings": self._adapt(
                        self.items[shuffled_indices], device
                    ),
                    "shuffled_history_mask": torch.from_numpy(
                        np.asarray(batch["shuffled_history_mask"])
                    ).to(device),
                    "shuffled_history_event_weight": _tensor(
                        batch["shuffled_history_event_weights"], device
                    ),
                    "shuffled_repeat_mask": torch.from_numpy(
                        np.asarray(batch["shuffled_repeat_mask"])
                    ).to(device),
                    "coarse_history_embeddings": self._adapt(
                        self.category_centroids[category_ids], device
                    ),
                }
            )
        return result

    def dev_tensors(
        self,
        batch: dict[str, np.ndarray | list[str]],
        device: str,
        *,
        include_corruptions: bool,
    ) -> dict[str, torch.Tensor]:
        result = self.tensors(batch, device, include_corruptions=include_corruptions)
        # The train-only label placeholder is never loaded for dev.  Replace
        # the recomputed D2p coordinate with the exact frozen float64 scores.
        result.pop("candidate_labels")
        result.pop("item_teacher_scores")
        result["base_scores"] = torch.from_numpy(
            np.asarray(batch["base_scores"], dtype=np.float64)
        ).to(device)
        return result

    def _adapt(self, values: np.ndarray, device: str) -> torch.Tensor:
        tensor = _tensor(values, device)
        if device not in self._adapter_by_device:
            self._adapter_by_device[device] = _tensor(self.item_adapter, device)
        weight = self._adapter_by_device[device]
        return F.normalize(F.linear(tensor, weight), dim=-1, eps=1e-6)


def model_inputs(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: tensors[key]
        for key in (
            "query_embeddings",
            "candidate_embeddings",
            "history_embeddings",
            "base_scores",
            "candidate_mask",
            "history_mask",
            "history_event_weight",
            "repeat_mask",
        )
    }


def corruption_inputs(
    tensors: dict[str, torch.Tensor], name: str
) -> dict[str, torch.Tensor]:
    values = model_inputs(tensors)
    if name == "wrong":
        values["history_embeddings"] = tensors["wrong_history_embeddings"]
        values["history_mask"] = tensors["wrong_history_mask"]
        values["history_event_weight"] = tensors["wrong_history_event_weight"]
        values["repeat_mask"] = tensors["wrong_repeat_mask"]
    elif name == "shuffle":
        values["history_embeddings"] = tensors["shuffled_history_embeddings"]
        values["history_mask"] = tensors["shuffled_history_mask"]
        values["history_event_weight"] = tensors["shuffled_history_event_weight"]
        values["repeat_mask"] = tensors["shuffled_repeat_mask"]
    elif name == "coarse":
        values["history_embeddings"] = tensors["coarse_history_embeddings"]
        values["repeat_mask"] = torch.zeros_like(tensors["repeat_mask"])
    elif name == "query_mask":
        values["query_embeddings"] = torch.zeros_like(tensors["query_embeddings"])
    else:
        raise ValueError(f"unknown corruption: {name}")
    return values


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def _tensor(values: Any, device: str) -> torch.Tensor:
    array = np.asarray(values, dtype=np.float32)
    return torch.from_numpy(array.copy()).to(device)


def zscore_numpy(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (values - values.mean()) / math.sqrt(float(values.var()) + 1e-6)
