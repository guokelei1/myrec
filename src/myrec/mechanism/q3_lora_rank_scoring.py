"""Q3 LoRA rank-path ablation primitives for the N10 diagnostic wave."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.deep_dive_native_patch import _q3_context
from myrec.mechanism.q3_native_readout_scoring import capture_q3_native_readout
from myrec.mechanism.optimizer_replay_math import lora_parameter_identity


LORA_RANK = 8
LORA_PATH_CONDITIONS = (
    "baseline_full",
    "baseline_null",
    "a_only",
    "b_only",
    "no_adapter_identity",
    *(f"outer_product_rank_{rank}" for rank in range(LORA_RANK)),
)


class Q3LoraFactorPatch:
    """Temporarily replace all 28x(q/v) LoRA factors with a registered path."""

    def __init__(self, model: Any, mode: str) -> None:
        self.model = model
        self.mode = str(mode)
        self._pairs: dict[str, dict[str, Any]] = {}
        self._original: dict[str, Any] = {}
        self._active = False
        valid = {"a_only", "b_only", "no_adapter_identity"} | {
            f"outer_product_rank_{rank}" for rank in range(LORA_RANK)
        }
        if self.mode not in valid:
            raise ValueError(f"unsupported Q3 LoRA factor path: {mode}")

    def __enter__(self) -> "Q3LoraFactorPatch":
        if self._active:
            raise RuntimeError("Q3 LoRA factor patch is already active")
        named = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if ".lora_A." in name or ".lora_B." in name
        }
        for name, parameter in named.items():
            identity = lora_parameter_identity(name)
            key = f"block_{identity['block_zero_based']:02d}.{identity['projection']}_proj"
            row = self._pairs.setdefault(
                key,
                {
                    "block_zero_based": identity["block_zero_based"],
                    "projection": identity["projection"],
                },
            )
            factor = identity["factor"]
            if factor in row:
                raise ValueError(f"duplicate Q3 LoRA factor: {key}:{factor}")
            row[factor] = parameter
            self._original[name] = parameter.detach().clone()
        if len(self._pairs) != 56 or any(set(row) != {"block_zero_based", "projection", "A", "B"} for row in self._pairs.values()):
            raise ValueError("Q3 LoRA factor coverage is not 28 blocks x q/v pairs")
        with _torch().no_grad():
            for key in sorted(self._pairs):
                pair = self._pairs[key]
                a = pair["A"]
                b = pair["B"]
                if a.ndim != 2 or b.ndim != 2 or a.shape[0] != LORA_RANK or b.shape[1] != LORA_RANK:
                    raise ValueError(f"Q3 LoRA rank shape differs at {key}")
                if self.mode in {"a_only", "no_adapter_identity"}:
                    b.zero_()
                elif self.mode == "b_only":
                    a.zero_()
                else:
                    rank = int(self.mode.rsplit("_", 1)[-1])
                    if not 0 <= rank < LORA_RANK:
                        raise AssertionError(rank)
                    a_copy = a.detach().clone()
                    b_copy = b.detach().clone()
                    a.zero_()
                    b.zero_()
                    a[rank : rank + 1].copy_(a_copy[rank : rank + 1])
                    b[:, rank : rank + 1].copy_(b_copy[:, rank : rank + 1])
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self._active:
            with _torch().no_grad():
                for name, original in self._original.items():
                    parameter = next(
                        parameter
                        for parameter_name, parameter in self.model.named_parameters()
                        if parameter_name == name
                    )
                    parameter.copy_(original.to(parameter.device, dtype=parameter.dtype))
        self._pairs = {}
        self._original = {}
        self._active = False


def score_q3_lora_rank_chunk(
    model: Any,
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    history: Sequence[Mapping[str, Any]],
    device: str,
) -> dict[str, np.ndarray]:
    """Return Q3 native scores for every registered factor path."""

    if not candidates:
        raise ValueError("Q3 LoRA rank candidate chunk is empty")
    context = _q3_context(tokenizer, record, candidates, history, config, device)
    results: dict[str, np.ndarray] = {}
    results["baseline_full"] = _capture_score(model, context)
    for condition in LORA_PATH_CONDITIONS[2:]:
        with Q3LoraFactorPatch(model, condition):
            results[condition] = _capture_score(model, context)
    if any(values.shape != (len(candidates),) or not np.isfinite(values).all() for values in results.values()):
        raise FloatingPointError("Q3 LoRA rank path produced invalid score coverage")
    return results


def _capture_score(model: Any, context: Mapping[str, Any]) -> np.ndarray:
    value = capture_q3_native_readout(model, context)["score"]
    return value.detach().float().cpu().numpy()


def lora_rank_implementation_identity() -> dict[str, Any]:
    from pathlib import Path

    from myrec.mechanism.attention_edge_runtime import _canonical_sha256
    from myrec.utils.hashing import sha256_file

    root = Path(__file__).resolve().parents[3]
    paths = (
        "src/myrec/mechanism/q3_lora_rank_scoring.py",
        "src/myrec/mechanism/q3_native_readout_scoring.py",
        "src/myrec/mechanism/deep_dive_native_patch.py",
        "src/myrec/mechanism/optimizer_replay_math.py",
        "scripts/score_deep_dive_q3_lora_rank_paths.py",
    )
    files = [
        {"path": path, "sha256": sha256_file(root / path), "size_bytes": (root / path).stat().st_size}
        for path in paths
    ]
    return {"files": files, "digest": _canonical_sha256(files)}


def _torch() -> Any:
    import torch

    return torch

