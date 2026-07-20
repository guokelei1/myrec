"""Complete pre-mask scaled-QK-logit formation controls for N28."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from myrec.mechanism.transformer_instrumentation import _torch


QK_FORMATION_MODES = (
    "identity",
    "centered_scale_half",
    "centered_scale_double",
    "sign_flip",
    "head_preserving_random",
)


def _finite_center(logits: Any) -> tuple[Any, Any]:
    torch = _torch()
    if logits.ndim != 4 or not logits.is_floating_point():
        raise ValueError("QK logits must be a floating [batch,positions,heads,keys] tensor")
    finite = torch.isfinite(logits)
    if not bool(finite.any(dim=-1).all()):
        raise ValueError("QK logits contain an all-masked query row")
    safe = torch.where(finite, logits, torch.zeros_like(logits))
    count = finite.sum(dim=-1, keepdim=True).clamp_min(1)
    mean = safe.sum(dim=-1, keepdim=True) / count
    return finite, mean


def _head_preserving_random(
    logits: Any,
    finite: Any,
    *,
    identity_keys: Sequence[Sequence[str]],
    seed: int,
) -> Any:
    torch = _torch()
    if len(identity_keys) != int(logits.shape[0]) or any(
        len(row) != int(logits.shape[1]) for row in identity_keys
    ):
        raise ValueError("QK identity keys do not align with [batch,positions]")
    random = torch.empty_like(logits, dtype=torch.float32, device="cpu")
    batch, positions, heads, keys = (int(v) for v in logits.shape)
    for row in range(batch):
        for position in range(positions):
            for head in range(heads):
                digest = hashlib.sha256(
                    f"{int(seed)}\0{identity_keys[row][position]}\0head={head}".encode()
                ).digest()
                generator = torch.Generator(device="cpu")
                generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
                random[row, position, head] = torch.randn(
                    (keys,), generator=generator, dtype=torch.float32, device="cpu"
                )
    random = random.to(device=logits.device)
    random = torch.where(finite, random, torch.zeros_like(random))
    target = logits.float()
    target_mean = torch.where(finite, target, torch.zeros_like(target)).sum(-1, keepdim=True)
    count = finite.sum(-1, keepdim=True).clamp_min(1)
    target_mean = target_mean / count
    target_centered = torch.where(finite, target - target_mean, torch.zeros_like(target))
    random_mean = random.sum(-1, keepdim=True) / count
    random_centered = torch.where(finite, random - random_mean, torch.zeros_like(random))
    target_rms = target_centered.pow(2).sum(-1, keepdim=True).div(count).sqrt()
    random_rms = random_centered.pow(2).sum(-1, keepdim=True).div(count).sqrt().clamp_min(1e-12)
    result = target_mean + random_centered * (target_rms / random_rms)
    return torch.where(finite, result, logits.float()).to(dtype=logits.dtype)


def apply_qk_formation_operator(
    logits: Any,
    mode: str,
    *,
    identity_keys: Sequence[Sequence[str]] | None = None,
    random_seed: int = 20260720,
) -> Any:
    """Transform the complete scaled QK tensor while preserving mask entries."""

    torch = _torch()
    if mode not in QK_FORMATION_MODES:
        raise ValueError(f"unsupported QK formation mode={mode}")
    if mode == "identity":
        return logits
    finite, mean = _finite_center(logits)
    centered = torch.where(finite, logits - mean, torch.zeros_like(logits))
    if mode == "centered_scale_half":
        result = mean + centered * 0.5
    elif mode == "centered_scale_double":
        result = mean + centered * 2.0
    elif mode == "sign_flip":
        result = mean - centered
    elif mode == "head_preserving_random":
        if identity_keys is None:
            raise ValueError("head-preserving random mode requires identity keys")
        return _head_preserving_random(
            logits,
            finite,
            identity_keys=identity_keys,
            seed=int(random_seed),
        )
    else:  # pragma: no cover - constructor-like validation above
        raise AssertionError(mode)
    return torch.where(finite, result, logits)
