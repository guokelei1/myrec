"""Exact vector accounting for D7 AdamW and LoRA effective updates."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Mapping, Sequence


Q2_PARAMETER_FAMILIES = (
    "embedding_readout",
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_o",
    "rmsnorm",
    "mlp_gate",
    "mlp_up",
    "mlp_down",
)
_LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def parameter_order_rows(named_parameters: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact frozen optimizer order identity payload."""

    return [
        {
            "index": index,
            "name": str(name),
            "shape": list(parameter.shape),
            "dtype": str(parameter.dtype),
        }
        for index, (name, parameter) in enumerate(named_parameters)
    ]


def parameter_order_digest(named_parameters: Sequence[tuple[str, Any]]) -> str:
    payload = json.dumps(
        parameter_order_rows(named_parameters),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def q2_parameter_family(name: str) -> str:
    """Map every Qwen trainable object to one mutually-exclusive family."""

    name = str(name)
    if name in {"model.embed_tokens.weight", "lm_head.weight"}:
        return "embedding_readout"
    if name == "model.norm.weight" or any(
        token in name
        for token in (
            ".input_layernorm.",
            ".post_attention_layernorm.",
            ".self_attn.q_norm.",
            ".self_attn.k_norm.",
        )
    ):
        return "rmsnorm"
    for projection, family in (
        ("q_proj", "attention_q"),
        ("k_proj", "attention_k"),
        ("v_proj", "attention_v"),
        ("o_proj", "attention_o"),
    ):
        if f".self_attn.{projection}." in name:
            return family
    for projection, family in (
        ("gate_proj", "mlp_gate"),
        ("up_proj", "mlp_up"),
        ("down_proj", "mlp_down"),
    ):
        if f".mlp.{projection}." in name:
            return family
    raise ValueError(f"unclassified Q2 optimizer parameter: {name}")


def lora_parameter_identity(name: str) -> dict[str, Any]:
    match = _LAYER.search(str(name))
    if match is None:
        raise ValueError(f"LoRA parameter lacks a layer index: {name}")
    projection = "q" if ".q_proj." in name else "v" if ".v_proj." in name else None
    factor = "A" if ".lora_A." in name else "B" if ".lora_B." in name else None
    if projection is None or factor is None:
        raise ValueError(f"invalid q/v LoRA parameter: {name}")
    return {
        "block_zero_based": int(match.group(1)),
        "projection": projection,
        "factor": factor,
    }


def global_gradient_norm(gradients: Mapping[str, Any]) -> float:
    squared = sum(
        float(gradient.detach().double().square().sum().item())
        for gradient in gradients.values()
        if gradient is not None
    )
    if not math.isfinite(squared) or squared < 0:
        raise FloatingPointError("gradient norm mass is invalid")
    return math.sqrt(squared)


def clip_gradients(
    gradients: Mapping[str, Any], max_norm: float
) -> tuple[dict[str, Any], float, float]:
    """Apply PyTorch global-norm clipping algebra without mutating inputs."""

    norm = global_gradient_norm(gradients)
    max_norm = float(max_norm)
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError("max_norm must be finite and positive")
    coefficient = min(1.0, max_norm / (norm + 1.0e-6))
    return {
        name: None if gradient is None else gradient * coefficient
        for name, gradient in gradients.items()
    }, coefficient, norm


def adamw_exact_delta(
    parameter: Any,
    gradient: Any,
    state: Mapping[str, Any],
    group: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the exact next-step AdamW components for one dense tensor."""

    torch = _torch()
    if gradient is None:
        return {
            "preconditioned_direction": None,
            "moment_delta": None,
            "weight_decay_delta": None,
            "total_delta": None,
        }
    if bool(group.get("amsgrad", False)) or bool(group.get("maximize", False)):
        raise ValueError("registered D7 replay supports standard non-AMSGrad AdamW")
    beta1, beta2 = map(float, group["betas"])
    epsilon = float(group["eps"])
    learning_rate = float(group["lr"])
    weight_decay = float(group["weight_decay"])
    step_value = state["step"]
    step = int(step_value.item() if hasattr(step_value, "item") else step_value) + 1
    exp_avg = state["exp_avg"].to(parameter.device, dtype=gradient.dtype)
    exp_avg_sq = state["exp_avg_sq"].to(parameter.device, dtype=gradient.dtype)
    next_exp_avg = exp_avg * beta1 + gradient * (1.0 - beta1)
    next_exp_avg_sq = exp_avg_sq * beta2 + gradient.square() * (1.0 - beta2)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2 = 1.0 - beta2**step
    direction = (next_exp_avg / bias_correction1) / (
        (next_exp_avg_sq / bias_correction2).sqrt() + epsilon
    )
    moment_delta = -learning_rate * direction
    weight_decay_delta = -learning_rate * weight_decay * parameter.detach()
    total_delta = moment_delta + weight_decay_delta
    for value in (direction, moment_delta, weight_decay_delta, total_delta):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError("AdamW replay produced a non-finite tensor")
    return {
        "preconditioned_direction": direction,
        "moment_delta": moment_delta,
        "weight_decay_delta": weight_decay_delta,
        "total_delta": total_delta,
    }


def vector_summary(
    vectors: Mapping[str, Any],
    *,
    family_by_name: Mapping[str, str],
) -> dict[str, Any]:
    """Summarize squared-L2 mass for a mutually-exclusive partition."""

    if set(vectors) != set(family_by_name):
        raise ValueError("vector names and family partition differ")
    family_mass: dict[str, float] = {}
    total = 0.0
    active = 0
    for name, vector in vectors.items():
        if vector is None:
            continue
        mass = float(vector.detach().double().square().sum().item())
        if not math.isfinite(mass):
            raise FloatingPointError(f"vector mass is non-finite: {name}")
        family = family_by_name[name]
        family_mass[family] = family_mass.get(family, 0.0) + mass
        total += mass
        active += 1
    return {
        "active_parameter_objects": active,
        "norm": math.sqrt(total),
        "squared_norm": total,
        "family_squared_norm": dict(sorted(family_mass.items())),
        "family_share": {
            family: mass / total if total > 0 else None
            for family, mass in sorted(family_mass.items())
        },
    }


def vector_cosine(left: Mapping[str, Any], right: Mapping[str, Any]) -> float | None:
    if set(left) != set(right) or not left:
        raise ValueError("cosine vectors have different coverage")
    dot = left_mass = right_mass = 0.0
    for name in left:
        lvalue, rvalue = left[name], right[name]
        if lvalue is None and rvalue is None:
            continue
        if lvalue is None or rvalue is None or lvalue.shape != rvalue.shape:
            raise ValueError("cosine vectors have different active shapes")
        left_double = lvalue.detach().double()
        right_double = rvalue.detach().double()
        dot += float((left_double * right_double).sum().item())
        left_mass += float(left_double.square().sum().item())
        right_mass += float(right_double.square().sum().item())
    if left_mass <= 0 or right_mass <= 0:
        return None
    return max(-1.0, min(1.0, dot / math.sqrt(left_mass * right_mass)))


def vector_relative_error(
    observed: Mapping[str, Any], expected: Mapping[str, Any]
) -> dict[str, float]:
    """Return exact max-absolute and relative-L2 vector identity errors."""

    if set(observed) != set(expected):
        raise ValueError("vector identity coverage differs")
    maximum = error_mass = reference_mass = 0.0
    for name in observed:
        left = observed[name].detach().double()
        right = expected[name].detach().double()
        if left.shape != right.shape:
            raise ValueError(f"vector identity shape differs: {name}")
        error = left - right
        maximum = max(maximum, float(error.abs().max().item()))
        error_mass += float(error.square().sum().item())
        reference_mass += float(right.square().sum().item())
    return {
        "maximum_absolute_error": maximum,
        "relative_l2_error": (
            math.sqrt(error_mass / reference_mass) if reference_mass > 0 else 0.0
        ),
    }


def gradient_pair_summary(
    names: Sequence[str],
    left: Sequence[Any],
    right: Sequence[Any],
    *,
    family_by_name: Mapping[str, str],
    chunk_size: int = 1_048_576,
) -> dict[str, Any]:
    """Summarize two full-model gradients without materializing double copies."""

    if not (len(names) == len(left) == len(right)) or set(names) != set(family_by_name):
        raise ValueError("full-gradient pair coverage differs")
    dot = left_mass = right_mass = 0.0
    left_family: dict[str, float] = {}
    right_family: dict[str, float] = {}
    for name, lvalue, rvalue in zip(names, left, right):
        if lvalue is None or rvalue is None or lvalue.shape != rvalue.shape:
            raise ValueError(f"full-gradient tensor differs: {name}")
        family = family_by_name[name]
        local_dot = local_left = local_right = 0.0
        lflat = lvalue.detach().reshape(-1)
        rflat = rvalue.detach().reshape(-1)
        for start in range(0, lflat.numel(), int(chunk_size)):
            lchunk = lflat[start : start + int(chunk_size)].double()
            rchunk = rflat[start : start + int(chunk_size)].double()
            local_dot += float((lchunk * rchunk).sum().item())
            local_left += float(lchunk.square().sum().item())
            local_right += float(rchunk.square().sum().item())
        dot += local_dot
        left_mass += local_left
        right_mass += local_right
        left_family[family] = left_family.get(family, 0.0) + local_left
        right_family[family] = right_family.get(family, 0.0) + local_right
    cosine = None
    if left_mass > 0 and right_mass > 0:
        cosine = max(-1.0, min(1.0, dot / math.sqrt(left_mass * right_mass)))
    return {
        "cosine": cosine,
        "left_norm": math.sqrt(left_mass),
        "right_norm": math.sqrt(right_mass),
        "left_family_share": {
            family: mass / left_mass if left_mass > 0 else None
            for family, mass in sorted(left_family.items())
        },
        "right_family_share": {
            family: mass / right_mass if right_mass > 0 else None
            for family, mass in sorted(right_family.items())
        },
    }


def lora_function_delta(
    a: Any, b: Any, delta_a: Any, delta_b: Any, *, scaling: float = 2.0
) -> dict[str, Any]:
    """Exact gauge-invariant LoRA function update decomposition."""

    a_only = float(scaling) * (b @ delta_a)
    b_only = float(scaling) * (delta_b @ a)
    interaction = float(scaling) * (delta_b @ delta_a)
    joint = float(scaling) * ((b + delta_b) @ (a + delta_a) - b @ a)
    residual = joint - a_only - b_only - interaction
    return {
        "a_only": a_only,
        "b_only": b_only,
        "interaction": interaction,
        "joint": joint,
        "recomposition_max_abs_error": float(residual.abs().max().item()),
    }


def lora_singular_values(a: Any, b: Any, *, scaling: float = 2.0) -> dict[str, Any]:
    """Return nonzero singular values from the small rank-space Gram matrix."""

    torch = _torch()
    # DeltaW is at most rank r.  The full matrices here are only 2048x1024,
    # but direct SVD is needlessly expensive; QR factors preserve singular
    # values exactly up to floating-point algebra.
    qb, rb = torch.linalg.qr(b.detach().double(), mode="reduced")
    qa, ra_t = torch.linalg.qr(a.detach().double().T, mode="reduced")
    core = float(scaling) * (rb @ ra_t.T)
    singular = torch.linalg.svdvals(core)
    largest = float(singular[0].item()) if singular.numel() else 0.0
    threshold = max(largest * 1.0e-6, 1.0e-8)
    values = [float(value) for value in singular.tolist()]
    return {
        "singular_values_descending": values,
        "near_zero_threshold": threshold,
        "effective_rank": sum(value > threshold for value in values),
        "degenerate_adjacent_pairs": [
            index
            for index in range(max(0, len(values) - 1))
            if values[index] > threshold
            and abs(values[index] - values[index + 1]) / values[index] < 1.0e-4
        ],
    }


def _torch() -> Any:
    import torch

    return torch
