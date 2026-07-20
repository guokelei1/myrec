"""Native Q2 final-RMSNorm/readout capture and intervention primitives."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from myrec.baselines.motivation_v12_contracts import ModelRecord
from myrec.mechanism.patch_scorer import _left_pad_sequences
from myrec.mechanism.representation_probe import instrument_pointwise_prompt

from myrec.mechanism.transformer_instrumentation import (
    NodeSpec,
    QwenNodeCapture,
    QwenNodePatch,
)


Q2_FINAL_NODES = ("final_rmsnorm_input", "final_rmsnorm_output")


def build_q2_pointwise_batch(
    tokenizer: Any,
    record: ModelRecord,
    candidates: Sequence[Mapping[str, Any]],
    history: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    device: str,
) -> tuple[Any, Any, Any]:
    """Build one frozen Q2 pointwise batch and its native readout positions."""

    if not candidates:
        raise ValueError("Q2 pointwise readout batch is empty")
    if config.get("method_id") != "q2_recranker_generalqwen":
        raise ValueError("native Q2 readout builder received another method")
    training = config["training"]
    prompts = [
        instrument_pointwise_prompt(
            tokenizer,
            str(config["method_id"]),
            record,
            candidate,
            history=history,
            history_budget=int(training["history_budget"]),
            max_length=int(training["max_length"]),
        )
        for candidate in candidates
    ]
    ids, mask, padding = _left_pad_sequences(
        [prompt.token_ids for prompt in prompts], tokenizer.pad_token_id, device
    )
    positions = _torch().tensor(
        [
            [left + int(prompt.candidate_readout)]
            for left, prompt in zip(padding, prompts)
        ],
        dtype=_torch().long,
        device=device,
    )
    if any(int(position) != ids.shape[1] - 1 for position in positions[:, 0]):
        raise ValueError("Q2 candidate readout is not the final prompt token")
    return ids, mask, positions


def capture_q2_native_readout(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    positions: Any,
    *,
    yes_token_id: int = 9693,
    no_token_id: int = 2152,
) -> dict[str, Any]:
    """Capture Q2 final-norm states and verify the tied-row score algebra."""

    torch = _torch()
    specs = tuple(NodeSpec(node_id=name, block=None) for name in Q2_FINAL_NODES)
    with QwenNodeCapture(model, specs) as capture:
        output, states = capture.capture_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            positions=positions,
            model_kwargs={"logits_to_keep": 1},
        )
    if positions.shape[1] != 1:
        raise ValueError("Q2 native readout requires one position per pointwise row")
    logits = output.logits[:, -1].float()
    yes = logits[:, int(yes_token_id)]
    no = logits[:, int(no_token_id)]
    native = yes - no
    final_input = states["final_rmsnorm_input"][:, 0]
    final_output = states["final_rmsnorm_output"][:, 0]
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None or not hasattr(output_embeddings, "weight"):
        raise TypeError("Q2 model has no output embedding weight")
    weight = output_embeddings.weight
    readout_direction = (
        weight[int(yes_token_id)] - weight[int(no_token_id)]
    ).to(device=final_output.device, dtype=final_output.dtype)
    algebra = torch.nn.functional.linear(final_output, readout_direction).float()
    algebra_error = float((algebra - native).abs().max().item())
    algebra_bound = 4.0 * (2.0**-7) * torch.maximum(
        torch.ones_like(native), native.abs()
    )
    algebra_ratio = float(((algebra - native).abs() / algebra_bound).max().item())
    common_offset = 0.5 * (yes + no)
    for name, value in (
        ("native_score", native),
        ("algebra_score", algebra),
        ("common_offset", common_offset),
        ("final_input", final_input),
        ("final_output", final_output),
    ):
        if not bool(torch.isfinite(value).all().item()):
            raise FloatingPointError(f"Q2 readout capture is non-finite: {name}")
    return {
        "native_score": native,
        "algebra_score": algebra,
        "algebra_max_abs_error": algebra_error,
        "algebra_low_precision_max_ratio": algebra_ratio,
        "common_offset": common_offset,
        "yes_logit": yes,
        "no_logit": no,
        "final_rmsnorm_input": final_input,
        "final_rmsnorm_output": final_output,
        "readout_direction": readout_direction,
        "geometry": {
            "input_norm": final_input.float().norm(dim=-1),
            "output_norm": final_output.float().norm(dim=-1),
            "input_output_cosine": torch.nn.functional.cosine_similarity(
                final_input.float(), final_output.float(), dim=-1
            ),
        },
    }


def score_q2_with_final_node_patch(
    model: Any,
    input_ids: Any,
    attention_mask: Any,
    positions: Any,
    donor_vectors: Any,
    *,
    node_id: str,
    yes_token_id: int = 9693,
    no_token_id: int = 2152,
) -> Any:
    """Patch one Q2 final-norm node and return native yes-minus-no scores."""

    if node_id not in Q2_FINAL_NODES:
        raise ValueError("Q2 native readout patch node is not registered")
    if positions.ndim != 2 or positions.shape[1] != 1:
        raise ValueError("Q2 final-node patch requires [batch,1] positions")
    if donor_vectors.ndim == 2:
        donor_vectors = donor_vectors[:, None, :]
    if donor_vectors.ndim != 3 or donor_vectors.shape[:2] != positions.shape:
        raise ValueError("Q2 final-node donor vectors are misaligned")
    with QwenNodePatch(model, NodeSpec(node_id=node_id, block=None)) as patch:
        patch.arm(positions, donor_vectors, sequence_length=input_ids.shape[1])
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            logits_to_keep=1,
        )
        patch.disarm()
    logits = output.logits[:, -1].float()
    score = logits[:, int(yes_token_id)] - logits[:, int(no_token_id)]
    if not bool(_torch().isfinite(score).all().item()):
        raise FloatingPointError("Q2 final-node patch score is non-finite")
    return score


def decompose_request_scores(scores: Any) -> Mapping[str, Any]:
    """Return the exact per-request common and candidate-relative components."""

    if scores.ndim != 1 or scores.numel() <= 0:
        raise ValueError("request score decomposition requires a nonempty vector")
    common = scores.mean()
    relative = scores - common
    residual = scores - (common + relative)
    return {
        "common": common,
        "relative": relative,
        "recomposition_max_abs_error": float(residual.abs().max().item()),
        "relative_sum_abs_error": float(relative.sum().abs().item()),
    }


def _torch() -> Any:
    import torch

    return torch
