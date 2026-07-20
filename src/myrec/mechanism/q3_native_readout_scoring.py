"""Exact three-state native-readout decomposition for frozen Q3 scoring.

Q3 ranks a candidate with two teacher-forced likelihood paths.  Its scalar
score therefore contains four log-probability terms at three causal states:
the shared prompt state, the state after ``Yes``, and the state after ``No``.
This module keeps those terms separate and implements final-readout output
interventions by exact term substitution.  No qrels are read here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from myrec.mechanism.transformer_instrumentation import NodeSpec, QwenNodeCapture


Q3_FINAL_NODES = ("final_rmsnorm_input", "final_rmsnorm_output")
Q3_READOUT_SCOPES = (
    "shared_prompt",
    "yes_context",
    "no_context",
    "joint",
)
Q3_TERM_NAMES = (
    "prompt_predict_yes",
    "yes_predict_terminator",
    "prompt_predict_no",
    "no_predict_terminator",
)
LOW_PRECISION_RATIO_TOLERANCE = 1.0e-4
_SCOPE_TERM_INDICES = {
    "shared_prompt": (0, 2),
    "yes_context": (1,),
    "no_context": (3,),
    "joint": (0, 1, 2, 3),
}


def capture_q3_native_readout(
    model: Any,
    context: Mapping[str, Any],
) -> dict[str, Any]:
    """Capture Q3's final-norm states and four native likelihood terms."""

    torch = _torch()
    paths = context.get("paths")
    if not isinstance(paths, Mapping) or set(paths) != {"yes", "no"}:
        raise ValueError("Q3 native readout requires equal Yes/No paths")
    specs = tuple(NodeSpec(node_id=name, block=None) for name in Q3_FINAL_NODES)
    branch: dict[str, Any] = {}
    native_terms = []
    algebra_terms = []
    with QwenNodeCapture(model, specs) as capture:
        for name in ("yes", "no"):
            path = paths[name]
            positions = path["positions"]
            if positions.ndim != 2 or positions.shape[1] != 2:
                raise ValueError("Q3 native path must expose two readout positions")
            output, states = capture.capture_forward(
                input_ids=path["ids"],
                attention_mask=path["mask"],
                positions=positions,
                model_kwargs={"logits_to_keep": 3},
            )
            terms = _path_terms(output, path)
            reconstructed = _hidden_path_terms(
                model, states["final_rmsnorm_output"], path["target"]
            )
            native_terms.append(terms)
            algebra_terms.append(reconstructed)
            branch[name] = {
                "final_rmsnorm_input": states["final_rmsnorm_input"],
                "final_rmsnorm_output": states["final_rmsnorm_output"],
                "terms": terms,
                "algebra_terms": reconstructed,
            }

    terms = _combine_path_terms(native_terms[0], native_terms[1])
    reconstructed = _combine_path_terms(algebra_terms[0], algebra_terms[1])
    score = q3_native_score_from_terms(terms)
    algebra_score = q3_native_score_from_terms(reconstructed)
    error = (reconstructed - terms).abs()
    bound = 4.0 * (2.0**-7) * torch.maximum(torch.ones_like(terms), terms.abs())
    shared_deltas = {
        node: float(
            (
                branch["yes"][node][:, 0].float()
                - branch["no"][node][:, 0].float()
            )
            .abs()
            .max()
            .item()
        )
        for node in Q3_FINAL_NODES
    }
    geometry = {}
    for name in ("yes", "no"):
        input_state = branch[name]["final_rmsnorm_input"].float()
        output_state = branch[name]["final_rmsnorm_output"].float()
        geometry[name] = {
            "input_norm": input_state.norm(dim=-1),
            "output_norm": output_state.norm(dim=-1),
            "input_output_cosine": torch.nn.functional.cosine_similarity(
                input_state, output_state, dim=-1
            ),
        }
    result = {
        "terms": terms,
        "score": score,
        "algebra_terms": reconstructed,
        "algebra_score": algebra_score,
        "algebra_max_abs_error": float(error.max().item()),
        "algebra_low_precision_max_ratio": float((error / bound).max().item()),
        "shared_prompt_path_max_abs_delta": shared_deltas,
        "branches": branch,
        "geometry": geometry,
    }
    if not _all_finite(result):
        raise FloatingPointError("Q3 native-readout capture contains a non-finite value")
    return result


def compose_q3_readout_terms(
    recipient_terms: Any,
    donor_terms: Any,
    *,
    scope: str,
) -> dict[str, Any]:
    """Apply an exact final-readout intervention by substituting native terms.

    Final RMSNorm output is consumed only by the tied lm-head.  It is not fed
    back into later token states.  Replacing a registered final-output state is
    therefore exactly equivalent to replacing the corresponding target-token
    log-probability term computed from that state.
    """

    if scope not in _SCOPE_TERM_INDICES:
        raise ValueError(f"unregistered Q3 readout scope: {scope}")
    if recipient_terms.ndim != 2 or recipient_terms.shape[1] != 4:
        raise ValueError("Q3 recipient terms must have shape [batch,4]")
    if donor_terms.shape != recipient_terms.shape:
        raise ValueError("Q3 donor terms are misaligned")
    mixed = recipient_terms.clone()
    indices = _torch().tensor(
        _SCOPE_TERM_INDICES[scope], dtype=_torch().long, device=mixed.device
    )
    mixed.index_copy_(1, indices, donor_terms.index_select(1, indices))
    return {"terms": mixed, "score": q3_native_score_from_terms(mixed)}


def q3_native_score_from_terms(terms: Any) -> Any:
    """Return ``mean(Yes path)-mean(No path)`` from ``[batch,4]`` terms."""

    if terms.ndim != 2 or terms.shape[1] != 4:
        raise ValueError("Q3 native score terms must have shape [batch,4]")
    score = 0.5 * (terms[:, 0] + terms[:, 1] - terms[:, 2] - terms[:, 3])
    if not bool(_torch().isfinite(score).all().item()):
        raise FloatingPointError("Q3 native score is non-finite")
    return score


def q3_score_low_precision_bound(terms: Any) -> Any:
    """Propagate the frozen BF16 bound through Q3's signed path sum.

    Bounding only the final Yes-minus-No scalar is invalid when the two
    O(1) likelihood paths cancel.  Each path is bounded first and the signed
    subtraction is then covered by the triangle inequality.
    """

    if terms.ndim != 2 or terms.shape[1] != 4:
        raise ValueError("Q3 low-precision terms must have shape [batch,4]")
    torch = _torch()
    yes_path = 0.5 * (terms[:, 0] + terms[:, 1])
    no_path = 0.5 * (terms[:, 2] + terms[:, 3])
    unit = 4.0 * (2.0**-7)
    return unit * (
        torch.maximum(torch.ones_like(yes_path), yes_path.abs())
        + torch.maximum(torch.ones_like(no_path), no_path.abs())
    )


def _path_terms(output: Any, path: Mapping[str, Any]) -> Any:
    torch = _torch()
    target = list(path["target"])
    if len(target) != 2:
        raise ValueError("Q3 readout path target must contain exactly two tokens")
    logits = output.logits[:, -3:-1].float()
    expected = torch.tensor(target, dtype=torch.long, device=logits.device)
    expected = expected[None, :, None].expand(logits.shape[0], -1, -1)
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(
        2, expected
    ).squeeze(2)


def _hidden_path_terms(model: Any, hidden: Any, target: Sequence[int]) -> Any:
    torch = _torch()
    if hidden.ndim != 3 or hidden.shape[1] != 2 or len(target) != 2:
        raise ValueError("Q3 hidden readout algebra received an invalid path")
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise TypeError("Q3 model has no output embeddings")
    logits = output_embeddings(hidden).float()
    expected = torch.tensor(list(target), dtype=torch.long, device=logits.device)
    expected = expected[None, :, None].expand(logits.shape[0], -1, -1)
    return torch.nn.functional.log_softmax(logits, dim=-1).gather(
        2, expected
    ).squeeze(2)


def _combine_path_terms(yes_terms: Any, no_terms: Any) -> Any:
    if yes_terms.ndim != 2 or yes_terms.shape[1] != 2 or no_terms.shape != yes_terms.shape:
        raise ValueError("Q3 Yes/No likelihood terms are misaligned")
    return _torch().stack(
        (yes_terms[:, 0], yes_terms[:, 1], no_terms[:, 0], no_terms[:, 1]),
        dim=1,
    )


def _all_finite(value: Any) -> bool:
    torch = _torch()
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if hasattr(value, "is_floating_point") and value.is_floating_point():
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, (float, int)):
        return math.isfinite(float(value))
    return True


def _torch() -> Any:
    import torch

    return torch
