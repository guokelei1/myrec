"""Per-head selected-row observations without materializing full attention."""

from __future__ import annotations

from typing import Any, Mapping

from myrec.mechanism.attention_edge_interventions import (
    _apply_selected_attention_mask,
    _repeat_kv_for_query_heads,
)
from myrec.mechanism.transformer_instrumentation import resolve_qwen_backbone


OBSERVATION_QUERY_SCOPES = ("history_summary", "native_readout")
OBSERVATION_SPANS = ("query", "history", "candidate")


class QwenAttentionHeadObserver:
    """Capture Q/K stages and exact selected-row per-head routing summaries."""

    def __init__(self, model: Any, block: int) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("attention observation block must be in [0,27]")
        self.model = model
        self.block = block
        self.backbone = resolve_qwen_backbone(model)
        self.attention = self.backbone.layers[block].self_attn
        self.interface: Any = None
        self.original_function: Any = None
        self.original_implementation: str | None = None
        self.original_key_present = False
        self.handles: list[Any] = []
        self.capture_positions: Any = None
        self.query_positions: dict[str, Any] = {}
        self.spans: dict[str, tuple[Any, Any]] = {}
        self.captures: dict[str, Any] = {}
        self.fire_counts: dict[str, int] = {}
        self.observations: dict[str, Any] = {}
        self._active = False

    def __enter__(self) -> "QwenAttentionHeadObserver":
        if self._active:
            raise RuntimeError("attention head observer is already active")
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

        implementations = {
            str(layer.self_attn.config._attn_implementation)
            for layer in self.backbone.layers
        }
        if len(implementations) != 1:
            raise ValueError("Qwen layers do not share one attention backend")
        implementation = next(iter(implementations))
        self.interface = ALL_ATTENTION_FUNCTIONS
        self.original_implementation = implementation
        self.original_key_present = implementation in self.interface
        self.original_function = self.interface.get_interface(
            implementation, eager_attention_forward
        )
        self.interface[implementation] = self._wrapper
        self.handles = [
            self.attention.q_proj.register_forward_hook(self._projection_hook("q_pre_norm", "q")),
            self.attention.q_norm.register_forward_hook(self._norm_hook("q_post_norm")),
            self.attention.k_proj.register_forward_hook(self._projection_hook("k_pre_norm", "k")),
            self.attention.k_norm.register_forward_hook(self._norm_hook("k_post_norm")),
        ]
        self._active = True
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        if self._active and self.interface is not None and self.original_implementation:
            if self.original_key_present:
                self.interface[self.original_implementation] = self.original_function
            elif self.original_implementation in self.interface:
                del self.interface[self.original_implementation]
        self._active = False
        self.interface = None
        self.original_function = None
        self.original_implementation = None
        self.original_key_present = False
        self._clear()

    def arm(
        self,
        capture_positions: Any,
        query_positions: Mapping[str, Any],
        spans: Mapping[str, tuple[Any, Any]],
        *,
        sequence_length: int,
    ) -> None:
        if not self._active or self.capture_positions is not None:
            raise RuntimeError("attention head observer cannot be armed")
        if capture_positions.ndim != 2 or capture_positions.shape[1] <= 0:
            raise ValueError("observation capture positions must be [batch,positive]")
        if set(query_positions) != set(OBSERVATION_QUERY_SCOPES):
            raise ValueError("attention observation query scopes differ")
        if set(spans) != set(OBSERVATION_SPANS):
            raise ValueError("attention observation spans differ")
        batch = capture_positions.shape[0]
        sequence_length = int(sequence_length)
        if int(capture_positions.min()) < 0 or int(capture_positions.max()) >= sequence_length:
            raise ValueError("attention capture position is outside sequence")
        normalized_queries = {}
        for name, positions in query_positions.items():
            if positions.ndim == 1:
                positions = positions[:, None]
            if positions.ndim != 2 or positions.shape[0] != batch or positions.shape[1] <= 0:
                raise ValueError(f"attention query positions differ: {name}")
            if int(positions.min()) < 0 or int(positions.max()) >= sequence_length:
                raise ValueError(f"attention query position outside sequence: {name}")
            normalized_queries[name] = positions
        normalized_spans = {}
        for name, (starts, ends) in spans.items():
            if starts.ndim != 1 or starts.shape != ends.shape or len(starts) != batch:
                raise ValueError(f"attention span arrays differ: {name}")
            if int(starts.min()) < 0 or int(ends.max()) > sequence_length or bool((ends <= starts).any()):
                raise ValueError(f"attention span invalid: {name}")
            normalized_spans[name] = (starts, ends)
        self.capture_positions = capture_positions
        self.query_positions = normalized_queries
        self.spans = normalized_spans
        self.captures = {}
        self.observations = {}
        self.fire_counts = {
            "q_pre_norm": 0,
            "q_post_norm": 0,
            "k_pre_norm": 0,
            "k_post_norm": 0,
            "interface": 0,
        }

    def disarm(self) -> dict[str, Any]:
        if self.capture_positions is None:
            raise RuntimeError("attention head observer is not armed")
        invalid = {key: count for key, count in self.fire_counts.items() if count != 1}
        if invalid:
            raise RuntimeError(f"attention observation fire-count mismatch: {invalid}")
        result = {
            "captures": dict(self.captures),
            "observations": dict(self.observations),
        }
        self._clear()
        return result

    def _clear(self) -> None:
        self.capture_positions = None
        self.query_positions = {}
        self.spans = {}
        self.captures = {}
        self.observations = {}
        self.fire_counts = {}

    def _projection_hook(self, name: str, kind: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if self.capture_positions is None:
                raise RuntimeError(f"{name} observation hook fired while unarmed")
            heads = int(
                self.attention.config.num_attention_heads
                if kind == "q"
                else self.attention.config.num_key_value_heads
            )
            reshaped = output.view(output.shape[0], output.shape[1], heads, -1)
            self.captures[name] = _select_positions(reshaped, self.capture_positions).detach()
            self.fire_counts[name] += 1
        return hook

    def _norm_hook(self, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if self.capture_positions is None:
                raise RuntimeError(f"{name} observation hook fired while unarmed")
            self.captures[name] = _select_positions(output, self.capture_positions).detach()
            self.fire_counts[name] += 1
        return hook

    def _wrapper(
        self,
        module: Any,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Any,
        **kwargs: Any,
    ) -> tuple[Any, Any]:
        assert self.original_function is not None
        baseline_output, baseline_weights = self.original_function(
            module, query, key, value, attention_mask, **kwargs
        )
        if int(module.layer_idx) != self.block:
            return baseline_output, baseline_weights
        if self.capture_positions is None:
            raise RuntimeError("registered attention observation block fired while unarmed")
        self.captures["q_post_rope"] = _select_positions(
            query.transpose(1, 2), self.capture_positions
        ).detach()
        self.captures["k_post_rope"] = _select_positions(
            key.transpose(1, 2), self.capture_positions
        ).detach()
        repeated_key, repeated_value = _repeat_kv_for_query_heads(module, key, value)
        scope_results = {}
        maximum_manual_error = 0.0
        maximum_manual_low_precision_ratio = 0.0
        for scope, positions in self.query_positions.items():
            selected_query = _select_positions(query.transpose(1, 2), positions)
            scaling = kwargs.get("scaling")
            if scaling is None:
                scaling = getattr(module, "scaling", query.shape[-1] ** -0.5)
            # The native SDPA backend accumulates the QK product, softmax, and
            # probability-weighted V sum in FP32 for BF16 inputs.  Casting the
            # QK product back to BF16 before softmax creates an avoidable second
            # quantization and can exceed the frozen 4*eps reconstruction gate
            # even though the delegated native score is unchanged.  Keep this
            # independent eager reconstruction in FP32 through the V sum.
            logits = _torch().einsum(
                "bphd,bhkd->bphk",
                selected_query.float(),
                repeated_key.float(),
            )
            logits = _apply_selected_attention_mask(
                logits * float(scaling), attention_mask,
                _torch().arange(query.shape[0], device=query.device)[:, None],
                positions.to(query.device),
            )
            probabilities = _torch().softmax(logits, dim=-1)
            manual_total = _torch().einsum(
                "bphk,bhkd->bphd", probabilities, repeated_value.float()
            )
            native = _select_positions(baseline_output, positions)
            manual_error = float(
                (native.float() - manual_total.float()).abs().max().item()
            )
            reference_scale = max(
                1.0,
                float(native.detach().float().abs().max().item()),
            )
            epsilon = float(_torch().finfo(native.dtype).eps)
            manual_bound = 4.0 * epsilon * reference_scale
            maximum_manual_error = max(maximum_manual_error, manual_error)
            maximum_manual_low_precision_ratio = max(
                maximum_manual_low_precision_ratio,
                manual_error / manual_bound,
            )
            span_results = {}
            for span, (starts, ends) in self.spans.items():
                selector = _span_selector(starts, ends, key.shape[2], key.device)
                selected_probabilities = probabilities * selector[:, None, None, :]
                contribution = _torch().einsum(
                    "bphk,bhkd->bphd",
                    selected_probabilities,
                    repeated_value.float(),
                )
                norm, cosine = projected_contribution_metrics(
                    contribution, manual_total, module.o_proj.weight
                )
                span_results[span] = {
                    "attention_mass": selected_probabilities.sum(dim=-1).detach(),
                    "o_proj_contribution_norm": norm.detach(),
                    "o_proj_contribution_cosine_to_total_head": cosine.detach(),
                }
            scope_results[scope] = span_results
        self.observations = {
            "scopes": scope_results,
            "manual_selected_row_native_max_abs_error": maximum_manual_error,
            "manual_selected_row_native_low_precision_ratio": (
                maximum_manual_low_precision_ratio
            ),
            "manual_reconstruction_dtype": "float32",
            "query_heads": int(query.shape[1]),
            "kv_heads": int(key.shape[1]),
            "gqa_heads_per_kv": int(query.shape[1] // key.shape[1]),
        }
        self.fire_counts["interface"] += 1
        return baseline_output, baseline_weights


def projected_contribution_metrics(
    contribution: Any,
    total: Any,
    o_proj_weight: Any,
) -> tuple[Any, Any]:
    """Project each head through its true o_proj columns and return norm/cosine."""

    torch = _torch()
    if contribution.shape != total.shape or contribution.ndim != 4:
        raise ValueError("head contribution/total arrays must align [B,P,H,D]")
    heads, dimension = int(contribution.shape[2]), int(contribution.shape[3])
    if o_proj_weight.ndim != 2 or o_proj_weight.shape[1] != heads * dimension:
        raise ValueError("o_proj weight does not align with head dimensions")
    norms = []
    cosines = []
    for head in range(heads):
        columns = o_proj_weight[
            :, head * dimension : (head + 1) * dimension
        ].float()
        projected = torch.einsum(
            "bpd,od->bpo", contribution[:, :, head].float(), columns
        )
        projected_total = torch.einsum(
            "bpd,od->bpo", total[:, :, head].float(), columns
        )
        norm = projected.norm(dim=-1)
        denominator = norm * projected_total.norm(dim=-1)
        dot = (projected * projected_total).sum(dim=-1)
        cosine = torch.where(denominator > 0, dot / denominator, torch.zeros_like(dot))
        norms.append(norm)
        cosines.append(cosine.clamp(-1.0, 1.0))
    return torch.stack(norms, dim=-1), torch.stack(cosines, dim=-1)


def _select_positions(tensor: Any, positions: Any) -> Any:
    if tensor.ndim < 3 or positions.ndim != 2 or tensor.shape[0] != positions.shape[0]:
        raise ValueError("selected-position tensor alignment differs")
    positions = positions.to(tensor.device)
    rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
    return tensor[rows, positions]


def _span_selector(starts: Any, ends: Any, length: int, device: Any) -> Any:
    keys = _torch().arange(int(length), device=device)[None, :]
    return (keys >= starts.to(device)[:, None]) & (keys < ends.to(device)[:, None])


def _torch() -> Any:
    import torch
    return torch
