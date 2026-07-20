"""Token-embedding lookup interventions for the registered N30 boundary."""

from __future__ import annotations

from typing import Any, Sequence

from myrec.mechanism.transformer_instrumentation import (
    _tensor_output,
    _torch,
    resolve_qwen_backbone,
    rms_matched_random_direction,
)


EMBEDDING_SCOPES = ("query", "history", "candidate")
EMBEDDING_MODES = (
    "identity",
    "zero",
    "scale_half",
    "scale_double",
    "sign_flip",
    "output_norm_matched_random",
)


def apply_embedding_operator(
    embedding: Any,
    mode: str,
    *,
    identity_keys: Sequence[Sequence[str]] | None = None,
    random_seed: int = 20260720,
) -> Any:
    """Transform selected embedding rows with identity-bound controls."""

    torch = _torch()
    if mode not in EMBEDDING_MODES:
        raise ValueError(f"unsupported embedding mode={mode}")
    if embedding.ndim < 3 or not embedding.is_floating_point():
        raise ValueError("embedding rows must be a floating rank-three tensor")
    if not torch.isfinite(embedding).all():
        raise ValueError("embedding rows contain non-finite values")
    if mode == "identity":
        return embedding
    if mode == "zero":
        return torch.zeros_like(embedding)
    if mode == "scale_half":
        return embedding * 0.5
    if mode == "scale_double":
        return embedding * 2.0
    if mode == "sign_flip":
        return -embedding
    if identity_keys is None:
        raise ValueError("random embedding mode requires identity keys")
    return rms_matched_random_direction(
        embedding,
        seed=int(random_seed),
        identity_keys=identity_keys,
    )


def _resolve_embedding_module(model: Any) -> Any:
    """Resolve the single Qwen ``embed_tokens`` module through PEFT wrappers."""

    backbone = resolve_qwen_backbone(model)
    owners: list[Any] = []
    if backbone.owner_name:
        try:
            owner = model.get_submodule(backbone.owner_name)
            module = getattr(owner, "embed_tokens", None)
            if module is not None:
                owners.append(module)
        except (AttributeError, KeyError):
            pass
    for _name, module in model.named_modules():
        if module.__class__.__name__.lower().endswith("embedding") and module not in owners:
            # Avoid selecting an adapter's auxiliary embedding table.  Qwen's
            # input table is the one whose output dimension matches the block
            # hidden size.
            if hasattr(module, "embedding_dim"):
                owners.append(module)
    unique = {id(module): module for module in owners}
    if len(unique) != 1:
        raise TypeError(f"expected one Qwen input embedding module, observed {len(unique)}")
    return next(iter(unique.values()))


class QwenEmbeddingOperatorPatch:
    """Patch selected token rows after the native embedding lookup."""

    def __init__(
        self,
        model: Any,
        scope: str,
        mode: str,
        *,
        random_seed: int = 20260720,
    ) -> None:
        if scope not in EMBEDDING_SCOPES:
            raise ValueError(f"unsupported embedding scope={scope}")
        if mode not in EMBEDDING_MODES:
            raise ValueError(f"unsupported embedding mode={mode}")
        self.module = _resolve_embedding_module(model)
        self.scope = scope
        self.mode = mode
        self.random_seed = int(random_seed)
        self.positions: Any = None
        self.identity_keys: Sequence[Sequence[str]] | None = None
        self.expected_input_ids: Any = None
        self.sequence_length: int | None = None
        self.fire_count = 0
        self.last_summary: dict[str, Any] = {}
        self.handle: Any = None

    def __enter__(self) -> "QwenEmbeddingOperatorPatch":
        if self.handle is not None:
            raise RuntimeError("embedding operator patch is already active")
        self.handle = self.module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self._clear()

    def arm(
        self,
        positions: Any,
        *,
        identity_keys: Sequence[Sequence[str]],
        input_ids: Any,
        sequence_length: int,
    ) -> None:
        if self.handle is None or self.positions is not None:
            raise RuntimeError("embedding operator patch cannot be armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("embedding positions must have shape [batch, positive_count]")
        if input_ids.ndim != 2 or tuple(input_ids.shape) != tuple(
            (positions.shape[0], int(sequence_length))
        ):
            raise ValueError("embedding input_ids and sequence geometry do not align")
        if len(identity_keys) != int(positions.shape[0]) or any(
            len(row) != int(positions.shape[1]) for row in identity_keys
        ):
            raise ValueError("embedding identity keys do not align with positions")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("embedding sequence length must be positive")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= sequence_length:
            raise ValueError("embedding position is outside sequence")
        self.positions = positions
        self.identity_keys = identity_keys
        self.expected_input_ids = input_ids.detach().clone()
        self.sequence_length = sequence_length
        self.fire_count = 0
        self.last_summary = {}

    def disarm(self) -> dict[str, Any]:
        if self.positions is None or self.fire_count != 1:
            raise RuntimeError(
                f"embedding lookup fired {self.fire_count} times; expected one"
            )
        summary = dict(self.last_summary)
        self._clear()
        return summary

    def _clear(self) -> None:
        self.positions = None
        self.identity_keys = None
        self.expected_input_ids = None
        self.sequence_length = None
        self.fire_count = 0
        self.last_summary = {}

    def _hook(self, _module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
        if self.positions is None or self.identity_keys is None:
            raise RuntimeError("embedding hook fired while unarmed")
        if not inputs or self.expected_input_ids is None:
            raise RuntimeError("embedding hook received no input_ids")
        input_ids = inputs[0]
        if not _torch().equal(input_ids.detach(), self.expected_input_ids):
            raise RuntimeError("token IDs changed while embedding intervention was armed")
        native = _tensor_output(output, "token_embedding_lookup")
        if native.ndim != 3 or native.shape[:2] != input_ids.shape:
            raise ValueError("embedding output/input shape mismatch")
        rows = _torch().arange(native.shape[0], device=native.device)[:, None]
        if self.positions.device != native.device:
            raise ValueError("embedding positions and output use different devices")
        selected_native = native[rows, self.positions]
        selected = apply_embedding_operator(
            selected_native,
            self.mode,
            identity_keys=self.identity_keys,
            random_seed=self.random_seed,
        )
        modified = native.clone()
        modified[rows, self.positions] = selected.to(dtype=native.dtype)
        self.fire_count += 1
        self.last_summary = {
            "scope": self.scope,
            "mode": self.mode,
            "rows": int(native.shape[0]),
            "positions_per_row": int(self.positions.shape[1]),
            "maximum_output_delta": float(
                _torch().max(_torch().abs(selected.float() - selected_native.float())).item()
            ),
            "token_ids_unchanged": True,
        }
        return modified
