"""Project-owned Qwen3 node capture and patch primitives.

The first mechanism stage patched only mixed post-block hidden states.  This
module exposes branch-local nodes without changing the frozen ranker or the
Transformers implementation.  It deliberately contains no data loading,
qrels access, metric computation, or outcome-dependent node selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


BLOCK_NODE_IDS = (
    "block_input_residual",
    "input_rmsnorm_output",
    "q_pre_norm",
    "k_pre_norm",
    "q_post_norm_pre_rope",
    "k_post_norm_pre_rope",
    "v_projection",
    "attention_head_output_pre_o",
    "attention_o_projection",
    "post_attention_residual",
    "post_attention_rmsnorm_output",
    "mlp_gate_projection",
    "mlp_up_projection",
    "mlp_swiglu_product",
    "mlp_down_projection",
    "block_output_residual",
)
FINAL_NODE_IDS = (
    "final_rmsnorm_input",
    "final_rmsnorm_output",
)
SUPPORTED_NODE_IDS = (*BLOCK_NODE_IDS, *FINAL_NODE_IDS)


@dataclass(frozen=True, order=True)
class NodeSpec:
    """One registered Transformer node.

    ``block`` is zero-based for block nodes and must be ``None`` for final
    RMSNorm nodes.
    """

    node_id: str
    block: int | None

    def __post_init__(self) -> None:
        if self.node_id not in SUPPORTED_NODE_IDS:
            raise ValueError(f"unsupported node_id={self.node_id!r}")
        if self.node_id in FINAL_NODE_IDS:
            if self.block is not None:
                raise ValueError("final RMSNorm nodes require block=None")
        elif self.block is None or not 0 <= int(self.block) < 28:
            raise ValueError("block node requires a zero-based block in [0, 27]")

    @property
    def key(self) -> str:
        return (
            self.node_id
            if self.block is None
            else f"block_{int(self.block):02d}.{self.node_id}"
        )


@dataclass(frozen=True)
class QwenBackbone:
    layers: Any
    final_norm: Any
    owner_name: str


def resolve_qwen_backbone(model: Any) -> QwenBackbone:
    """Resolve the unique 28-layer Qwen backbone through optional PEFT wrappers."""

    import torch

    candidates: list[QwenBackbone] = []
    seen: set[int] = set()
    for name, module in model.named_modules():
        layers = getattr(module, "layers", None)
        final_norm = getattr(module, "norm", None)
        if (
            isinstance(layers, torch.nn.ModuleList)
            and len(layers) == 28
            and isinstance(final_norm, torch.nn.Module)
            and id(layers) not in seen
        ):
            seen.add(id(layers))
            candidates.append(
                QwenBackbone(
                    layers=layers,
                    final_norm=final_norm,
                    owner_name=name,
                )
            )
    if len(candidates) != 1:
        raise TypeError(
            f"expected one 28-block Qwen backbone with final norm, observed "
            f"{len(candidates)}"
        )
    return candidates[0]


def canonical_deep_dive_specs(blocks: Sequence[int]) -> tuple[NodeSpec, ...]:
    """Return every hookable registered node for fixed blocks plus final norm."""

    normalized = tuple(int(block) for block in blocks)
    if len(set(normalized)) != len(normalized):
        raise ValueError("deep-dive blocks must be unique")
    specs = [NodeSpec(node_id=node_id, block=block) for block in normalized for node_id in BLOCK_NODE_IDS]
    specs.extend(NodeSpec(node_id=node_id, block=None) for node_id in FINAL_NODE_IDS)
    return tuple(specs)


class QwenNodeCapture:
    """Capture selected token rows from registered branch-local Qwen nodes."""

    def __init__(self, model: Any, specs: Iterable[NodeSpec]) -> None:
        self.model = model
        self.backbone = resolve_qwen_backbone(model)
        self.specs = tuple(specs)
        if not self.specs or len({spec.key for spec in self.specs}) != len(self.specs):
            raise ValueError("capture specs must be nonempty and unique")
        self._positions: Any = None
        self._rows: Any = None
        self._captured: dict[str, Any] = {}
        self._fire_counts: dict[str, int] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "QwenNodeCapture":
        if self._handles:
            raise RuntimeError("node capture is already active")
        for spec in self.specs:
            module, hook_kind = _resolve_node_module(self.backbone, spec)
            if hook_kind == "input":
                handle = module.register_forward_pre_hook(self._input_hook(spec))
            else:
                handle = module.register_forward_hook(self._output_hook(spec))
            self._handles.append(handle)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._positions = None
        self._rows = None
        self._captured.clear()
        self._fire_counts.clear()

    def arm(self, positions: Any, *, sequence_length: int) -> None:
        if self._positions is not None:
            raise RuntimeError("node capture is already armed")
        if positions.ndim != 2 or positions.shape[1] <= 0:
            raise ValueError("capture positions must have shape [batch, positive_count]")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("capture sequence_length must be positive")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= sequence_length:
            raise ValueError("capture position is outside the registered sequence")
        self._positions = positions
        self._rows = _torch().arange(positions.shape[0], device=positions.device)[:, None]
        self._captured = {}
        self._fire_counts = {spec.key: 0 for spec in self.specs}

    def disarm(self) -> dict[str, Any]:
        if self._positions is None:
            raise RuntimeError("node capture is not armed")
        missing = [key for key, count in self._fire_counts.items() if count != 1]
        if missing:
            details = {key: self._fire_counts[key] for key in missing}
            raise RuntimeError(f"capture hook fire-count mismatch: {details}")
        result = dict(self._captured)
        self._positions = None
        self._rows = None
        self._captured = {}
        self._fire_counts = {}
        return result


    def capture_forward(
        self,
        *,
        input_ids: Any,
        attention_mask: Any,
        positions: Any,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        positions = positions.to(input_ids.device)
        self.arm(positions, sequence_length=int(input_ids.shape[1]))
        try:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                **dict(model_kwargs or {}),
            )
            captured = self.disarm()
        except Exception:
            self._positions = None
            self._rows = None
            self._captured = {}
            self._fire_counts = {}
            raise
        return output, captured

    def _input_hook(self, spec: NodeSpec):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs:
                raise RuntimeError(f"{spec.key} received no positional input")
            self._store(spec, inputs[0])

        return hook

    def _output_hook(self, spec: NodeSpec):
        def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            self._store(spec, _tensor_output(output, spec.key))

        return hook

    def _store(self, spec: NodeSpec, tensor: Any) -> None:
        if self._positions is None:
            raise RuntimeError(f"capture hook {spec.key} fired while unarmed")
        if tensor.ndim < 3:
            raise ValueError(f"node {spec.key} is not token-indexed rank >=3")
        positions = self._positions
        rows = self._rows
        if positions.device != tensor.device or rows is None or rows.device != tensor.device:
            raise ValueError(f"node {spec.key} device differs from armed positions")
        if tensor.shape[0] != positions.shape[0]:
            raise ValueError(f"node {spec.key} batch differs from capture positions")
        selected = tensor[rows, positions].detach()
        if spec.key in self._captured:
            raise RuntimeError(f"capture node {spec.key} fired more than once")
        self._captured[spec.key] = selected
        self._fire_counts[spec.key] += 1


class QwenNodeCallAudit:
    """Record phase-aware decoder call shapes without changing computation."""

    def __init__(self, model: Any) -> None:
        self.backbone = resolve_qwen_backbone(model)
        self.handles: list[Any] = []
        self.calls: dict[int, list[tuple[int, ...]]] = {}

    def __enter__(self) -> "QwenNodeCallAudit":
        if self.handles:
            raise RuntimeError("node call audit is already active")
        self.calls = {block: [] for block in range(28)}
        for block, layer in enumerate(self.backbone.layers):
            self.handles.append(
                layer.register_forward_pre_hook(self._hook(block))
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def result(self) -> dict[str, Any]:
        counts = {block: len(shapes) for block, shapes in self.calls.items()}
        if not counts or len(set(counts.values())) != 1:
            raise RuntimeError(f"decoder call-count mismatch: {counts}")
        reference = self.calls[0]
        if any(self.calls[block] != reference for block in range(1, 28)):
            raise RuntimeError("decoder call shapes differ across blocks")
        return {
            "calls_per_block": next(iter(counts.values())),
            "all_blocks_identical": True,
            "block_0_input_shapes": [list(shape) for shape in reference],
        }

    def _hook(self, block: int):
        def hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if not inputs or not hasattr(inputs[0], "shape"):
                raise RuntimeError(f"decoder block {block} received no tensor input")
            self.calls[block].append(tuple(int(value) for value in inputs[0].shape))

        return hook


class QwenNodePatch:
    """Patch one node at explicit token positions for one model forward."""

    def __init__(self, model: Any, spec: NodeSpec) -> None:
        self.model = model
        self.backbone = resolve_qwen_backbone(model)
        self.spec = spec
        self.positions: Any = None
        self.vectors: Any = None
        self.fire_count = 0
        self.handle: Any = None
        self.hook_kind: str | None = None

    def __enter__(self) -> "QwenNodePatch":
        if self.handle is not None:
            raise RuntimeError("node patch is already active")
        if self.spec.node_id == "post_attention_residual":
            raise ValueError(
                "post_attention_residual is capture-only in QwenNodePatch; use "
                "QwenPostAttentionStatePatch for composition-safe replacement"
            )
        module, self.hook_kind = _resolve_node_module(self.backbone, self.spec)
        if self.hook_kind == "input":
            self.handle = module.register_forward_pre_hook(self._input_hook)
        else:
            self.handle = module.register_forward_hook(self._output_hook)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.handle is not None:
            self.handle.remove()
        self.handle = None
        self.hook_kind = None
        self.positions = None
        self.vectors = None
        self.fire_count = 0

    def arm(self, positions: Any, vectors: Any, *, sequence_length: int) -> None:
        if self.positions is not None:
            raise RuntimeError("node patch is already armed")
        if positions.ndim != 2 or vectors.ndim < 3:
            raise ValueError("patch expects positions [batch,count] and vectors [batch,count,...]")
        if tuple(vectors.shape[:2]) != tuple(positions.shape):
            raise ValueError("patch vectors do not align with positions")
        sequence_length = int(sequence_length)
        if sequence_length <= 0:
            raise ValueError("patch sequence_length must be positive")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= sequence_length:
            raise ValueError("patch position is outside the registered sequence")
        self.positions = positions
        self.vectors = vectors
        self.fire_count = 0

    def disarm(self) -> None:
        if self.positions is None:
            raise RuntimeError("node patch is not armed")
        if self.fire_count != 1:
            raise RuntimeError(f"node patch hook fired {self.fire_count} times")
        self.positions = None
        self.vectors = None
        self.fire_count = 0

    def _input_hook(self, _module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        if not inputs:
            raise RuntimeError(f"{self.spec.key} received no positional input")
        modified = self._replace(inputs[0])
        return (modified, *inputs[1:])

    def _output_hook(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> Any:
        tensor = _tensor_output(output, self.spec.key)
        modified = self._replace(tensor)
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    def _replace(self, tensor: Any) -> Any:
        if self.positions is None or self.vectors is None:
            raise RuntimeError(f"patch hook {self.spec.key} fired while unarmed")
        if tensor.ndim < 3 or tensor.shape[0] != self.positions.shape[0]:
            raise ValueError(f"patch tensor shape mismatch for {self.spec.key}")
        positions = self.positions
        if positions.device != tensor.device:
            raise ValueError(f"patch positions for {self.spec.key} are on the wrong device")
        replacement = self.vectors.to(device=tensor.device, dtype=tensor.dtype)
        expected = (tensor.shape[0], positions.shape[1], *tensor.shape[2:])
        if tuple(replacement.shape) != expected:
            raise ValueError(
                f"patch vector shape for {self.spec.key} is {tuple(replacement.shape)}, "
                f"expected {expected}"
            )
        modified = tensor.clone()
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        modified[rows, positions] = replacement
        self.fire_count += 1
        return modified


class QwenPostAttentionStatePatch:
    """Replace ``u = block_input + attention_increment`` exactly.

    Qwen's decoder stores ``u`` in a Python local before the MLP residual add,
    so modifying only the input to ``post_attention_layernorm`` does not replace
    the residual state. This intervention writes the attention boundary,
    supplies ``desired_u`` to the MLP norm, and reconstructs the block output as
    ``desired_u + mlp(desired_u)``. The paired boundary hooks also remove BF16
    subtraction/addition drift from the intended state intervention.
    """

    def __init__(self, model: Any, block: int) -> None:
        block = int(block)
        if not 0 <= block < 28:
            raise ValueError("post-attention patch block must be in [0, 27]")
        self.backbone = resolve_qwen_backbone(model)
        self.layer = self.backbone.layers[block]
        self.positions: Any = None
        self.desired_state: Any = None
        self.recipient_input: Any = None
        self.mlp_increment: Any = None
        self.input_fire_count = 0
        self.attention_fire_count = 0
        self.norm_fire_count = 0
        self.mlp_fire_count = 0
        self.output_fire_count = 0
        self.handles: list[Any] = []

    def __enter__(self) -> "QwenPostAttentionStatePatch":
        if self.handles:
            raise RuntimeError("post-attention state patch is already active")
        self.handles = [
            self.layer.register_forward_pre_hook(self._block_input_hook),
            self.layer.self_attn.register_forward_hook(self._attention_output_hook),
            self.layer.post_attention_layernorm.register_forward_pre_hook(
                self._post_attention_norm_input_hook
            ),
            self.layer.mlp.down_proj.register_forward_hook(self._mlp_output_hook),
            self.layer.register_forward_hook(self._block_output_hook),
        ]
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.positions = None
        self.desired_state = None
        self.recipient_input = None
        self.mlp_increment = None
        self.input_fire_count = 0
        self.attention_fire_count = 0
        self.norm_fire_count = 0
        self.mlp_fire_count = 0
        self.output_fire_count = 0

    def arm(self, positions: Any, desired_state: Any, *, sequence_length: int) -> None:
        if self.positions is not None:
            raise RuntimeError("post-attention state patch is already armed")
        if positions.ndim != 2 or desired_state.ndim != 3:
            raise ValueError("post-attention patch shapes must be [batch,count] and [batch,count,hidden]")
        if tuple(desired_state.shape[:2]) != tuple(positions.shape):
            raise ValueError("post-attention desired state does not align with positions")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= int(sequence_length):
            raise ValueError("post-attention patch position is outside sequence")
        self.positions = positions
        self.desired_state = desired_state
        self.recipient_input = None
        self.mlp_increment = None
        self.input_fire_count = 0
        self.attention_fire_count = 0
        self.norm_fire_count = 0
        self.mlp_fire_count = 0
        self.output_fire_count = 0

    def disarm(self) -> None:
        counts = (
            self.input_fire_count,
            self.attention_fire_count,
            self.norm_fire_count,
            self.mlp_fire_count,
            self.output_fire_count,
        )
        if counts != (1, 1, 1, 1, 1):
            raise RuntimeError(
                "post-attention patch fire-count mismatch: "
                f"input,attention,norm,mlp,output={counts}"
            )
        self.positions = None
        self.desired_state = None
        self.recipient_input = None
        self.mlp_increment = None
        self.input_fire_count = 0
        self.attention_fire_count = 0
        self.norm_fire_count = 0
        self.mlp_fire_count = 0
        self.output_fire_count = 0

    def _block_input_hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self.positions is None or self.desired_state is None:
            raise RuntimeError("post-attention patch fired while unarmed")
        if not inputs:
            raise RuntimeError("patched block received no hidden-state input")
        hidden = inputs[0]
        positions = self.positions
        if positions.device != hidden.device or hidden.shape[0] != positions.shape[0]:
            raise ValueError("post-attention block input does not align with positions")
        rows = _torch().arange(hidden.shape[0], device=hidden.device)[:, None]
        self.recipient_input = hidden[rows, positions].detach()
        self.input_fire_count += 1

    def _attention_output_hook(
        self, _module: Any, _inputs: tuple[Any, ...], output: Any
    ) -> Any:
        if self.recipient_input is None or self.desired_state is None or self.positions is None:
            raise RuntimeError("attention output fired before recipient block input")
        tensor = _tensor_output(output, "post_attention_residual")
        positions = self.positions
        desired = self.desired_state.to(device=tensor.device, dtype=tensor.dtype)
        recipient = self.recipient_input.to(device=tensor.device, dtype=tensor.dtype)
        if desired.shape != recipient.shape or desired.shape[-1] != tensor.shape[-1]:
            raise ValueError("post-attention donor/recipient hidden shape mismatch")
        modified = tensor.clone()
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        modified[rows, positions] = desired - recipient
        self.attention_fire_count += 1
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified

    def _post_attention_norm_input_hook(
        self, _module: Any, inputs: tuple[Any, ...]
    ) -> tuple[Any, ...]:
        if self.positions is None or self.desired_state is None or not inputs:
            raise RuntimeError("post-attention norm input fired while unarmed")
        tensor = inputs[0]
        desired = self.desired_state.to(device=tensor.device, dtype=tensor.dtype)
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        modified = tensor.clone()
        modified[rows, self.positions] = desired
        self.norm_fire_count += 1
        return (modified, *inputs[1:])

    def _mlp_output_hook(
        self, _module: Any, _inputs: tuple[Any, ...], output: Any
    ) -> Any:
        if self.positions is None:
            raise RuntimeError("post-attention MLP output fired while unarmed")
        tensor = _tensor_output(output, "post_attention_mlp_increment")
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        self.mlp_increment = tensor[rows, self.positions].detach()
        self.mlp_fire_count += 1
        return output

    def _block_output_hook(
        self, _module: Any, _inputs: tuple[Any, ...], output: Any
    ) -> Any:
        if (
            self.positions is None
            or self.desired_state is None
            or self.mlp_increment is None
        ):
            raise RuntimeError("post-attention block output fired before MLP output")
        tensor = _tensor_output(output, "post_attention_block_output")
        desired = self.desired_state.to(device=tensor.device, dtype=tensor.dtype)
        mlp = self.mlp_increment.to(device=tensor.device, dtype=tensor.dtype)
        target = desired + mlp
        rows = _torch().arange(tensor.shape[0], device=tensor.device)[:, None]
        modified = tensor.clone()
        modified[rows, self.positions] = target
        self.output_fire_count += 1
        if isinstance(output, tuple):
            return (modified, *output[1:])
        return modified


def rms_matched_random_direction(
    recipient: Any,
    *,
    seed: int,
    identity_keys: Sequence[Sequence[str]],
    reduce_dims: Sequence[int] | None = None,
) -> Any:
    """Return identity-stable random directions with matched RMS.

    Randomness is bound to each ``[batch, position]`` identity so batching,
    sharding, and resume boundaries cannot change the control.
    """

    import hashlib
    torch = _torch()
    if recipient.ndim < 3 or not recipient.is_floating_point():
        raise ValueError("RMS-matched control requires a floating tensor")
    if len(identity_keys) != recipient.shape[0] or any(
        len(row) != recipient.shape[1] for row in identity_keys
    ):
        raise ValueError("identity_keys must align with recipient [batch, position]")
    random = torch.empty_like(recipient, dtype=torch.float32, device="cpu")
    trailing = tuple(recipient.shape[2:])
    for batch, row in enumerate(identity_keys):
        for position, identity in enumerate(row):
            digest = hashlib.sha256(
                f"{int(seed)}\0{identity}".encode("utf-8")
            ).digest()
            generator = torch.Generator(device="cpu")
            generator.manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
            random[batch, position] = torch.randn(
                trailing,
                generator=generator,
                dtype=torch.float32,
                device="cpu",
            )
    random = random.to(recipient.device)
    dims = tuple(int(value) for value in (reduce_dims or range(2, recipient.ndim)))
    if not dims or any(value < 2 or value >= recipient.ndim for value in dims):
        raise ValueError("reduce_dims must name one or more trailing node dimensions")
    target_rms = recipient.float().pow(2).mean(dim=dims, keepdim=True).sqrt()
    random_rms = random.pow(2).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-12)
    return (random * (target_rms / random_rms)).to(recipient.dtype)


def _resolve_node_module(backbone: QwenBackbone, spec: NodeSpec) -> tuple[Any, str]:
    if spec.node_id == "final_rmsnorm_input":
        return backbone.final_norm, "input"
    if spec.node_id == "final_rmsnorm_output":
        return backbone.final_norm, "output"
    assert spec.block is not None
    layer = backbone.layers[int(spec.block)]
    mapping = {
        "block_input_residual": (layer, "input"),
        "input_rmsnorm_output": (layer.input_layernorm, "output"),
        "q_pre_norm": (layer.self_attn.q_proj, "output"),
        "k_pre_norm": (layer.self_attn.k_proj, "output"),
        "q_post_norm_pre_rope": (layer.self_attn.q_norm, "output"),
        "k_post_norm_pre_rope": (layer.self_attn.k_norm, "output"),
        "v_projection": (layer.self_attn.v_proj, "output"),
        "attention_head_output_pre_o": (layer.self_attn.o_proj, "input"),
        "attention_o_projection": (layer.self_attn.o_proj, "output"),
        "post_attention_residual": (layer.post_attention_layernorm, "input"),
        "post_attention_rmsnorm_output": (layer.post_attention_layernorm, "output"),
        "mlp_gate_projection": (layer.mlp.gate_proj, "output"),
        "mlp_up_projection": (layer.mlp.up_proj, "output"),
        "mlp_swiglu_product": (layer.mlp.down_proj, "input"),
        "mlp_down_projection": (layer.mlp.down_proj, "output"),
        "block_output_residual": (layer, "output"),
    }
    try:
        return mapping[spec.node_id]
    except KeyError as exc:
        raise ValueError(f"node has no hook mapping: {spec.node_id}") from exc


def _tensor_output(output: Any, node_key: str) -> Any:
    tensor = output[0] if isinstance(output, tuple) else output
    if not hasattr(tensor, "ndim"):
        raise TypeError(f"node {node_key} did not produce a tensor")
    return tensor


def _torch() -> Any:
    import torch

    return torch
