from __future__ import annotations

import torch
from torch import nn

from myrec.mechanism.q3_lora_branch_interventions import QwenQ3LoraBranchPatch


class _FakeLoraLinear(nn.Module):
    def __init__(self, hidden: int = 8, rank: int = 2) -> None:
        super().__init__()
        self.base_layer = nn.Linear(hidden, hidden, bias=False)
        self.lora_A = nn.ModuleDict({"default": nn.Linear(hidden, rank, bias=False)})
        self.lora_B = nn.ModuleDict({"default": nn.Linear(rank, hidden, bias=False)})
        self.lora_dropout = nn.ModuleDict({"default": nn.Identity()})
        self.scaling = {"default": 2.0}
        self.active_adapters = ["default"]
        self.merged = False
        self.disable_adapters = False

    def forward(self, x):
        return self.base_layer(x) + self.lora_B["default"](self.lora_A["default"](x)) * 2.0


class _Layer:
    def __init__(self, component: str) -> None:
        self.self_attn = type("Attention", (), {})()
        setattr(self.self_attn, f"{component}_proj", _FakeLoraLinear())


def _patch_with_fake(component: str, mode: str):
    # Bypass backbone resolution only for this unit test; the hook itself is
    # exercised through a real PEFT-shaped module and not through a score path.
    patch = object.__new__(QwenQ3LoraBranchPatch)
    patch.block = 13
    patch.component = component
    patch.mode = mode
    patch.module = getattr(_Layer(component).self_attn, f"{component}_proj")
    patch.positions = torch.tensor([[3], [3]])
    patch.history_starts = torch.tensor([0, 0])
    patch.history_ends = torch.tensor([2, 2])
    patch.sequence_length = 4
    patch.fire_count = 0
    patch.adapter_name = "default"
    patch.last_summary = {}
    patch.handle = None
    return patch


def test_q3_lora_identity_hook_is_exact():
    patch = _patch_with_fake("q", "identity")
    module = patch.module
    x = torch.randn(2, 4, 8)
    native = module(x)
    patched = patch._hook(module, (x,), native)
    torch.testing.assert_close(patched, native, rtol=0.0, atol=0.0)
    assert patch.last_summary["maximum_applied_delta"] == 0.0


def test_q3_lora_zero_and_random_branch_are_finite():
    x = torch.randn(2, 4, 8)
    for mode in ("zero", "output_norm_matched_random"):
        patch = _patch_with_fake("v", mode)
        native = patch.module(x)
        patched = patch._hook(patch.module, (x,), native)
        assert torch.isfinite(patched).all()
        assert patch.last_summary["maximum_applied_delta"] > 0.0

