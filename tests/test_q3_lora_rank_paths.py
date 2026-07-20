from __future__ import annotations

import torch

from myrec.mechanism.q3_lora_rank_runtime import N10_RANK_MANIFEST_PATH, _load_rank_manifest
from myrec.mechanism.q3_lora_rank_scoring import LORA_PATH_CONDITIONS
from myrec.mechanism.q3_lora_rank_scoring import Q3LoraFactorPatch


class _Projection(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(4, 8, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(8, 4, bias=False)})


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.q_proj = _Projection()
        self.self_attn.v_proj = _Projection()


class _Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer() for _ in range(28)])


def test_rank_patch_keeps_exactly_one_outer_product_and_restores() -> None:
    model = _Model()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    with Q3LoraFactorPatch(model, "outer_product_rank_3"):
        for name, parameter in model.named_parameters():
            if ".lora_A." in name:
                assert torch.count_nonzero(parameter[:3]).item() == 0
                assert torch.count_nonzero(parameter[4:]).item() == 0
            elif ".lora_B." in name:
                assert torch.count_nonzero(parameter[:, :3]).item() == 0
                assert torch.count_nonzero(parameter[:, 4:]).item() == 0
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, before[name])


def test_a_only_and_b_only_controls_zero_effective_factor() -> None:
    model = _Model()
    with Q3LoraFactorPatch(model, "a_only"):
        assert all(
            torch.count_nonzero(parameter).item() == 0
            for name, parameter in model.named_parameters()
            if ".lora_B." in name
        )
    with Q3LoraFactorPatch(model, "b_only"):
        assert all(
            torch.count_nonzero(parameter).item() == 0
            for name, parameter in model.named_parameters()
            if ".lora_A." in name
        )


def test_n10_rank_manifest_and_condition_order_are_frozen() -> None:
    manifest = _load_rank_manifest(N10_RANK_MANIFEST_PATH)
    assert tuple(manifest["conditions"]) == LORA_PATH_CONDITIONS
    assert manifest["scope"]["lora_rank"] == 8
    assert manifest["factor_definition"]["report_all_rank_groups"] is True
