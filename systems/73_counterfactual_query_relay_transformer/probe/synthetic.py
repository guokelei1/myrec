"""Frozen C73 two-hop associative-ranking generator and perturbations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Any

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SyntheticData:
    query_tokens: torch.Tensor
    history_tokens: torch.Tensor
    candidate_tokens: torch.Tensor
    history_mask: torch.Tensor
    candidate_mask: torch.Tensor
    base_scores: torch.Tensor
    item_only_scores: torch.Tensor
    repeat_request: torch.Tensor
    query_present: torch.Tensor
    labels: torch.Tensor
    supported_request: torch.Tensor
    no_history_request: torch.Tensor

    def __len__(self) -> int:
        return int(self.query_tokens.shape[0])

    def to(self, device: torch.device) -> "SyntheticData":
        return SyntheticData(
            **{
                name: value.to(device)
                for name, value in self.__dict__.items()
            }
        )

    def index(self, values: torch.Tensor) -> "SyntheticData":
        return SyntheticData(
            **{
                name: value[values]
                for name, value in self.__dict__.items()
            }
        )

    def forward_kwargs(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in (
                "query_tokens",
                "history_tokens",
                "candidate_tokens",
                "history_mask",
                "candidate_mask",
                "base_scores",
                "item_only_scores",
                "repeat_request",
                "query_present",
            )
        }


def _unit(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value, dim=-1, eps=1e-6)


def make_dataset(
    config: Mapping[str, Any], *, examples: int, seed: int, split: str
) -> SyntheticData:
    if split not in {"train", "validation"}:
        raise ValueError("C73 synthetic split differs")
    row = config["data"]
    q_count = int(row["query_tokens"])
    h_count = int(row["history_events"])
    c_count = int(row["candidates"])
    dim = int(row["dimension"])
    key_dim = int(row["key_dimension"])
    value_dim = int(row["value_dimension"])
    if q_count != 4 or h_count != 8 or c_count != q_count * 3:
        raise ValueError("C73 frozen generator cardinality differs")
    if key_dim + value_dim + 4 > dim:
        raise ValueError("C73 frozen generator dimension differs")
    generator = torch.Generator().manual_seed(int(seed))
    noise = float(row["noise_std"])

    keys = _unit(torch.randn(examples, q_count, key_dim, generator=generator))
    target_facet = torch.randint(q_count, (examples,), generator=generator)
    query = torch.zeros(examples, q_count, dim)
    query[..., :key_dim] = keys
    query[..., key_dim + value_dim] = 0.15
    query[
        torch.arange(examples), target_facet, key_dim + value_dim
    ] = 1.0
    query += noise * torch.randn(query.shape, generator=generator)

    old_value = _unit(torch.randn(examples, value_dim, generator=generator))
    new_value = _unit(torch.randn(examples, value_dim, generator=generator))
    facet_values = _unit(
        torch.randn(examples, q_count, value_dim, generator=generator)
    )
    facet_values[torch.arange(examples), target_facet] = new_value

    candidates = torch.zeros(examples, c_count, dim)
    positive_slot = torch.empty(examples, dtype=torch.long)
    old_slot = torch.empty(examples, dtype=torch.long)
    for facet in range(q_count):
        start = 3 * facet
        candidates[:, start : start + 3, :key_dim] = keys[:, facet : facet + 1]
        values = _unit(torch.randn(examples, 3, value_dim, generator=generator))
        members = target_facet == facet
        values[members, 0] = new_value[members]
        values[members, 1] = old_value[members]
        candidates[:, start : start + 3, key_dim : key_dim + value_dim] = values
        candidates[:, start : start + 3, key_dim + value_dim + 1] = 1.0
    positive_slot = target_facet * 3
    old_slot = positive_slot + 1
    candidates += noise * torch.randn(candidates.shape, generator=generator)

    history = torch.zeros(examples, h_count, dim)
    history_mask = torch.ones(examples, h_count, dtype=torch.bool)
    row_index = torch.arange(examples)
    target_key = keys[row_index, target_facet]
    history[:, 1, :key_dim] = target_key
    history[:, 1, key_dim : key_dim + value_dim] = old_value
    history[:, 1, key_dim + value_dim + 2] = 1.0
    history[:, 6, :key_dim] = target_key
    history[:, 6, key_dim : key_dim + value_dim] = new_value
    history[:, 6, key_dim + value_dim + 2] = 1.0

    remaining = [0, 2, 3, 4]
    other_order = torch.argsort(
        torch.rand(examples, q_count, generator=generator), dim=-1
    )
    for position_index, position in enumerate(remaining):
        facet = other_order[:, position_index]
        history[:, position, :key_dim] = keys[row_index, facet]
        history[:, position, key_dim : key_dim + value_dim] = facet_values[
            row_index, facet
        ]
        history[:, position, key_dim + value_dim + 2] = 1.0

    off_query_key = _unit(torch.randn(examples, key_dim, generator=generator))
    history[:, 5, :key_dim] = off_query_key
    nuisance_slot = positive_slot if split == "train" else old_slot
    history[:, 5, key_dim : key_dim + value_dim] = candidates[
        row_index, nuisance_slot, key_dim : key_dim + value_dim
    ]
    history[:, 5, key_dim + value_dim + 3] = 1.0
    history[:, 7, :key_dim] = _unit(
        torch.randn(examples, key_dim, generator=generator)
    )
    history[:, 7, key_dim : key_dim + value_dim] = _unit(
        torch.randn(examples, value_dim, generator=generator)
    )
    history[:, 7, key_dim + value_dim + 3] = 1.0
    history += noise * torch.randn(history.shape, generator=generator)

    target_query = keys[row_index, target_facet]
    base = torch.einsum("bcd,bd->bc", candidates[..., :key_dim], target_query)
    base += 0.04 * torch.randn(base.shape, generator=generator)
    labels = torch.zeros(examples, c_count)
    labels[row_index, positive_slot] = 1.0
    item_only = base.clone()

    order = torch.rand(examples, generator=generator)
    no_history = order < float(row["no_history_fraction"])
    repeat = (order >= float(row["no_history_fraction"])) & (
        order
        < float(row["no_history_fraction"]) + float(row["repeat_fraction"])
    )
    supported = ~(no_history | repeat)

    if bool(no_history.any()):
        chosen = base[no_history].argmax(-1)
        labels[no_history] = 0.0
        labels[no_history, chosen] = 1.0
        history_mask[no_history] = False
        history[no_history] = 0.0
    if bool(repeat.any()):
        repeat_rows = row_index[repeat]
        chosen = positive_slot[repeat]
        item_only[repeat_rows, chosen] += 6.0
        labels[repeat] = 0.0
        labels[repeat_rows, chosen] = 1.0

    permutation = torch.argsort(
        torch.rand(examples, c_count, generator=generator), dim=-1
    )
    candidates = candidates.gather(
        1, permutation[..., None].expand(-1, -1, dim)
    )
    base = base.gather(1, permutation)
    item_only = item_only.gather(1, permutation)
    labels = labels.gather(1, permutation)

    return SyntheticData(
        query_tokens=query,
        history_tokens=history,
        candidate_tokens=candidates,
        history_mask=history_mask,
        candidate_mask=torch.ones(examples, c_count, dtype=torch.bool),
        base_scores=base,
        item_only_scores=item_only,
        repeat_request=repeat,
        query_present=torch.ones(examples, dtype=torch.bool),
        labels=labels,
        supported_request=supported,
        no_history_request=no_history,
    )


def wrong_history(data: SyntheticData) -> SyntheticData:
    eligible = torch.nonzero(data.history_mask.any(-1), as_tuple=False).flatten()
    donor = torch.arange(len(data))
    if len(eligible) > 1:
        donor[eligible] = eligible.roll(1)
    return replace(
        data,
        history_tokens=data.history_tokens[donor],
        history_mask=data.history_mask[donor],
    )


def shuffled_history(data: SyntheticData) -> SyntheticData:
    order = torch.arange(data.history_tokens.shape[1] - 1, -1, -1)
    return replace(
        data,
        history_tokens=data.history_tokens[:, order],
        history_mask=data.history_mask[:, order],
    )


def coarse_history(data: SyntheticData, config: Mapping[str, Any]) -> SyntheticData:
    key_dim = int(config["data"]["key_dimension"])
    value_dim = int(config["data"]["value_dimension"])
    history = data.history_tokens.clone()
    history[..., key_dim : key_dim + value_dim] = 0.0
    return replace(data, history_tokens=history)


def query_masked(data: SyntheticData) -> SyntheticData:
    return replace(data, query_present=torch.zeros_like(data.query_present))
