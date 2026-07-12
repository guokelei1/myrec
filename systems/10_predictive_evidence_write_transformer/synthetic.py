"""Frozen-shape synthetic task for the C10 pre-outcome gate."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class SyntheticSpec:
    categories: int = 4
    attributes: int = 5
    item_variants: int = 16
    history_events: int = 8
    candidates: int = 7
    query_tokens: int = 2
    item_tokens: int = 3
    exact_repeat_probability: float = 0.35

    @property
    def category_offset(self) -> int:
        return 4

    @property
    def attribute_offset(self) -> int:
        return self.category_offset + self.categories

    @property
    def item_offset(self) -> int:
        return self.attribute_offset + self.attributes

    @property
    def item_count(self) -> int:
        return self.categories * self.attributes * self.item_variants

    @property
    def vocab_size(self) -> int:
        return self.item_offset + self.item_count

    @property
    def query_marker_token(self) -> int:
        return 1

    @property
    def masked_query_token(self) -> int:
        return 2


@dataclass
class SyntheticBatch:
    query_tokens: Tensor
    candidate_tokens: Tensor
    history_tokens: Tensor
    history_mask: Tensor
    targets: Tensor
    exact_repeat: Tensor

    def to(self, device: torch.device | str) -> "SyntheticBatch":
        return SyntheticBatch(**{name: value.to(device) for name, value in self.__dict__.items()})


def _item_token(spec: SyntheticSpec, category: Tensor, attribute: Tensor, variant: Tensor) -> Tensor:
    index = (category * spec.attributes + attribute) * spec.item_variants + variant
    return spec.item_offset + index


def _product_tokens(spec: SyntheticSpec, category: Tensor, attribute: Tensor, variant: Tensor) -> Tensor:
    return torch.stack(
        (
            _item_token(spec, category, attribute, variant),
            spec.category_offset + category,
            spec.attribute_offset + attribute,
        ),
        dim=-1,
    )


def generate_batch(spec: SyntheticSpec, batch_size: int, generator: torch.Generator) -> SyntheticBatch:
    """Generate query-conditional user preferences without dataset-specific cases.

    Every user has a different preferred attribute for each category.  Histories
    interleave categories, and the final event is the reliable current signal for
    the query category.  The relevant candidate matches both query category and
    its user-specific attribute.  Candidate order is randomized.
    """

    categories = torch.randint(spec.categories, (batch_size,), generator=generator)
    preferences = torch.randint(spec.attributes, (batch_size, spec.categories), generator=generator)
    wanted_attribute = preferences.gather(1, categories[:, None]).squeeze(1)

    history_category = torch.randint(
        spec.categories, (batch_size, spec.history_events), generator=generator
    )
    history_category[:, -1] = categories
    preferred_by_event = preferences.gather(1, history_category)
    reliable = torch.rand((batch_size, spec.history_events), generator=generator) < 0.80
    random_attribute = torch.randint(
        spec.attributes, (batch_size, spec.history_events), generator=generator
    )
    history_attribute = torch.where(reliable, preferred_by_event, random_attribute)
    history_attribute[:, -1] = wanted_attribute
    history_variant = torch.randint(
        spec.item_variants, (batch_size, spec.history_events), generator=generator
    )
    history_tokens = _product_tokens(spec, history_category, history_attribute, history_variant)
    history_mask = torch.ones((batch_size, spec.history_events), dtype=torch.bool)

    exact_repeat = torch.rand((batch_size,), generator=generator) < spec.exact_repeat_probability
    used_variant = torch.zeros((batch_size, spec.item_variants), dtype=torch.bool)
    relevant_event = history_category.eq(categories[:, None]) & history_attribute.eq(wanted_attribute[:, None])
    for event in range(spec.history_events):
        rows = torch.arange(batch_size)[relevant_event[:, event]]
        used_variant[rows, history_variant[rows, event]] = True
    unseen_variant = (~used_variant).to(torch.long).argmax(dim=1)
    positive_variant = torch.where(exact_repeat, history_variant[:, -1], unseen_variant)

    candidate_category = torch.empty((batch_size, spec.candidates), dtype=torch.long)
    candidate_attribute = torch.empty_like(candidate_category)
    candidate_variant = torch.randint(
        spec.item_variants, (batch_size, spec.candidates), generator=generator
    )
    candidate_category[:, 0] = categories
    candidate_attribute[:, 0] = wanted_attribute
    candidate_variant[:, 0] = positive_variant

    for position in range(1, spec.candidates):
        if position <= 3:
            candidate_category[:, position] = categories
            shift = torch.randint(1, spec.attributes, (batch_size,), generator=generator)
            candidate_attribute[:, position] = (wanted_attribute + shift) % spec.attributes
        elif position == 4:
            shift = torch.randint(1, spec.categories, (batch_size,), generator=generator)
            candidate_category[:, position] = (categories + shift) % spec.categories
            candidate_attribute[:, position] = wanted_attribute
        else:
            candidate_category[:, position] = torch.randint(
                spec.categories, (batch_size,), generator=generator
            )
            candidate_attribute[:, position] = torch.randint(
                spec.attributes, (batch_size,), generator=generator
            )

    candidates = _product_tokens(spec, candidate_category, candidate_attribute, candidate_variant)
    random_key = torch.rand((batch_size, spec.candidates), generator=generator)
    permutation = random_key.argsort(dim=1)
    gather_index = permutation.unsqueeze(-1).expand(-1, -1, spec.item_tokens)
    candidate_tokens = candidates.gather(1, gather_index)
    targets = permutation.eq(0).to(torch.long).argmax(dim=1)

    query_tokens = torch.stack(
        (torch.full_like(categories, spec.query_marker_token), spec.category_offset + categories),
        dim=1,
    )
    return SyntheticBatch(
        query_tokens=query_tokens,
        candidate_tokens=candidate_tokens,
        history_tokens=history_tokens,
        history_mask=history_mask,
        targets=targets,
        exact_repeat=exact_repeat,
    )


def corrupt_history(batch: SyntheticBatch, kind: str, generator: torch.Generator) -> tuple[Tensor, Tensor | None]:
    """Return a corruption and an optional evidence-only query override."""

    if kind == "wrong_user":
        permutation = torch.randperm(batch.history_tokens.shape[0], generator=generator)
        return batch.history_tokens[permutation], None
    if kind == "shuffle_events":
        key = torch.rand(batch.history_mask.shape, generator=generator)
        permutation = key.argsort(dim=1)
        index = permutation.unsqueeze(-1).expand_as(batch.history_tokens)
        return batch.history_tokens.gather(1, index), None
    if kind == "query_mask":
        masked = batch.query_tokens.clone()
        masked[:, 1] = 2
        return batch.history_tokens, masked
    raise ValueError(f"unknown corruption: {kind}")
