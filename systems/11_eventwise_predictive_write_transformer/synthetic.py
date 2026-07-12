"""Synthetic generator and pre-outcome construct audit for C11."""

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
    def item_tokens(self) -> int:
        return 3


@dataclass
class SyntheticBatch:
    query_tokens: Tensor
    candidate_tokens: Tensor
    history_tokens: Tensor
    history_mask: Tensor
    targets: Tensor
    exact_repeat: Tensor

    def to(self, device: torch.device | str) -> "SyntheticBatch":
        return SyntheticBatch(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )


def _item_token(spec: SyntheticSpec, category: Tensor, attribute: Tensor, variant: Tensor) -> Tensor:
    local = (category * spec.attributes + attribute) * spec.item_variants + variant
    return spec.item_offset + local


def _product_tokens(spec: SyntheticSpec, category: Tensor, attribute: Tensor, variant: Tensor) -> Tensor:
    return torch.stack(
        (
            _item_token(spec, category, attribute, variant),
            spec.category_offset + category,
            spec.attribute_offset + attribute,
        ),
        dim=-1,
    )


def generate_batch(spec: SyntheticSpec, examples: int, generator: torch.Generator) -> SyntheticBatch:
    """Generate role-balanced candidates and eventwise query-conditional history.

    Candidate variants are sampled iid for every role before repeat membership is
    constructed.  Thus the non-repeat positive has no "smallest unseen variant"
    signature.  History is changed, never the candidate, to enforce non-membership.
    """

    query_category = torch.randint(spec.categories, (examples,), generator=generator)
    user_preference = torch.randint(
        spec.attributes, (examples, spec.categories), generator=generator
    )
    wanted_attribute = user_preference.gather(1, query_category[:, None]).squeeze(1)

    candidate_category = torch.empty((examples, spec.candidates), dtype=torch.long)
    candidate_attribute = torch.empty_like(candidate_category)
    candidate_variant = torch.randint(
        spec.item_variants, (examples, spec.candidates), generator=generator
    )
    candidate_category[:, 0] = query_category
    candidate_attribute[:, 0] = wanted_attribute
    for position in range(1, spec.candidates):
        if position <= 3:
            candidate_category[:, position] = query_category
            offset = torch.randint(1, spec.attributes, (examples,), generator=generator)
            candidate_attribute[:, position] = (wanted_attribute + offset) % spec.attributes
        elif position == 4:
            offset = torch.randint(1, spec.categories, (examples,), generator=generator)
            candidate_category[:, position] = (query_category + offset) % spec.categories
            candidate_attribute[:, position] = wanted_attribute
        else:
            candidate_category[:, position] = torch.randint(
                spec.categories, (examples,), generator=generator
            )
            candidate_attribute[:, position] = torch.randint(
                spec.attributes, (examples,), generator=generator
            )
    unshuffled_candidates = _product_tokens(
        spec, candidate_category, candidate_attribute, candidate_variant
    )

    history_category = torch.randint(
        spec.categories, (examples, spec.history_events), generator=generator
    )
    history_category[:, -1] = query_category
    event_preference = user_preference.gather(1, history_category)
    reliable = torch.rand((examples, spec.history_events), generator=generator) < 0.75
    random_attribute = torch.randint(
        spec.attributes, (examples, spec.history_events), generator=generator
    )
    history_attribute = torch.where(reliable, event_preference, random_attribute)
    history_attribute[:, -1] = wanted_attribute
    history_variant = torch.randint(
        spec.item_variants, (examples, spec.history_events), generator=generator
    )

    exact_repeat = (
        torch.rand((examples,), generator=generator) < spec.exact_repeat_probability
    )
    positive_variant = candidate_variant[:, 0]
    # Exact requests copy the independently sampled candidate into the last event.
    history_variant[:, -1] = torch.where(
        exact_repeat, positive_variant, history_variant[:, -1]
    )
    # Non-repeat requests change colliding history variants by a non-zero uniform
    # modular offset.  Candidate marginals remain untouched and iid across roles.
    positive_item = _item_token(
        spec, query_category, wanted_attribute, positive_variant
    )
    for event in range(spec.history_events):
        event_item = _item_token(
            spec,
            history_category[:, event],
            history_attribute[:, event],
            history_variant[:, event],
        )
        collision = (~exact_repeat) & event_item.eq(positive_item)
        offset = torch.randint(1, spec.item_variants, (examples,), generator=generator)
        changed = (history_variant[:, event] + offset) % spec.item_variants
        history_variant[:, event] = torch.where(
            collision, changed, history_variant[:, event]
        )

    history_tokens = _product_tokens(
        spec, history_category, history_attribute, history_variant
    )
    history_mask = torch.ones((examples, spec.history_events), dtype=torch.bool)

    key = torch.rand((examples, spec.candidates), generator=generator)
    permutation = key.argsort(dim=1)
    candidate_tokens = unshuffled_candidates.gather(
        1, permutation.unsqueeze(-1).expand(-1, -1, spec.item_tokens)
    )
    targets = permutation.eq(0).to(torch.long).argmax(dim=1)
    query_tokens = torch.stack(
        (
            torch.ones_like(query_category),
            spec.category_offset + query_category,
        ),
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


def _total_variation(first: Tensor, second: Tensor, bins: int) -> float:
    first_count = torch.bincount(first, minlength=bins).to(torch.float64)
    second_count = torch.bincount(second, minlength=bins).to(torch.float64)
    first_probability = first_count / first_count.sum()
    second_probability = second_count / second_count.sum()
    return float(0.5 * (first_probability - second_probability).abs().sum())


def construct_audit(batch: SyntheticBatch, spec: SyntheticSpec) -> dict[str, float | bool]:
    """Audit target-position and candidate-local role marginals before training."""

    examples = batch.targets.shape[0]
    row = torch.arange(examples)
    positive = batch.candidate_tokens[row, batch.targets]
    candidate_category = batch.candidate_tokens[:, :, 1]
    query_category = batch.query_tokens[:, 1]
    hard_negative = candidate_category.eq(query_category[:, None])
    hard_negative[row, batch.targets] = False
    negative = batch.candidate_tokens[hard_negative]

    positive_variant = (positive[:, 0] - spec.item_offset) % spec.item_variants
    negative_variant = (negative[:, 0] - spec.item_offset) % spec.item_variants
    positive_attribute = positive[:, 2] - spec.attribute_offset
    negative_attribute = negative[:, 2] - spec.attribute_offset
    positive_joint = positive_attribute * spec.item_variants + positive_variant
    negative_joint = negative_attribute * spec.item_variants + negative_variant

    target_probability = torch.bincount(
        batch.targets, minlength=spec.candidates
    ).to(torch.float64) / examples
    target_deviation = float(
        (target_probability - 1.0 / spec.candidates).abs().amax()
    )
    positive_item = positive[:, 0]
    membership = positive_item[:, None].eq(batch.history_tokens[:, :, 0]).any(dim=1)
    exact_membership_ok = bool(torch.equal(membership, batch.exact_repeat))
    hard_negative_count_ok = bool(torch.all(hard_negative.sum(dim=1) >= 3))
    return {
        "target_position_max_deviation": target_deviation,
        "variant_total_variation": _total_variation(
            positive_variant, negative_variant, spec.item_variants
        ),
        "attribute_variant_total_variation": _total_variation(
            positive_joint, negative_joint, spec.attributes * spec.item_variants
        ),
        "exact_membership_ok": exact_membership_ok,
        "hard_negative_count_ok": hard_negative_count_ok,
        "positive_variant_zero_rate": float(positive_variant.eq(0).float().mean()),
        "negative_variant_zero_rate": float(negative_variant.eq(0).float().mean()),
    }


def corrupt_history(
    batch: SyntheticBatch, kind: str, generator: torch.Generator
) -> tuple[Tensor, Tensor | None]:
    if kind == "wrong_user":
        permutation = torch.randperm(batch.history_tokens.shape[0], generator=generator)
        return batch.history_tokens[permutation], None
    if kind == "shuffle_events":
        permutation = torch.rand(batch.history_mask.shape, generator=generator).argsort(dim=1)
        return batch.history_tokens.gather(
            1, permutation.unsqueeze(-1).expand_as(batch.history_tokens)
        ), None
    if kind == "query_mask":
        evidence_query = batch.query_tokens.clone()
        evidence_query[:, 1] = 2
        return batch.history_tokens, evidence_query
    raise ValueError(f"unknown corruption: {kind}")
