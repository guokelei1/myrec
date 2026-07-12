"""Deterministic exchangeable synthetic task for C20."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NO_HISTORY = 0
EXACT_REPEAT = 1
SUPPORTED_COMPOSITION = 2


@dataclass
class SyntheticBatch:
    query: torch.Tensor
    candidates: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_ids: torch.Tensor
    canonical_ids: torch.Tensor
    target_index: torch.Tensor
    reverse_index: torch.Tensor
    true_coefficients: torch.Tensor
    stratum: torch.Tensor
    request_index: torch.Tensor

    def __len__(self) -> int:
        return int(self.query.shape[0])

    def subset(self, indices: torch.Tensor) -> "SyntheticBatch":
        return SyntheticBatch(
            **{name: getattr(self, name)[indices] for name in self.__dataclass_fields__}
        )

    def to(self, device: torch.device | str) -> "SyntheticBatch":
        return SyntheticBatch(
            **{name: getattr(self, name).to(device) for name in self.__dataclass_fields__}
        )

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "query": self.query,
            "candidates": self.candidates,
            "history": self.history,
            "history_mask": self.history_mask,
            "candidate_mask": self.candidate_mask,
        }


def _strata(requests: int, weights: list[float], generator: torch.Generator) -> torch.Tensor:
    counts = torch.floor(torch.tensor(weights, dtype=torch.float64) * requests).long()
    counts[-1] += requests - int(counts.sum())
    values = torch.cat(
        [torch.full((int(count),), index, dtype=torch.long) for index, count in enumerate(counts)]
    )
    return values[torch.randperm(requests, generator=generator)]


def _orthogonal(dim: int, generator: torch.Generator) -> torch.Tensor:
    matrix = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(matrix)
    signs = torch.where(torch.diag(r).ge(0), 1.0, -1.0)
    return q * signs.unsqueeze(0)


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value / torch.linalg.vector_norm(value).clamp_min(1e-8)


def generate_split(
    *,
    seed: int,
    split: str,
    requests: int,
    candidates: int,
    history_slots: int,
    input_dim: int,
    relation_raw_dim: int,
    strata_weights: list[float],
) -> SyntheticBatch:
    if candidates != 8 or history_slots != 7 or relation_raw_dim != 8 or input_dim < 12:
        raise ValueError("frozen C20 task requires C=8, H=7, relation_raw_dim=8, input_dim>=12")
    offset = 1701 if split == "train" else 2901
    generator = torch.Generator().manual_seed(seed * 10_000 + offset)
    strata = _strata(requests, strata_weights, generator)

    query = torch.zeros(requests, input_dim)
    candidate_values = torch.zeros(requests, candidates, input_dim)
    history = torch.zeros(requests, history_slots, input_dim)
    history_mask = torch.zeros(requests, history_slots, dtype=torch.bool)
    candidate_ids = torch.zeros(requests, candidates, dtype=torch.long)
    canonical_ids = torch.zeros(requests, candidates, dtype=torch.long)
    target_index = torch.zeros(requests, dtype=torch.long)
    reverse_index = torch.zeros(requests, dtype=torch.long)
    true_coefficients = torch.zeros(requests, history_slots - 1)
    target_radius = 1.0

    for row in range(requests):
        permutation = torch.randperm(candidates, generator=generator)
        canonical_ids[row] = permutation
        opaque_ids = torch.randperm(candidates, generator=generator) + row * candidates
        candidate_ids[row] = opaque_ids[permutation]
        kind = int(strata[row])

        if kind == NO_HISTORY:
            q_relation = torch.randn(relation_raw_dim, generator=generator) * 0.25
            query[row, :relation_raw_dim] = q_relation
            canonical = torch.randn(candidates, input_dim, generator=generator) * 0.05
            quality = torch.rand(candidates, generator=generator) * 2.0 - 1.0
            canonical[:, 8] = quality
            target_id = int(quality.argmax())
            reverse_id = (target_id + 1) % candidates
        elif kind == EXACT_REPEAT:
            q_relation = torch.randn(relation_raw_dim, generator=generator) * 0.35
            query[row, :relation_raw_dim] = q_relation
            canonical = torch.zeros(candidates, input_dim)
            canonical[0, :relation_raw_dim] = q_relation
            for index in range(1, candidates):
                direction = _unit(torch.randn(relation_raw_dim, generator=generator))
                canonical[index, :relation_raw_dim] = q_relation + target_radius * direction
            history[row, :, :relation_raw_dim] = q_relation
            history_mask[row] = True
            target_id, reverse_id = 0, 1
        else:
            rotation = _orthogonal(relation_raw_dim, generator)
            q_relation = rotation @ (torch.randn(relation_raw_dim, generator=generator) * 0.25)
            query[row, :relation_raw_dim] = q_relation
            canonical_rays: list[torch.Tensor] = []
            for _ in range(history_slots - 1):
                tangent = _unit(torch.randn(relation_raw_dim - 1, generator=generator))
                ray = torch.cat((torch.tensor([0.12]), (1.0 - 0.12**2) ** 0.5 * tangent))
                canonical_rays.append(ray)
            transition_values = torch.stack(tuple(rotation @ value for value in canonical_rays))
            start = rotation @ (torch.randn(relation_raw_dim, generator=generator) * 0.4)
            event_values = [start]
            for transition in transition_values:
                event_values.append(event_values[-1] + transition)
            history[row, :, :relation_raw_dim] = torch.stack(event_values)
            history_mask[row] = True

            active = torch.randperm(history_slots - 1, generator=generator)[:3]
            coefficients = torch.zeros(history_slots - 1)
            coefficients[active] = torch.exp(torch.randn(3, generator=generator))
            target_displacement = coefficients @ transition_values
            normalization = torch.linalg.vector_norm(target_displacement).clamp_min(1e-8)
            coefficients = coefficients / normalization
            target_direction = target_displacement / normalization
            true_coefficients[row] = coefficients
            complement_seed = torch.randn(relation_raw_dim, 3, generator=generator)
            complement_seed = complement_seed - target_direction[:, None] * (
                target_direction @ complement_seed
            )[None, :]
            complement = torch.linalg.qr(complement_seed, mode="reduced").Q.T
            frame = torch.cat((target_direction.unsqueeze(0), complement), dim=0)
            displacements = torch.stack(
                tuple(value for direction in frame for value in (direction, -direction))
            ) * target_radius
            canonical = torch.zeros(candidates, input_dim)
            canonical[:, :relation_raw_dim] = q_relation.unsqueeze(0) + displacements
            target_id, reverse_id = 0, 1

        candidate_values[row] = canonical[permutation]
        target_index[row] = int(torch.nonzero(permutation.eq(target_id))[0])
        reverse_index[row] = int(torch.nonzero(permutation.eq(reverse_id))[0])

    return SyntheticBatch(
        query=query,
        candidates=candidate_values,
        history=history,
        history_mask=history_mask,
        candidate_mask=torch.ones(requests, candidates, dtype=torch.bool),
        candidate_ids=candidate_ids,
        canonical_ids=canonical_ids,
        target_index=target_index,
        reverse_index=reverse_index,
        true_coefficients=true_coefficients,
        stratum=strata,
        request_index=torch.arange(requests),
    )


def corrupt_supported(batch: SyntheticBatch, *, seed: int, corruption: str) -> SyntheticBatch:
    if not bool(batch.stratum.eq(SUPPORTED_COMPOSITION).all()):
        raise ValueError("corruption expects supported-composition rows")
    output = batch.subset(torch.arange(len(batch)))
    generator = torch.Generator().manual_seed(seed * 10_000 + 3901)
    if corruption == "wrong_history":
        offset = 1 + int(torch.randint(len(batch) - 1, (), generator=generator))
        donor = (torch.arange(len(batch)) + offset) % len(batch)
        output.history = batch.history[donor].clone()
        output.history_mask = batch.history_mask[donor].clone()
    elif corruption == "shuffled_event":
        keys = torch.rand(len(batch), batch.history.shape[1], generator=generator)
        permutation = torch.argsort(keys, dim=1, stable=True)
        output.history = torch.gather(
            batch.history, 1, permutation.unsqueeze(-1).expand_as(batch.history)
        )
        output.history_mask = torch.gather(batch.history_mask, 1, permutation)
    elif corruption == "query_mask":
        output.query = torch.zeros_like(batch.query)
    elif corruption == "coarse_only":
        output.history = batch.history.clone()
        output.history[:, :, :8] = 0.0
    elif corruption == "reversed_event":
        output.history = batch.history.flip(1)
        output.history_mask = batch.history_mask.flip(1)
    else:
        raise ValueError(corruption)
    return output


def permute_candidates(batch: SyntheticBatch, permutation: torch.Tensor) -> SyntheticBatch:
    output = batch.subset(torch.arange(len(batch)))
    output.candidates = batch.candidates[:, permutation].clone()
    output.candidate_mask = batch.candidate_mask[:, permutation].clone()
    output.candidate_ids = batch.candidate_ids[:, permutation].clone()
    output.canonical_ids = batch.canonical_ids[:, permutation].clone()
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    output.target_index = inverse[batch.target_index]
    output.reverse_index = inverse[batch.reverse_index]
    return output


def batch_schedule(*, seed: int, requests: int, steps: int, batch_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 10_000 + 4701)
    pieces: list[torch.Tensor] = []
    total = 0
    while total < steps * batch_size:
        piece = torch.randperm(requests, generator=generator)
        pieces.append(piece)
        total += piece.numel()
    return torch.cat(pieces)[: steps * batch_size].view(steps, batch_size)
