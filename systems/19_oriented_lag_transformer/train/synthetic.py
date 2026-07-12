"""Deterministic in-memory temporal-cofactor task for C19."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NO_HISTORY = 0
EXACT_REPEAT = 1
SUPPORTED_SUCCESSOR = 2


@dataclass
class SyntheticBatch:
    query: torch.Tensor
    candidates: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    identity_relation: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_ids: torch.Tensor
    history_item_ids: torch.Tensor
    target_index: torch.Tensor
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
            "identity_relation": self.identity_relation,
            "candidate_mask": self.candidate_mask,
        }


def _semantic(
    semantic_id: int,
    *,
    input_dim: int,
    generator: torch.Generator,
    role: str,
) -> torch.Tensor:
    value = torch.zeros(input_dim)
    value[semantic_id] = 0.95 + 0.10 * float(torch.rand((), generator=generator))
    if role == "candidate":
        value[9] = 1.0
    elif role == "history":
        value[10] = 1.0
    elif role == "query":
        value[11] = 1.0
    else:
        raise ValueError(role)
    if input_dim > 12:
        value[12:] = torch.randn(input_dim - 12, generator=generator) * 0.015
    return value


def _strata(requests: int, weights: list[float], generator: torch.Generator) -> torch.Tensor:
    counts = torch.floor(torch.tensor(weights, dtype=torch.float64) * requests).long()
    counts[-1] += requests - int(counts.sum())
    values = torch.cat(
        [torch.full((int(count),), index, dtype=torch.long) for index, count in enumerate(counts)]
    )
    return values[torch.randperm(requests, generator=generator)]


def _recompute_identity(candidate_ids: torch.Tensor, history_ids: torch.Tensor) -> torch.Tensor:
    valid_history = history_ids.ge(0)
    return candidate_ids.unsqueeze(2).eq(history_ids.unsqueeze(1)) & valid_history.unsqueeze(1)


def generate_split(
    *,
    seed: int,
    split: str,
    requests: int,
    candidates: int,
    history_slots: int,
    input_dim: int,
    semantic_items: int,
    strata_weights: list[float],
) -> SyntheticBatch:
    if candidates != semantic_items or history_slots != semantic_items or semantic_items > 8:
        raise ValueError("frozen task requires one candidate/event per semantic item, at most eight")
    if input_dim < 12:
        raise ValueError("input_dim must be at least 12")
    offset = 1201 if split == "train" else 2301
    generator = torch.Generator().manual_seed(seed * 10_000 + offset)
    strata = _strata(requests, strata_weights, generator)

    query = torch.zeros(requests, input_dim)
    candidate_values = torch.zeros(requests, candidates, input_dim)
    history = torch.zeros(requests, history_slots, input_dim)
    history_mask = torch.zeros(requests, history_slots, dtype=torch.bool)
    candidate_ids = torch.zeros(requests, candidates, dtype=torch.long)
    history_ids = torch.full((requests, history_slots), -1, dtype=torch.long)
    target_index = torch.zeros(requests, dtype=torch.long)

    for row in range(requests):
        canonical_candidates = torch.stack(
            [
                _semantic(
                    index,
                    input_dim=input_dim,
                    generator=generator,
                    role="candidate",
                )
                for index in range(candidates)
            ]
        )
        quality = torch.rand(candidates, generator=generator) * 0.6 - 0.3
        canonical_candidates[:, 8] = quality
        candidate_permutation = torch.randperm(candidates, generator=generator)

        if int(strata[row]) == NO_HISTORY:
            query_semantic = int(torch.randint(semantic_items, (), generator=generator))
            target_semantic = int(quality.argmax())
        else:
            event_semantics = torch.randperm(semantic_items, generator=generator)
            canonical_history = torch.stack(
                [
                    _semantic(
                        int(index),
                        input_dim=input_dim,
                        generator=generator,
                        role="history",
                    )
                    for index in event_semantics
                ]
            )
            history[row] = canonical_history
            history_mask[row] = True
            history_ids[row] = torch.arange(history_slots) + 10_000 + row * history_slots

            if int(strata[row]) == EXACT_REPEAT:
                target_semantic = int(torch.randint(semantic_items, (), generator=generator))
                query_semantic = target_semantic
                target_event = int(torch.nonzero(event_semantics.eq(target_semantic))[0])
                history_ids[row, target_event] = target_semantic
                quality[target_semantic] = -0.15
                distractor = (target_semantic + 1 + int(torch.randint(semantic_items - 1, (), generator=generator))) % semantic_items
                quality[distractor] = 0.80
                canonical_candidates[:, 8] = quality
            else:
                pivot = int(torch.randint(1, history_slots - 1, (), generator=generator))
                query_semantic = int(event_semantics[pivot])
                target_semantic = int(event_semantics[pivot + 1])
                predecessor_semantic = int(event_semantics[pivot - 1])
                quality[:] = torch.rand(candidates, generator=generator) * 0.2 - 0.2
                quality[predecessor_semantic] = 1.00
                quality[target_semantic] = 0.00
                canonical_candidates[:, 8] = quality

        query[row] = _semantic(
            query_semantic,
            input_dim=input_dim,
            generator=generator,
            role="query",
        )
        candidate_values[row] = canonical_candidates[candidate_permutation]
        candidate_ids[row] = candidate_permutation
        target_index[row] = int(torch.nonzero(candidate_permutation.eq(target_semantic))[0])

    identity = _recompute_identity(candidate_ids, history_ids)
    return SyntheticBatch(
        query=query,
        candidates=candidate_values,
        history=history,
        history_mask=history_mask,
        identity_relation=identity,
        candidate_mask=torch.ones(requests, candidates, dtype=torch.bool),
        candidate_ids=candidate_ids,
        history_item_ids=history_ids,
        target_index=target_index,
        stratum=strata,
        request_index=torch.arange(requests),
    )


def corrupt_supported(
    batch: SyntheticBatch, *, seed: int, corruption: str
) -> SyntheticBatch:
    if not bool(batch.stratum.eq(SUPPORTED_SUCCESSOR).all()):
        raise ValueError("corruption expects supported-successor rows")
    output = batch.subset(torch.arange(len(batch)))
    generator = torch.Generator().manual_seed(seed * 10_000 + 3401)
    if corruption == "wrong_history":
        offset = 1 + int(torch.randint(len(batch) - 1, (), generator=generator))
        donor = (torch.arange(len(batch)) + offset) % len(batch)
        output.history = batch.history[donor].clone()
        output.history_mask = batch.history_mask[donor].clone()
        output.history_item_ids = batch.history_item_ids[donor].clone()
    elif corruption == "shuffled_event":
        keys = torch.rand(len(batch), batch.history.shape[1], generator=generator)
        permutation = torch.argsort(keys, dim=1, stable=True)
        output.history = torch.gather(
            batch.history, 1, permutation.unsqueeze(-1).expand_as(batch.history)
        )
        output.history_mask = torch.gather(batch.history_mask, 1, permutation)
        output.history_item_ids = torch.gather(batch.history_item_ids, 1, permutation)
    elif corruption == "query_mask":
        output.query = torch.zeros_like(batch.query)
    elif corruption == "coarse_only":
        output.history = batch.history.clone()
        output.history[:, :, :8] = 0.0
    elif corruption == "reversed_event":
        output.history = batch.history.flip(1)
        output.history_mask = batch.history_mask.flip(1)
        output.history_item_ids = batch.history_item_ids.flip(1)
    else:
        raise ValueError(corruption)
    output.identity_relation = _recompute_identity(output.candidate_ids, output.history_item_ids)
    return output


def batch_schedule(*, seed: int, requests: int, steps: int, batch_size: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 10_000 + 4501)
    pieces: list[torch.Tensor] = []
    total = 0
    while total < steps * batch_size:
        piece = torch.randperm(requests, generator=generator)
        pieces.append(piece)
        total += piece.numel()
    return torch.cat(pieces)[: steps * batch_size].view(steps, batch_size)


def permute_candidates(batch: SyntheticBatch, permutation: torch.Tensor) -> SyntheticBatch:
    output = batch.subset(torch.arange(len(batch)))
    output.candidates = batch.candidates[:, permutation].clone()
    output.candidate_ids = batch.candidate_ids[:, permutation].clone()
    output.identity_relation = batch.identity_relation[:, permutation].clone()
    output.candidate_mask = batch.candidate_mask[:, permutation].clone()
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    output.target_index = inverse[batch.target_index]
    return output
