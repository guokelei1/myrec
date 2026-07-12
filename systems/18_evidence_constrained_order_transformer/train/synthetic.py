"""Deterministic in-memory falsifier for C18."""

from __future__ import annotations

from dataclasses import dataclass

import torch


NO_HISTORY = 0
REPEAT_CONFLICT = 1
SUPPORTED_NONREPEAT = 2


@dataclass
class SyntheticBatch:
    query: torch.Tensor
    candidates: torch.Tensor
    history: torch.Tensor
    history_mask: torch.Tensor
    repeat_mask: torch.Tensor
    candidate_mask: torch.Tensor
    candidate_ids: torch.Tensor
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
            "repeat_mask": self.repeat_mask,
            "candidate_mask": self.candidate_mask,
        }


def _rand(
    shape: tuple[int, ...] | tuple[()], generator: torch.Generator
) -> torch.Tensor:
    return torch.rand(shape, generator=generator)


def _semantic_vector(
    semantic_id: int,
    *,
    input_dim: int,
    generator: torch.Generator,
    candidate: bool,
) -> torch.Tensor:
    topic = semantic_id // 2
    style = -1.0 if semantic_id % 2 == 0 else 1.0
    value = torch.zeros(input_dim)
    value[topic] = 0.95 + 0.10 * float(_rand((), generator))
    style_scale = (0.72 if candidate else 0.56) + 0.06 * float(_rand((), generator))
    value[4 + topic] = style * style_scale
    value[8] = -0.25 + 0.50 * float(_rand((), generator))
    value[9 if candidate else 10] = 1.0
    if input_dim > 12:
        noise = torch.randn(input_dim - 12, generator=generator) * 0.025
        value[12:] = noise
    return value


def _strata(requests: int, weights: list[float], generator: torch.Generator) -> torch.Tensor:
    raw = torch.tensor(weights, dtype=torch.float64) * requests
    counts = torch.floor(raw).to(torch.long)
    counts[-1] += requests - int(counts.sum())
    values = torch.cat(
        [torch.full((int(count),), index, dtype=torch.long) for index, count in enumerate(counts)]
    )
    return values[torch.randperm(requests, generator=generator)]


def generate_split(
    *,
    seed: int,
    split: str,
    requests: int,
    candidates: int,
    history_slots: int,
    input_dim: int,
    topics: int,
    strata_weights: list[float],
) -> SyntheticBatch:
    if candidates != topics * 2 or history_slots != topics * 2:
        raise ValueError("the frozen generator requires two styles per topic")
    if input_dim < 12:
        raise ValueError("input_dim must be at least 12")
    offset = 1101 if split == "train" else 2201
    generator = torch.Generator().manual_seed(seed * 10_000 + offset)
    stratum = _strata(requests, strata_weights, generator)

    query = torch.zeros(requests, input_dim)
    candidate_values = torch.zeros(requests, candidates, input_dim)
    history = torch.zeros(requests, history_slots, input_dim)
    history_mask = torch.zeros(requests, history_slots, dtype=torch.bool)
    repeat_mask = torch.zeros(requests, candidates, dtype=torch.bool)
    candidate_mask = torch.ones(requests, candidates, dtype=torch.bool)
    candidate_ids = torch.zeros(requests, candidates, dtype=torch.long)
    target_index = torch.zeros(requests, dtype=torch.long)

    for row in range(requests):
        topic = int(torch.randint(topics, (), generator=generator))
        query[row, topic] = 1.0
        query[row, 11] = 1.0
        query[row, 8] = -0.10 + 0.20 * float(_rand((), generator))

        canonical_candidates = torch.stack(
            [
                _semantic_vector(
                    semantic_id,
                    input_dim=input_dim,
                    generator=generator,
                    candidate=True,
                )
                for semantic_id in range(candidates)
            ]
        )
        permutation = torch.randperm(candidates, generator=generator)
        candidate_values[row] = canonical_candidates[permutation]
        candidate_ids[row] = permutation

        proxy = (
            2.0 * candidate_values[row, :, topic]
            + 0.20 * candidate_values[row, :, 8]
        )
        target_index[row] = int(proxy.argmax())

        if int(stratum[row]) == NO_HISTORY:
            continue

        canonical_history = torch.stack(
            [
                _semantic_vector(
                    semantic_id,
                    input_dim=input_dim,
                    generator=generator,
                    candidate=False,
                )
                for semantic_id in range(history_slots)
            ]
        )
        event_semantics = torch.randperm(history_slots, generator=generator)
        row_history = canonical_history[event_semantics]

        if int(stratum[row]) == REPEAT_CONFLICT:
            target_style = int(torch.randint(2, (), generator=generator))
            target_semantic = 2 * topic + target_style
            opposite_semantic = 2 * topic + (1 - target_style)
            target_position = int(torch.nonzero(event_semantics.eq(target_semantic))[0])
            opposite_position = int(torch.nonzero(event_semantics.eq(opposite_semantic))[0])
            # The transferable semantic rule (latest query-topic event) is made
            # deliberately inconsistent with the exact identity atom.
            if target_position > opposite_position:
                row_history[[target_position, opposite_position]] = row_history[
                    [opposite_position, target_position]
                ]
                event_semantics[[target_position, opposite_position]] = event_semantics[
                    [opposite_position, target_position]
                ]
                target_position, opposite_position = opposite_position, target_position
            candidate_position = int(torch.nonzero(permutation.eq(target_semantic))[0])
            repeat_mask[row, candidate_position] = True
            target_index[row] = candidate_position
        else:
            positions = torch.nonzero(event_semantics.div(2, rounding_mode="floor").eq(topic)).view(-1)
            latest_position = int(positions.max())
            target_semantic = int(event_semantics[latest_position])
            target_index[row] = int(torch.nonzero(permutation.eq(target_semantic))[0])

        history[row] = row_history
        history_mask[row] = True

    return SyntheticBatch(
        query=query,
        candidates=candidate_values,
        history=history,
        history_mask=history_mask,
        repeat_mask=repeat_mask,
        candidate_mask=candidate_mask,
        candidate_ids=candidate_ids,
        target_index=target_index,
        stratum=stratum,
        request_index=torch.arange(requests),
    )


def corrupt_supported(
    batch: SyntheticBatch, *, seed: int, corruption: str
) -> SyntheticBatch:
    if not bool(batch.stratum.eq(SUPPORTED_NONREPEAT).all()):
        raise ValueError("corruptions are defined only for supported non-repeat rows")
    output = batch.subset(torch.arange(len(batch)))
    generator = torch.Generator().manual_seed(seed * 10_000 + 3301)
    if corruption == "wrong_history":
        offset = 1 + int(torch.randint(len(batch) - 1, (), generator=generator))
        donor = (torch.arange(len(batch)) + offset) % len(batch)
        output.history = batch.history[donor].clone()
        output.history_mask = batch.history_mask[donor].clone()
    elif corruption == "shuffled_event":
        keys = torch.rand(len(batch), batch.history.shape[1], generator=generator)
        permutation = torch.argsort(keys, dim=1, stable=True)
        output.history = torch.gather(
            batch.history,
            1,
            permutation.unsqueeze(-1).expand_as(batch.history),
        )
        output.history_mask = torch.gather(batch.history_mask, 1, permutation)
    elif corruption == "query_mask":
        output.query = torch.zeros_like(batch.query)
    elif corruption == "coarse_only":
        output.history = batch.history.clone()
        output.history[:, :, 4:8] = 0.0
    else:
        raise ValueError(corruption)
    return output


def batch_schedule(
    *, seed: int, requests: int, steps: int, batch_size: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 10_000 + 4401)
    chunks: list[torch.Tensor] = []
    while sum(chunk.numel() for chunk in chunks) < steps * batch_size:
        chunks.append(torch.randperm(requests, generator=generator))
    return torch.cat(chunks)[: steps * batch_size].view(steps, batch_size)


def permute_candidates(batch: SyntheticBatch, permutation: torch.Tensor) -> SyntheticBatch:
    output = batch.subset(torch.arange(len(batch)))
    output.candidates = batch.candidates[:, permutation].clone()
    output.repeat_mask = batch.repeat_mask[:, permutation].clone()
    output.candidate_mask = batch.candidate_mask[:, permutation].clone()
    output.candidate_ids = batch.candidate_ids[:, permutation].clone()
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(permutation.numel())
    output.target_index = inverse[batch.target_index]
    return output
