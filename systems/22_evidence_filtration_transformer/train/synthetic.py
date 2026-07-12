"""Synthetic strata and interventions for the C22 falsifier."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch.nn import functional as F


NO_HISTORY = 0
EXACT_RECURRENCE = 1
SUPPORTED_TRANSFER = 2


@dataclass(frozen=True)
class SyntheticBatch:
    query: torch.Tensor
    candidates: torch.Tensor
    history: torch.Tensor
    identity: torch.Tensor
    event_strength: torch.Tensor
    history_mask: torch.Tensor
    candidate_mask: torch.Tensor
    base_scores: torch.Tensor
    target_index: torch.Tensor
    candidate_ids: torch.Tensor
    stratum: torch.Tensor

    def __len__(self) -> int:
        return int(self.query.shape[0])

    def subset(self, indices: torch.Tensor) -> "SyntheticBatch":
        return SyntheticBatch(
            **{name: value[indices] for name, value in self.__dict__.items()}
        )

    def to(self, device: torch.device) -> "SyntheticBatch":
        return SyntheticBatch(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    def model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "query": self.query,
            "candidates": self.candidates,
            "history": self.history,
            "identity": self.identity,
            "event_strength": self.event_strength,
            "history_mask": self.history_mask,
            "candidate_mask": self.candidate_mask,
            "base_scores": self.base_scores,
        }


def _generator(seed: int, split: str) -> torch.Generator:
    offset = {"train": 0, "eval": 10_000_019, "audit": 20_000_033}[split]
    return torch.Generator().manual_seed(seed * 100_003 + offset)


def generate_split(
    *,
    seed: int,
    split: str,
    requests: int,
    candidates: int,
    history_slots: int,
    input_dim: int,
    strata_weights: list[float],
) -> SyntheticBatch:
    if candidates % 2 or input_dim < 8:
        raise ValueError("C22 requires sign-paired candidates and at least eight dimensions")
    generator = _generator(seed, split)
    probabilities = torch.tensor(strata_weights, dtype=torch.float64)
    probabilities = probabilities / probabilities.sum()
    stratum = torch.multinomial(probabilities, requests, replacement=True, generator=generator).long()
    target = torch.randint(candidates, (requests,), generator=generator)
    query = F.normalize(torch.randn(requests, input_dim, generator=generator), dim=-1)

    half = candidates // 2
    positive = F.normalize(torch.randn(requests, half, input_dim, generator=generator), dim=-1)
    candidate_states = torch.cat([positive, -positive], dim=1)
    # Randomly permute candidate positions independently per request so pair
    # structure and tensor position cannot identify the target.
    permutations = torch.stack(
        [torch.randperm(candidates, generator=generator) for _ in range(requests)]
    )
    candidate_states = candidate_states.gather(
        1, permutations[:, :, None].expand(-1, -1, input_dim)
    )
    candidate_ids = torch.stack(
        [torch.randperm(candidates, generator=generator) for _ in range(requests)]
    ).long()

    history = F.normalize(
        torch.randn(requests, history_slots, input_dim, generator=generator), dim=-1
    )
    history_mask = stratum.ne(NO_HISTORY)[:, None].expand(-1, history_slots).clone()
    identity = torch.zeros(requests, candidates, history_slots, dtype=torch.bool)
    event_strength = torch.linspace(0.5, 1.0, history_slots)[None, :].expand(requests, -1).clone()
    event_strength += 0.05 * torch.randn(requests, history_slots, generator=generator)
    event_strength = event_strength.clamp(0.25, 1.25) * history_mask
    base_scores = 0.02 * torch.randn(requests, candidates, generator=generator)

    rows = torch.arange(requests)
    # No-history rows are solved only by the supplied base, establishing exact
    # fallback independently of the personalized Transformer.
    nohistory = stratum.eq(NO_HISTORY)
    base_scores[nohistory] = -2.0
    base_scores[nohistory, target[nohistory]] = 4.0

    # Exact recurrence is a symbolic relation only: raw event content is
    # independently sampled, so removing the equality tensor removes the atom.
    repeated = stratum.eq(EXACT_RECURRENCE)
    repeated_rows = rows[repeated]
    repeated_targets = target[repeated]
    repeated_slots = torch.randint(history_slots, (len(repeated_rows),), generator=generator)
    identity[repeated_rows, repeated_targets, repeated_slots] = True
    event_strength[repeated_rows, repeated_slots] += 0.75

    # Supported non-repeat rows require the joint relation q + h_last = c*.
    # Query or candidate marginals alone are independent of the chosen target.
    supported = stratum.eq(SUPPORTED_TRANSFER)
    supported_rows = rows[supported]
    supported_targets = target[supported]
    target_states = candidate_states[supported_rows, supported_targets]
    final_event = target_states - query[supported_rows]
    history[supported_rows, -1] = final_event
    # Put a plausible but independent distractor in the penultimate position.
    distractor = (supported_targets + candidates // 2) % candidates
    history[supported_rows, -2] = candidate_states[supported_rows, distractor]
    event_strength[supported_rows, -1] = 1.25
    event_strength[supported_rows, -2] = 1.0

    return SyntheticBatch(
        query=query.float(),
        candidates=candidate_states.float(),
        history=history.float(),
        identity=identity,
        event_strength=event_strength.float(),
        history_mask=history_mask,
        candidate_mask=torch.ones(requests, candidates, dtype=torch.bool),
        base_scores=base_scores.float(),
        target_index=target,
        candidate_ids=candidate_ids,
        stratum=stratum,
    )


def batch_schedule(
    *, seed: int, requests: int, steps: int, batch_size: int
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed * 1_000_003 + 2201)
    return torch.randint(requests, (steps, batch_size), generator=generator)


def corrupt_supported(
    batch: SyntheticBatch, *, seed: int, corruption: str
) -> SyntheticBatch:
    if not bool(batch.stratum.eq(SUPPORTED_TRANSFER).all()):
        raise ValueError("C22 corruptions require supported-only rows")
    generator = torch.Generator().manual_seed(seed * 1009 + sum(map(ord, corruption)))
    if corruption == "wrong_history":
        offset = int(torch.randint(1, len(batch), (1,), generator=generator))
        order = torch.roll(torch.arange(len(batch)), shifts=offset)
        return replace(
            batch,
            history=batch.history[order],
            identity=batch.identity[order],
            event_strength=batch.event_strength[order],
            history_mask=batch.history_mask[order],
        )
    if corruption == "shuffled_event":
        permutations = torch.stack(
            [torch.randperm(batch.history.shape[1], generator=generator) for _ in range(len(batch))]
        )
        identity_permutation = permutations[:, None, :].expand(-1, batch.identity.shape[1], -1)
        return replace(
            batch,
            history=batch.history.gather(
                1, permutations[:, :, None].expand(-1, -1, batch.history.shape[-1])
            ),
            identity=batch.identity.gather(2, identity_permutation),
            event_strength=batch.event_strength.gather(1, permutations),
            history_mask=batch.history_mask.gather(1, permutations),
        )
    if corruption == "coarse_only":
        query = torch.zeros_like(batch.query)
        candidates = torch.zeros_like(batch.candidates)
        history = torch.zeros_like(batch.history)
        width = max(1, batch.query.shape[-1] // 8)
        query[:, :width] = batch.query[:, :width]
        candidates[:, :, :width] = batch.candidates[:, :, :width]
        history[:, :, :width] = batch.history[:, :, :width]
        return replace(batch, query=query, candidates=candidates, history=history)
    raise ValueError(f"unknown C22 supported corruption: {corruption}")


def remove_identity(batch: SyntheticBatch) -> SyntheticBatch:
    return replace(batch, identity=torch.zeros_like(batch.identity))


def permute_candidates(
    batch: SyntheticBatch, permutation: torch.Tensor
) -> SyntheticBatch:
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(len(permutation))
    return replace(
        batch,
        candidates=batch.candidates[:, permutation],
        identity=batch.identity[:, permutation],
        candidate_mask=batch.candidate_mask[:, permutation],
        base_scores=batch.base_scores[:, permutation],
        candidate_ids=batch.candidate_ids[:, permutation],
        target_index=inverse[batch.target_index],
    )
