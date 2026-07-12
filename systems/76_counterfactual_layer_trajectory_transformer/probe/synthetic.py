"""Frozen raw-token synthetic surface for the C76 design gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from model.cltt import CANDIDATE, HISTORY, QUERY


CLS = 1
SEP = 2
ATTR_BASE = 10
PAIR_BASE = 32
NUISANCE_POSITIVE = 100
NUISANCE_NEGATIVE = 101
NULL_QUERY = 102
NULL_HISTORY = 103
ITEM_BASE = 120


@dataclass
class SyntheticSurface:
    tokens: torch.Tensor
    wrong_tokens: torch.Tensor
    shuffled_tokens: torch.Tensor
    query_masked_tokens: torch.Tensor
    segments: torch.Tensor
    labels: torch.Tensor
    base_scores: torch.Tensor
    repeat_scores: torch.Tensor
    history_present: torch.Tensor
    repeat_present: torch.Tensor
    strata: torch.Tensor

    def to(self, device: torch.device) -> "SyntheticSurface":
        return SyntheticSurface(
            **{name: value.to(device) for name, value in self.__dict__.items()}
        )

    def subset(self, indices: np.ndarray | torch.Tensor) -> "SyntheticSurface":
        index = torch.as_tensor(indices, dtype=torch.long)
        return SyntheticSurface(
            **{name: value[index] for name, value in self.__dict__.items()}
        )


def pair_token(attribute: int, value: int, values_per_attribute: int) -> int:
    return PAIR_BASE + attribute * values_per_attribute + value


def make_surface(
    *,
    requests: int,
    candidates: int,
    history_events: int,
    attributes: int,
    values_per_attribute: int,
    seed: int,
    split: str,
) -> SyntheticSurface:
    if candidates != values_per_attribute:
        raise ValueError("C76 synthetic candidates must equal values per attribute")
    rng = np.random.default_rng(seed)
    query_length = 3
    candidate_length = attributes * 2 + 3
    history_event_length = 4
    length = query_length + candidate_length + history_events * history_event_length
    segments = np.asarray(
        [QUERY] * query_length
        + [CANDIDATE] * candidate_length
        + [HISTORY] * (history_events * history_event_length),
        dtype=np.int64,
    )
    tokens = np.empty((requests, candidates, length), dtype=np.int64)
    wrong = np.empty_like(tokens)
    shuffled = np.empty_like(tokens)
    query_masked = np.empty_like(tokens)
    labels = np.zeros((requests, candidates), dtype=np.float32)
    base_scores = np.zeros((requests, candidates), dtype=np.float32)
    repeat_scores = np.zeros((requests, candidates), dtype=np.float32)
    history_present = np.ones(requests, dtype=bool)
    repeat_present = np.zeros(requests, dtype=bool)
    strata = np.zeros(requests, dtype=np.int64)  # 0 supported, 1 repeat, 2 nohistory

    for row in range(requests):
        draw = rng.random()
        stratum = 0 if draw < 0.70 else (1 if draw < 0.85 else 2)
        strata[row] = stratum
        query_attribute = int(rng.integers(attributes))
        preference = int(rng.integers(values_per_attribute))
        positive = int(rng.integers(candidates))
        labels[row, positive] = 1.0
        candidate_values = rng.integers(
            values_per_attribute, size=(candidates, attributes), dtype=np.int64
        )
        target_values = np.arange(values_per_attribute, dtype=np.int64)
        rng.shuffle(target_values)
        candidate_values[:, query_attribute] = target_values
        positive = int(np.flatnonzero(target_values == preference)[0])
        labels[row] = 0.0
        labels[row, positive] = 1.0

        history_attributes = [query_attribute]
        choices = [value for value in range(attributes) if value != query_attribute]
        rng.shuffle(choices)
        history_attributes.extend(choices[: history_events - 1])
        history_values = [preference]
        history_values.extend(
            int(rng.integers(values_per_attribute)) for _ in range(history_events - 1)
        )
        event_ids = [int(rng.integers(candidates)) for _ in range(history_events)]
        if stratum == 1:
            event_ids[0] = positive
            repeat_present[row] = True
            repeat_scores[row, positive] = 5.0
        elif stratum == 2:
            history_present[row] = False
            base_scores[row, positive] = 5.0

        events = []
        for attribute, value, item_id in zip(history_attributes, history_values, event_ids):
            events.append(
                [
                    ATTR_BASE + attribute,
                    pair_token(attribute, value, values_per_attribute),
                    ITEM_BASE + item_id,
                    SEP,
                ]
            )
        wrong_events = [list(value) for value in events]
        wrong_preference = (preference + 1 + int(rng.integers(values_per_attribute - 1))) % values_per_attribute
        wrong_events[0][1] = pair_token(
            query_attribute, wrong_preference, values_per_attribute
        )
        permutation = rng.permutation(history_events)
        shuffled_events = [events[int(index)] for index in permutation]

        for candidate in range(candidates):
            if stratum == 0:
                if split == "train":
                    nuisance = NUISANCE_POSITIVE if candidate == positive else NUISANCE_NEGATIVE
                else:
                    nuisance = NUISANCE_NEGATIVE if candidate == positive else NUISANCE_POSITIVE
            else:
                nuisance = (
                    NUISANCE_POSITIVE
                    if bool(rng.integers(2))
                    else NUISANCE_NEGATIVE
                )
            candidate_part = []
            for attribute in range(attributes):
                candidate_part.extend(
                    [
                        ATTR_BASE + attribute,
                        pair_token(
                            attribute,
                            int(candidate_values[candidate, attribute]),
                            values_per_attribute,
                        ),
                    ]
                )
            candidate_part.extend([nuisance, ITEM_BASE + candidate, SEP])
            history_part = [token for event in events for token in event]
            wrong_part = [token for event in wrong_events for token in event]
            shuffled_part = [token for event in shuffled_events for token in event]
            if stratum == 2:
                history_part = [NULL_HISTORY] * len(history_part)
                wrong_part = list(history_part)
                shuffled_part = list(history_part)
            prefix = [CLS, ATTR_BASE + query_attribute, SEP, *candidate_part]
            tokens[row, candidate] = [*prefix, *history_part]
            wrong[row, candidate] = [*prefix, *wrong_part]
            shuffled[row, candidate] = [*prefix, *shuffled_part]
            masked_prefix = [CLS, NULL_QUERY, SEP, *candidate_part]
            query_masked[row, candidate] = [*masked_prefix, *history_part]

    return SyntheticSurface(
        tokens=torch.from_numpy(tokens),
        wrong_tokens=torch.from_numpy(wrong),
        shuffled_tokens=torch.from_numpy(shuffled),
        query_masked_tokens=torch.from_numpy(query_masked),
        segments=torch.from_numpy(np.broadcast_to(segments, (requests, candidates, length)).copy()),
        labels=torch.from_numpy(labels),
        base_scores=torch.from_numpy(base_scores),
        repeat_scores=torch.from_numpy(repeat_scores),
        history_present=torch.from_numpy(history_present),
        repeat_present=torch.from_numpy(repeat_present),
        strata=torch.from_numpy(strata),
    )
