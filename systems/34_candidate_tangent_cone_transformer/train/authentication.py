"""Strictly prequential user-memory authentication for C34 history events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from train.structure import PackedStructure


@dataclass(frozen=True)
class AuthenticationRows:
    request_indices: np.ndarray
    true_offsets: np.ndarray
    true_items: np.ndarray
    wrong_offsets: np.ndarray
    wrong_items: np.ndarray
    profile_sizes: np.ndarray
    true_source_counts: np.ndarray
    wrong_source_counts: np.ndarray
    audit: dict[str, int | float | bool]

    def __post_init__(self) -> None:
        count = len(self.request_indices)
        if len(self.true_offsets) != count + 1 or len(self.wrong_offsets) != count + 1:
            raise ValueError("C34 authentication offsets differ")
        if int(self.true_offsets[-1]) != len(self.true_items):
            raise ValueError("C34 true authentication values differ")
        if int(self.wrong_offsets[-1]) != len(self.wrong_items):
            raise ValueError("C34 wrong authentication values differ")
        for value in (self.profile_sizes, self.true_source_counts, self.wrong_source_counts):
            if len(value) != count:
                raise ValueError("C34 authentication row metadata differs")


def load_user_ids(path: str | Path, data: PackedStructure) -> list[str]:
    positions = {request_id: index for index, request_id in enumerate(data.request_ids)}
    users: list[str | None] = [None] * len(data.request_ids)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            index = positions.get(str(row["request_id"]))
            if index is None:
                continue
            if users[index] is not None:
                raise ValueError("duplicate C34 label-free request metadata")
            if str(row.get("split")) != "train":
                raise ValueError("C34 user metadata contains non-train packed request")
            if int(row["time_index"]) != int(data.timestamps[index]):
                raise ValueError("C34 user metadata timestamp differs")
            users[index] = str(row["user_id"])
    if any(value is None for value in users):
        missing = sum(value is None for value in users)
        raise ValueError(f"C34 user metadata missing {missing} packed requests")
    return [str(value) for value in users]


def build_authentication(
    *,
    data: PackedStructure,
    user_ids: Sequence[str],
    target_indices: Sequence[int],
    donor_by_recipient: Mapping[int, int],
) -> AuthenticationRows:
    """Authenticate against histories observed at strictly smaller timestamps.

    The memory traverses every packed train request.  All requests at the same
    timestamp read the same pre-update memory, then commit as one group.  Wrong
    histories are tested against the recipient user's memory.
    """

    if len(user_ids) != len(data.request_ids):
        raise ValueError("C34 user/request cardinality differs")
    targets = [int(value) for value in target_indices]
    if len(targets) != len(set(targets)):
        raise ValueError("C34 authentication targets overlap")
    target_set = set(targets)
    if set(donor_by_recipient) - target_set:
        raise ValueError("C34 donor mapping has non-target recipients")

    true_by_index: dict[int, np.ndarray] = {}
    wrong_by_index: dict[int, np.ndarray] = {}
    profile_by_index: dict[int, int] = {}
    true_count_by_index: dict[int, int] = {}
    wrong_count_by_index: dict[int, int] = {}
    profiles: dict[str, set[int]] = defaultdict(set)
    order = np.argsort(np.asarray(data.timestamps), kind="stable")
    cursor = 0
    timestamp_groups = 0
    same_timestamp_multi_request_groups = 0
    while cursor < len(order):
        timestamp = int(data.timestamps[int(order[cursor])])
        stop = cursor + 1
        while stop < len(order) and int(data.timestamps[int(order[stop])]) == timestamp:
            stop += 1
        group = [int(value) for value in order[cursor:stop]]
        timestamp_groups += 1
        same_timestamp_multi_request_groups += int(len(group) > 1)
        for index in group:
            if index not in target_set:
                continue
            profile = profiles[str(user_ids[index])]
            true_source = data.history_indices(index).astype(np.int64, copy=False)
            donor = donor_by_recipient.get(index)
            wrong_source = (
                data.history_indices(int(donor)).astype(np.int64, copy=False)
                if donor is not None
                else np.empty(0, dtype=np.int64)
            )
            true_by_index[index] = np.asarray(
                [int(value) for value in true_source if int(value) in profile], dtype=np.int64
            )
            wrong_by_index[index] = np.asarray(
                [int(value) for value in wrong_source if int(value) in profile], dtype=np.int64
            )
            profile_by_index[index] = len(profile)
            true_count_by_index[index] = len(true_source)
            wrong_count_by_index[index] = len(wrong_source)
        # Update only after the whole timestamp group's read phase.
        for index in group:
            profiles[str(user_ids[index])].update(
                int(value) for value in data.history_indices(index)
            )
        cursor = stop

    if set(true_by_index) != target_set or set(wrong_by_index) != target_set:
        raise RuntimeError("C34 authentication target coverage differs")
    true_offsets = [0]
    wrong_offsets = [0]
    true_rows: list[np.ndarray] = []
    wrong_rows: list[np.ndarray] = []
    for index in targets:
        true_rows.append(true_by_index[index])
        wrong_rows.append(wrong_by_index[index])
        true_offsets.append(true_offsets[-1] + len(true_rows[-1]))
        wrong_offsets.append(wrong_offsets[-1] + len(wrong_rows[-1]))
    true_items = (
        np.concatenate(true_rows).astype(np.int64, copy=False)
        if true_offsets[-1]
        else np.empty(0, dtype=np.int64)
    )
    wrong_items = (
        np.concatenate(wrong_rows).astype(np.int64, copy=False)
        if wrong_offsets[-1]
        else np.empty(0, dtype=np.int64)
    )
    true_source = np.asarray([true_count_by_index[index] for index in targets], dtype=np.int32)
    wrong_source = np.asarray([wrong_count_by_index[index] for index in targets], dtype=np.int32)
    true_authenticated = np.diff(np.asarray(true_offsets, dtype=np.int64))
    wrong_authenticated = np.diff(np.asarray(wrong_offsets, dtype=np.int64))
    true_fraction = true_authenticated / np.maximum(true_source, 1)
    wrong_fraction = wrong_authenticated / np.maximum(wrong_source, 1)
    return AuthenticationRows(
        request_indices=np.asarray(targets, dtype=np.int64),
        true_offsets=np.asarray(true_offsets, dtype=np.int64),
        true_items=true_items,
        wrong_offsets=np.asarray(wrong_offsets, dtype=np.int64),
        wrong_items=wrong_items,
        profile_sizes=np.asarray([profile_by_index[index] for index in targets], dtype=np.int32),
        true_source_counts=true_source,
        wrong_source_counts=wrong_source,
        audit={
            "packed_requests_traversed": len(data.request_ids),
            "target_requests": len(targets),
            "timestamp_groups": timestamp_groups,
            "same_timestamp_multi_request_groups": same_timestamp_multi_request_groups,
            "same_timestamp_score_before_update": True,
            "true_nonempty_requests": int((true_authenticated > 0).sum()),
            "wrong_nonempty_requests": int((wrong_authenticated > 0).sum()),
            "true_authenticated_events": int(true_authenticated.sum()),
            "wrong_authenticated_events": int(wrong_authenticated.sum()),
            "true_authenticity_mean": float(true_fraction.mean()),
            "wrong_authenticity_mean": float(wrong_fraction.mean()),
            "true_greater_than_wrong_requests": int((true_fraction > wrong_fraction).sum()),
            "wrong_greater_than_true_requests": int((wrong_fraction > true_fraction).sum()),
        },
    )
