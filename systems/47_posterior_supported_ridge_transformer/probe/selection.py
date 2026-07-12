"""Label-free, cross-domain C47 role selection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


OPENED_KUAI_A = (28, 29, 31, 32, 33, 35, 37, 43, 46)


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def stable_key(seed: int, role: str, request_id: str) -> bytes:
    return hashlib.sha256(f"c47:{seed}:{role}:{request_id}".encode()).digest()


def request_hash(request_ids: Sequence[str], indices: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        digest.update(str(request_ids[int(index)]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def compact_index_hash(indices: Sequence[int]) -> str:
    payload = json.dumps(sorted(map(int, indices)), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _role(selection: Mapping[str, Any], name: str) -> list[int]:
    return [int(value) for value in selection["roles"][name]["indices"]]


def kuai_roles(
    *,
    selections: Mapping[int, Mapping[str, Any]],
    request_ids: Sequence[str],
    histories: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    seed: int,
    fit_count: int,
    a_count: int,
    reserve_count: int,
    incident_hash: str,
) -> dict[str, Any]:
    incident = set()
    for role in ("internal_A", "delayed_B", "escrow"):
        incident.update(_role(selections[26], role))
    opened = set()
    for candidate in OPENED_KUAI_A:
        opened.update(_role(selections[candidate], "internal_A"))
    incident.difference_update(opened)
    if compact_index_hash(incident) != incident_hash:
        raise RuntimeError("C47 incident scope changed")

    pool = set(_role(selections[34], "internal_A")) | set(
        _role(selections[36], "internal_A")
    )
    pool.difference_update(opened)
    pool.difference_update(incident)
    eligible = [
        index
        for index in pool
        if len(histories[index])
        and not (set(map(int, histories[index])) & set(map(int, candidates[index])))
    ]
    eligible.sort(key=lambda i: (stable_key(seed, "kuai-A", request_ids[i]), i))
    if len(eligible) != a_count + reserve_count:
        raise RuntimeError(f"C47 Kuai blind pool changed: {len(eligible)}")
    internal_a = eligible[:a_count]
    reserve = eligible[a_count:]

    donor_by_target: dict[int, int] = {}
    for candidate in (34, 36):
        targets = _role(selections[candidate], "internal_A")
        donors = [
            int(value)
            for value in selections[candidate]["wrong_history_donors"]["internal_A"]["indices"]
        ]
        if len(targets) != len(donors):
            raise RuntimeError(f"C47 inherited C{candidate} donors differ")
        donor_by_target.update(zip(targets, donors))
    wrong = [donor_by_target[index] for index in internal_a]

    fit_pool = _role(selections[34], "fit")
    fit_pool.sort(key=lambda i: (stable_key(seed, "kuai-fit", request_ids[i]), i))
    fit = fit_pool[:fit_count]
    if len(fit) != fit_count:
        raise RuntimeError("C47 Kuai fit pool too small")
    if set(fit) & set(eligible):
        raise RuntimeError("C47 Kuai fit/outcome overlap")

    return {
        "fit": fit,
        "internal_A": internal_a,
        "reserve": reserve,
        "wrong_history_donors": wrong,
        "blind_pool_count": len(eligible),
        "blind_pool_sha256": compact_index_hash(eligible),
        "incident_count": len(incident),
    }



def length_bin(length: int, edges: Sequence[int]) -> int:
    for edge in edges:
        if length <= edge:
            return int(edge)
    return int(edges[-1])


def amazon_roles(
    *,
    c38: Mapping[str, Any],
    c39: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    seed: int,
    a_count: int,
    reserve_count: int,
    edges: Sequence[int],
) -> dict[str, Any]:
    pool = [int(value) for value in c39["reserve_indices"]]
    pool.sort(key=lambda i: (stable_key(seed, "amazon-A", records[i]["request_id"]), i))
    if len(pool) != a_count + reserve_count:
        raise RuntimeError(f"C47 Amazon blind pool changed: {len(pool)}")
    internal_a, reserve = pool[:a_count], pool[a_count:]
    fit = _role(c38, "fit")

    by_bin: dict[int, list[int]] = {}
    excluded = set(fit) | set(pool)
    for index, record in enumerate(records):
        if index in excluded or not record["history"]:
            continue
        by_bin.setdefault(length_bin(len(record["history"]), edges), []).append(index)
    for values in by_bin.values():
        values.sort(key=lambda i: (stable_key(seed, "amazon-donor", records[i]["request_id"]), i))

    donors = []
    for target in internal_a:
        recipient = records[target]
        candidate_ids = {str(item["item_id"]) for item in recipient["candidates"]}
        choices = by_bin[length_bin(len(recipient["history"]), edges)]
        start = int.from_bytes(stable_key(seed, "amazon-donor-start", recipient["request_id"])[:8], "big") % len(choices)
        donor = None
        for offset in range(len(choices)):
            value = choices[(start + offset) % len(choices)]
            row = records[value]
            if str(row["user_id"]) == str(recipient["user_id"]):
                continue
            if candidate_ids & {str(item["item_id"]) for item in row["history"]}:
                continue
            donor = value
            break
        if donor is None:
            raise RuntimeError(f"no C47 Amazon donor for {target}")
        donors.append(donor)
    return {
        "fit": fit,
        "internal_A": internal_a,
        "reserve": reserve,
        "wrong_history_donors": donors,
        "blind_pool_sha256": compact_index_hash(pool),
    }
