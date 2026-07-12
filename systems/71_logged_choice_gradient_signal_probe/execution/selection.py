from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def load_json_map(path: str | Path) -> dict[str, int]:
    return {str(key): int(value) for key, value in json.loads(Path(path).read_text()).items()}


def load_packed_request_ids(path: str | Path) -> set[str]:
    return {
        str(json.loads(line)["request_id"])
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line
    }


def structural_record(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "user_id": str(row["user_id"]),
        "ts": int(row["ts"]),
        "query": str(row.get("query", "")),
        "candidate_ids": [str(candidate["item_id"]) for candidate in row.get("candidates", [])],
        "history": [
            {"ts": int(event["ts"]), "item_id": str(event["item_id"])}
            for event in row.get("history", [])
        ],
    }


def iter_structural_records(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line:
                yield structural_record(json.loads(line))


def count_bin(value: int, bounds: list[int]) -> int:
    for index, bound in enumerate(bounds):
        if value <= int(bound):
            return index
    return len(bounds)


def candidate_key_sha256(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        row = [str(record["request_id"]), *[str(value) for value in record["candidate_ids"]]]
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def target_hash(seed: int, request_id: str) -> bytes:
    return hashlib.sha256(f"c71-target:{seed}:{request_id}".encode()).digest()


def normalized_query(query_embeddings: np.ndarray, index: int) -> np.ndarray:
    value = np.asarray(query_embeddings[int(index)], dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1e-8)


def materialize_selection(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    paths = config["paths"]
    settings = config["selection"]
    item_map = load_json_map(repo_root / paths["item_id_map"])
    query_map = load_json_map(repo_root / paths["request_query_map"])
    packed = load_packed_request_ids(repo_root / paths["packed_request_ids"])
    records_path = repo_root / paths["records_train"]

    preliminary: dict[str, dict[str, Any]] = {}
    needed: dict[tuple[str, int], set[str]] = defaultdict(set)
    for record in iter_structural_records(records_path):
        request_id = record["request_id"]
        if request_id in packed or request_id not in query_map:
            continue
        history_ids = {row["item_id"] for row in record["history"]}
        candidate_ids = set(record["candidate_ids"])
        if not record["history"] or history_ids & candidate_ids:
            continue
        if not all(value in item_map for value in (*history_ids, *candidate_ids)):
            continue
        if any(row["ts"] >= record["ts"] for row in record["history"]):
            continue
        preliminary[request_id] = record
        for event in record["history"]:
            needed[(record["user_id"], int(event["ts"]))].add(event["item_id"])

    source_options: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for record in iter_structural_records(records_path):
        key = (record["user_id"], int(record["ts"]))
        selected_items = needed.get(key)
        if not selected_items or record["request_id"] not in query_map:
            continue
        if not all(value in item_map for value in record["candidate_ids"]):
            continue
        candidates = set(record["candidate_ids"])
        for item_id in sorted(selected_items):
            if item_id in candidates:
                source_options[(key[0], key[1], item_id)].append(record["request_id"])

    eligible: list[dict[str, Any]] = []
    for request_id, record in preliminary.items():
        episodes = []
        for event in record["history"]:
            options = source_options.get((record["user_id"], int(event["ts"]), event["item_id"]), [])
            if not options:
                continue
            episodes.append(
                {
                    "history_ts": int(event["ts"]),
                    "selected_item_id": event["item_id"],
                    "source_request_id": min(options),
                }
            )
        episodes.sort(key=lambda row: (row["history_ts"], row["selected_item_id"], row["source_request_id"]))
        if len(episodes) < int(settings["minimum_linked_episodes"]):
            continue
        value = dict(record)
        value["query_embedding_index"] = int(query_map[request_id])
        value["episodes"] = episodes
        eligible.append(value)

    seed = int(settings["seed"])
    eligible.sort(key=lambda row: target_hash(seed, row["request_id"]))
    target_count = int(settings["target_requests"])
    donor_pool_count = int(settings["donor_pool_requests"])
    if len(eligible) < target_count + donor_pool_count:
        raise RuntimeError("C71 eligible unpacked pool is too small")
    targets = eligible[:target_count]
    donor_pool = eligible[target_count : target_count + donor_pool_count]

    query_embeddings = np.load(repo_root / paths["query_embeddings"], mmap_mode="r")
    episode_bounds = [int(value) for value in settings["episode_count_bins"]]
    candidate_bounds = [int(value) for value in settings["candidate_count_bins"]]
    donor_vectors = {
        row["request_id"]: normalized_query(query_embeddings, row["query_embedding_index"])
        for row in donor_pool
    }
    donor_rows = {row["request_id"]: row for row in donor_pool}
    wrong_donors = []
    selected_donor_ids: set[str] = set()
    for target in targets:
        target_vector = normalized_query(query_embeddings, target["query_embedding_index"])
        episode_bin = count_bin(len(target["episodes"]), episode_bounds)
        candidate_bin = count_bin(len(target["candidate_ids"]), candidate_bounds)
        target_candidates = set(target["candidate_ids"])
        choices = []
        for donor in donor_pool:
            if donor["user_id"] == target["user_id"]:
                continue
            if count_bin(len(donor["episodes"]), episode_bounds) != episode_bin:
                continue
            if count_bin(len(donor["candidate_ids"]), candidate_bounds) != candidate_bin:
                continue
            if target_candidates & {row["selected_item_id"] for row in donor["episodes"]}:
                continue
            cosine = float(target_vector @ donor_vectors[donor["request_id"]])
            choices.append((-cosine, donor["request_id"]))
        if not choices:
            raise RuntimeError(f"C71 no exact-bin wrong donor for {target['request_id']}")
        _, donor_id = min(choices)
        selected_donor_ids.add(donor_id)
        donor = donor_rows[donor_id]
        wrong_donors.append(
            {
                "target_request_id": target["request_id"],
                "donor_request_id": donor_id,
                "query_cosine": float(target_vector @ donor_vectors[donor_id]),
                "episode_count_bin": episode_bin,
                "candidate_count_bin": candidate_bin,
            }
        )

    selected_donors = [donor_rows[value] for value in sorted(selected_donor_ids)]
    target_hash_value = candidate_key_sha256(targets)
    checks = {
        "target_count_exact": len(targets) == target_count,
        "target_unique": len({row["request_id"] for row in targets}) == target_count,
        "target_outside_packed_pool": all(row["request_id"] not in packed for row in targets),
        "target_strict_nonrepeat": all(
            not ({event["item_id"] for event in row["history"]} & set(row["candidate_ids"]))
            for row in targets
        ),
        "minimum_linked_episodes": all(
            len(row["episodes"]) >= int(settings["minimum_linked_episodes"]) for row in targets
        ),
        "episodes_strictly_past": all(
            episode["history_ts"] < row["ts"] for row in targets for episode in row["episodes"]
        ),
        "wrong_donors_complete": len(wrong_donors) == target_count,
        "wrong_different_user": all(
            donor_rows[pair["donor_request_id"]]["user_id"]
            != preliminary[pair["target_request_id"]]["user_id"]
            for pair in wrong_donors
        ),
        "labels_not_accessed": True,
        "dev_test_qrels_not_accessed": True,
    }
    return {
        "schema": "myrec.c71.selection.v1",
        "candidate_id": "c71",
        "seed": seed,
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "pool_counts": {
            "historical_packed_requests_excluded": len(packed),
            "unpacked_strict_nonrepeat_embedding_complete": len(preliminary),
            "eligible_linked_episode_requests": len(eligible),
            "donor_pool": len(donor_pool),
        },
        "candidate_key_sha256": target_hash_value,
        "targets": targets,
        "wrong_donors": wrong_donors,
        "selected_donors": selected_donors,
        "source_request_ids": sorted(
            {
                episode["source_request_id"]
                for row in [*targets, *selected_donors]
                for episode in row["episodes"]
            }
        ),
        "isolation": {
            "target_labels_opened": False,
            "source_episode_labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
