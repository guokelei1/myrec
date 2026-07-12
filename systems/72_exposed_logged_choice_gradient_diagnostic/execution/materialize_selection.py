from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import atomic_json, load_config, sha256_file, timestamp, verify_inputs, verify_proposal_lock  # noqa: E402


def load_map(path: Path) -> dict[str, int]:
    return {str(key): int(value) for key, value in json.loads(path.read_text()).items()}


def structural(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "request_id": str(row["request_id"]),
        "user_id": str(row["user_id"]),
        "ts": int(row["ts"]),
        "query": str(row.get("query", "")),
        "candidate_ids": [str(value["item_id"]) for value in row["candidates"]],
        "history": [
            {"ts": int(value["ts"]), "item_id": str(value["item_id"])}
            for value in row["history"]
        ],
    }


def records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield structural(json.loads(line))


def count_bin(value: int, bounds: list[int]) -> int:
    for index, bound in enumerate(bounds):
        if value <= int(bound):
            return index
    return len(bounds)


def normalized(values: np.ndarray, index: int) -> np.ndarray:
    row = np.asarray(values[int(index)], dtype=np.float32)
    return row / max(float(np.linalg.norm(row)), 1e-8)


def candidate_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        value = [row["request_id"], *row["candidate_ids"]]
        digest.update(json.dumps(value, separators=(",", ":")).encode()); digest.update(b"\n")
    return digest.hexdigest()


def main() -> None:
    config = load_config(SYSTEM_ROOT / "configs/diagnostic.yaml")
    verify_inputs(config)
    _, proposal_hash = verify_proposal_lock(config)
    paths, settings = config["paths"], config["selection"]
    item_map = load_map(REPO_ROOT / paths["item_id_map"])
    query_map = load_map(REPO_ROOT / paths["request_query_map"])
    packed_ids = [
        str(json.loads(line)["request_id"])
        for line in (REPO_ROOT / paths["packed_request_ids"]).read_text().splitlines()
        if line
    ]
    c47 = json.loads((REPO_ROOT / paths["c47_selection"]).read_text())
    fit_indices = [int(value) for value in c47["roles"]["kuai_fit"]["indices"]]
    allowed = {packed_ids[index] for index in fit_indices}
    preliminary: dict[str, dict[str, Any]] = {}
    needed: dict[tuple[str, int], set[str]] = defaultdict(set)
    records_path = REPO_ROOT / paths["records_train"]
    for row in records(records_path):
        request_id = row["request_id"]
        if request_id not in allowed or request_id not in query_map or not row["history"]:
            continue
        history_ids = {value["item_id"] for value in row["history"]}
        candidate_ids = set(row["candidate_ids"])
        if history_ids & candidate_ids:
            continue
        if not all(value in item_map for value in (*history_ids, *candidate_ids)):
            continue
        preliminary[request_id] = row
        for event in row["history"]:
            needed[(row["user_id"], event["ts"])].add(event["item_id"])
    options: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for row in records(records_path):
        key = (row["user_id"], row["ts"])
        selected = needed.get(key)
        if not selected or row["request_id"] not in query_map:
            continue
        if not all(value in item_map for value in row["candidate_ids"]):
            continue
        candidates = set(row["candidate_ids"])
        for item_id in selected:
            if item_id in candidates:
                options[(key[0], key[1], item_id)].append(row["request_id"])
    eligible = []
    for request_id, row in preliminary.items():
        episodes = []
        for event in row["history"]:
            source = options.get((row["user_id"], event["ts"], event["item_id"]), [])
            if source:
                episodes.append({
                    "history_ts": event["ts"],
                    "selected_item_id": event["item_id"],
                    "source_request_id": min(source),
                })
        episodes.sort(key=lambda value: (value["history_ts"], value["selected_item_id"], value["source_request_id"]))
        if len(episodes) < int(settings["minimum_linked_episodes"]):
            continue
        value = dict(row); value["query_embedding_index"] = query_map[request_id]; value["episodes"] = episodes
        eligible.append(value)
    seed = int(settings["seed"])
    eligible.sort(key=lambda row: hashlib.sha256(f"c72-target:{seed}:{row['request_id']}".encode()).digest())
    target_count = int(settings["target_requests"]); donor_count = int(settings["donor_pool_requests"])
    if len(eligible) < target_count + donor_count:
        raise RuntimeError("C72 eligible exposed fit pool too small")
    targets = eligible[:target_count]; donor_pool = eligible[target_count:target_count + donor_count]
    query_embeddings = np.load(REPO_ROOT / paths["query_embeddings"], mmap_mode="r")
    donor_vectors = {row["request_id"]: normalized(query_embeddings, row["query_embedding_index"]) for row in donor_pool}
    donor_rows = {row["request_id"]: row for row in donor_pool}
    eb = [int(value) for value in settings["episode_count_bins"]]; cb = [int(value) for value in settings["candidate_count_bins"]]
    wrong = []; chosen = set()
    for target in targets:
        vector = normalized(query_embeddings, target["query_embedding_index"])
        ebin = count_bin(len(target["episodes"]), eb); cbin = count_bin(len(target["candidate_ids"]), cb)
        candidate_ids = set(target["candidate_ids"]); choices = []
        for donor in donor_pool:
            if donor["user_id"] == target["user_id"] or count_bin(len(donor["episodes"]), eb) != ebin or count_bin(len(donor["candidate_ids"]), cb) != cbin:
                continue
            if candidate_ids & {episode["selected_item_id"] for episode in donor["episodes"]}:
                continue
            cosine = float(vector @ donor_vectors[donor["request_id"]]); choices.append((-cosine, donor["request_id"]))
        if not choices:
            raise RuntimeError(f"C72 no exact donor for {target['request_id']}")
        _, donor_id = min(choices); chosen.add(donor_id)
        wrong.append({"target_request_id": target["request_id"], "donor_request_id": donor_id, "query_cosine": float(vector @ donor_vectors[donor_id]), "episode_count_bin": ebin, "candidate_count_bin": cbin})
    selected_donors = [donor_rows[value] for value in sorted(chosen)]
    checks = {
        "source_role_exact": len(allowed) == 6000,
        "target_count_exact": len(targets) == target_count,
        "target_within_c47_fit": all(row["request_id"] in allowed for row in targets),
        "target_strict_nonrepeat": all(not ({event["item_id"] for event in row["history"]} & set(row["candidate_ids"])) for row in targets),
        "minimum_linked_episodes": all(len(row["episodes"]) >= int(settings["minimum_linked_episodes"]) for row in targets),
        "wrong_donors_complete": len(wrong) == target_count,
        "labels_already_exposed_but_not_read_for_selection": True,
        "dev_test_qrels_not_accessed": True,
    }
    value = {
        "schema": "myrec.c72.selection.v1", "candidate_id": "c72", "created_at": timestamp(), "proposal_lock_sha256": proposal_hash,
        "status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "pool_counts": {"c47_fit": len(allowed), "strict_nonrepeat_complete": len(preliminary), "eligible_linked": len(eligible), "donor_pool": len(donor_pool)},
        "candidate_key_sha256": candidate_hash(targets), "targets": targets, "wrong_donors": wrong, "selected_donors": selected_donors,
        "source_request_ids": sorted({episode["source_request_id"] for row in [*targets, *selected_donors] for episode in row["episodes"]}),
        "claim_boundary": {"fresh": False, "formulation_only": True, "dev_test_qrels": False},
    }
    target_path = REPO_ROOT / paths["selection"]
    atomic_json(target_path, value)
    print(target_path.relative_to(REPO_ROOT)); print(sha256_file(target_path)); print(value["status"])


if __name__ == "__main__":
    main()
