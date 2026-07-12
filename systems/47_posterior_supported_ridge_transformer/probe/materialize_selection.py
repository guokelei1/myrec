"""Materialize C47 roles without opening any label-shaped input."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.locking import load_config, sha256_file, verify  # noqa: E402
from probe.selection import amazon_roles, compact_index_hash, kuai_roles, read_json  # noqa: E402
from probe.data import PackedTrain  # noqa: E402


def atomic_json(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def candidate_key_hash_kuai(data: PackedTrain, indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        row = [data.request_ids[index], *map(str, data.candidate_ids(index).tolist())]
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def candidate_key_hash_amazon(records: list[dict[str, Any]], indices: list[int]) -> str:
    digest = hashlib.sha256()
    for index in indices:
        row = [records[index]["request_id"], *[str(x["item_id"]) for x in records[index]["candidates"]]]
        digest.update(json.dumps(row, separators=(",", ":")).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def main(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    _, lock_hash = verify(config)
    paths, spec = config["paths"], config["selection"]
    data = PackedTrain(REPO_ROOT / paths["kuai_packed_root"])
    selections = {
        int(key.split("_c", 1)[1].split("_selection", 1)[0]): read_json(REPO_ROOT / value)
        for key, value in paths.items()
        if key.startswith("kuai_c") and key.endswith("_selection")
    }
    incident = read_json(REPO_ROOT / paths["incident_report"])
    kuai = kuai_roles(
        selections=selections,
        request_ids=data.request_ids,
        histories=[data.history(i) for i in range(len(data.request_ids))],
        candidates=[data.candidates(i) for i in range(len(data.request_ids))],
        seed=int(spec["seed"]),
        fit_count=int(spec["kuai_fit_requests"]),
        a_count=int(spec["kuai_internal_A_requests"]),
        reserve_count=int(spec["kuai_reserve_requests"]),
        incident_hash=incident["affected_sorted_indices_sha256"],
    )

    blind_path = REPO_ROOT / paths["amazon_records_train_blind"]
    records = [json.loads(line) for line in blind_path.read_text(encoding="utf-8").splitlines() if line]
    c38 = read_json(REPO_ROOT / paths["amazon_c38_selection"])
    c39 = read_json(REPO_ROOT / paths["amazon_c39_selection"])
    amazon = amazon_roles(
        c38=c38,
        c39=c39,
        records=records,
        seed=int(spec["seed"]),
        a_count=int(spec["amazon_internal_A_requests"]),
        reserve_count=int(spec["amazon_reserve_requests"]),
        edges=spec["history_length_bins"],
    )

    result = {
        "candidate_id": "c47",
        "selection_id": "c47_cross_domain_signal_v1",
        "proposal_lock_sha256": lock_hash,
        "seed": int(spec["seed"]),
        "roles": {
            "kuai_fit": {"indices": kuai["fit"]},
            "kuai_internal_A": {"indices": kuai["internal_A"]},
            "kuai_reserve": {"indices": kuai["reserve"]},
            "amazon_fit": {"indices": amazon["fit"]},
            "amazon_internal_A": {"indices": amazon["internal_A"]},
            "amazon_reserve": {"indices": amazon["reserve"]},
        },
        "wrong_history_donors": {
            "kuai_internal_A": {"indices": kuai["wrong_history_donors"]},
            "amazon_internal_A": {"indices": amazon["wrong_history_donors"]},
        },
        "candidate_key_sha256": {
            "kuai_internal_A": candidate_key_hash_kuai(data, kuai["internal_A"]),
            "amazon_internal_A": candidate_key_hash_amazon(records, amazon["internal_A"]),
        },
        "pools": {
            "kuai_blind_count": kuai["blind_pool_count"],
            "kuai_blind_sha256": kuai["blind_pool_sha256"],
            "amazon_blind_sha256": amazon["blind_pool_sha256"],
            "incident_count": kuai["incident_count"],
        },
        "source_hashes": {
            key: sha256_file(REPO_ROOT / value)
            for key, value in paths.items()
            if (key.endswith("_selection") or key in {"incident_report", "amazon_records_train_blind"})
        },
        "checks": {
            "labels_opened": False,
            "dev_test_qrels_opened": False,
            "kuai_strict_nonrepeat": all(
                len(data.history(i)) and not (set(data.history(i).tolist()) & set(data.candidates(i).tolist()))
                for i in kuai["internal_A"]
            ),
            "amazon_history_present": all(records[i]["history"] for i in amazon["internal_A"]),
            "roles_disjoint_within_domain": (
                not (set(kuai["fit"]) & (set(kuai["internal_A"]) | set(kuai["reserve"])))
                and not (set(kuai["internal_A"]) & set(kuai["reserve"]))
                and not (set(amazon["fit"]) & (set(amazon["internal_A"]) | set(amazon["reserve"])))
                and not (set(amazon["internal_A"]) & set(amazon["reserve"]))
            ),
            "counts_exact": (
                len(kuai["fit"]) == 6000 and len(kuai["internal_A"]) == 600 and len(kuai["reserve"]) == 395
                and len(amazon["fit"]) == 6000 and len(amazon["internal_A"]) == 300 and len(amazon["reserve"]) == 99
            ),
        },
    }
    if not all(result["checks"].values()):
        raise RuntimeError(result["checks"])
    output = REPO_ROOT / paths["selection"]
    atomic_json(output, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    value = main(args.config)
    print(json.dumps({"candidate_id": "c47", "status": "selection_materialized", "checks": value["checks"]}, sort_keys=True))
