"""C47 Amazon feature adapter around the already-audited C38 BGE encoder."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Mapping

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C38_ROOT = REPO_ROOT / "systems/38_cross_domain_global_tangent_transfer"
if str(C38_ROOT) not in sys.path:
    sys.path.insert(0, str(C38_ROOT))

from train.features import (  # noqa: E402
    collect_label_free_features,
    encode_feature_shard,
    finalize_embeddings,
)
from train.selection import (  # noqa: E402
    candidate_key_sha256,
    load_blind_records,
    sha256_file,
    write_json,
)


def build_amazon_adapter(config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    selection_path = REPO_ROOT / paths["selection"]
    if sha256_file(selection_path) != config["integrity"]["selection_sha256"]:
        raise RuntimeError("C47 selection changed before Amazon feature collection")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    records_path = REPO_ROOT / paths["amazon_records_train_blind"]
    records = load_blind_records(records_path)
    indices = [int(value) for value in selection["roles"]["amazon_internal_A"]["indices"]]
    donors = [
        int(value)
        for value in selection["wrong_history_donors"]["amazon_internal_A"]["indices"]
    ]
    if len(indices) != len(donors) or len(indices) != 300:
        raise ValueError("C47 Amazon A/donor cardinality differs")
    adapter = {
        "candidate_id": "c47_amazon_feature_adapter",
        "records_sha256": sha256_file(records_path),
        "roles": {
            "internal_A": {
                "indices": indices,
                "candidate_key_sha256": candidate_key_sha256(records, indices),
            }
        },
        "wrong_donors": {
            str(index): {"donor_index": donor}
            for index, donor in zip(indices, donors)
        },
        "label_access": {
            "records_train_blind_opened": True,
            "records_train_labels_opened": False,
            "dev_test_records_labels_qrels_opened": False,
        },
    }
    target = REPO_ROOT / paths["amazon_adapter_selection"]
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous != adapter:
            raise RuntimeError("existing C47 Amazon adapter differs")
    else:
        write_json(target, adapter)
    return adapter


def collect_amazon(config: Mapping[str, Any]) -> dict[str, Any]:
    build_amazon_adapter(config)
    paths = config["paths"]
    return collect_label_free_features(
        records_path=REPO_ROOT / paths["amazon_records_train_blind"],
        selection_path=REPO_ROOT / paths["amazon_adapter_selection"],
        output_root=REPO_ROOT / paths["amazon_feature_root"],
        roles=("internal_A",),
    )


def encode_amazon(
    config: Mapping[str, Any], *, shard_id: int, device: str
) -> dict[str, Any]:
    paths, encoding = config["paths"], config["encoding"]
    root = REPO_ROOT / paths["amazon_feature_root"]
    return encode_feature_shard(
        feature_root=root,
        snapshot=REPO_ROOT / paths["amazon_bge_snapshot"],
        output_path=root / f"embedding_shard_{shard_id}.npz",
        shard_id=shard_id,
        num_shards=int(encoding["num_shards"]),
        device=device,
        item_batch_size=int(encoding["item_batch_size"]),
        query_batch_size=int(encoding["query_batch_size"]),
        item_max_length=int(encoding["item_max_length"]),
        query_max_length=int(encoding["query_max_length"]),
        query_prefix=str(encoding["query_prefix"]),
    )


def finalize_amazon(config: Mapping[str, Any]) -> dict[str, Any]:
    paths, encoding = config["paths"], config["encoding"]
    root = REPO_ROOT / paths["amazon_feature_root"]
    return finalize_embeddings(
        feature_root=root,
        shard_paths=[
            root / f"embedding_shard_{shard}.npz"
            for shard in range(int(encoding["num_shards"]))
        ],
        embedding_dim=int(encoding["embedding_dim"]),
        repeat_boost=0.0,
    )
