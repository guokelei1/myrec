"""Run the already-audited C38 label-free BGE feature pipeline for C41."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C38_ROOT = REPO_ROOT / "systems" / "38_cross_domain_global_tangent_transfer"
sys.path.insert(0, str(C38_ROOT))

from train.features import (  # noqa: E402
    collect_label_free_features,
    encode_feature_shard,
    finalize_embeddings,
)


def load_config(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C41 config must be an object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("collect", "encode", "finalize"), required=True)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    paths = config["paths"]
    if args.stage == "collect":
        value = collect_label_free_features(
            records_path=paths["records_train_blind"],
            selection_path=paths["selection"],
            output_root=paths["feature_root"],
            roles=("fit", "internal_A"),
        )
    elif args.stage == "encode":
        if args.shard_id is None:
            raise ValueError("C41 encode requires --shard-id")
        count = int(config["encoding"]["num_shards"])
        value = encode_feature_shard(
            feature_root=paths["feature_root"],
            snapshot=paths["bge_snapshot"],
            output_path=Path(paths["feature_root"])
            / f"embedding_shard_{args.shard_id}.npz",
            shard_id=args.shard_id,
            num_shards=count,
            device=args.device,
            item_batch_size=int(config["encoding"]["item_batch_size"]),
            query_batch_size=int(config["encoding"]["query_batch_size"]),
            item_max_length=int(config["encoding"]["item_max_length"]),
            query_max_length=int(config["encoding"]["query_max_length"]),
            query_prefix=str(config["encoding"]["query_prefix"]),
        )
    else:
        count = int(config["encoding"]["num_shards"])
        value = finalize_embeddings(
            feature_root=paths["feature_root"],
            shard_paths=[
                Path(paths["feature_root"]) / f"embedding_shard_{index}.npz"
                for index in range(count)
            ],
            embedding_dim=int(config["model"]["embedding_dim"]),
            repeat_boost=float(config["base"]["repeat_boost"]),
        )
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
