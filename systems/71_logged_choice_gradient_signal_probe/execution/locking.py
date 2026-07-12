from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def verify_registered_inputs(config: Mapping[str, Any]) -> None:
    pairs = (
        ("records_train", "records_train_sha256"),
        ("standardized_manifest", "standardized_manifest_sha256"),
        ("candidate_manifest", "candidate_manifest_sha256"),
        ("item_id_map", "item_id_map_sha256"),
        ("item_embeddings", "item_embeddings_sha256"),
        ("request_query_map", "request_query_map_sha256"),
        ("query_embeddings", "query_embeddings_sha256"),
        ("packed_request_ids", "packed_request_ids_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("c70_coverage_report", "c70_coverage_report_sha256"),
    )
    for path_key, hash_key in pairs:
        actual = sha256_file(REPO_ROOT / config["paths"][path_key])
        if actual != str(config["integrity"][hash_key]):
            raise RuntimeError(f"C71 registered input changed: {path_key}")


def proposal_sources(config: Mapping[str, Any]) -> dict[str, Path]:
    return {
        "config": SYSTEM_ROOT / "configs/signal_gate.yaml",
        "proposal": REPO_ROOT / config["paths"]["proposal"],
        "nearest_neighbors": REPO_ROOT / config["paths"]["nearest_neighbors"],
        "preimplementation_review": REPO_ROOT / config["paths"]["preimplementation_review"],
        "readme": SYSTEM_ROOT / "README.md",
        "c70_coverage_report": REPO_ROOT / config["paths"]["c70_coverage_report"],
    }


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    for name, source in proposal_sources(config).items():
        if sha256_file(source) != value["design_sha256"][name]:
            raise RuntimeError(f"C71 proposal source changed: {name}")
    return value, sha256_file(path)


def verify_execution_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _, proposal_hash = verify_proposal_lock(config)
    path = REPO_ROOT / config["paths"]["execution_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value["proposal_lock_sha256"] != proposal_hash:
        raise RuntimeError("C71 execution lock points to another proposal")
    for relative, expected in value["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C71 locked source changed: {relative}")
    return value, sha256_file(path)
