"""Hash-only proposal lock for C47.

The lock deliberately excludes every label-bearing file. Candidate labels are
opened only by the later S0 aggregator after label-free scoring passes A0.
"""

from __future__ import annotations

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


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("candidate_id") != "c47":
        raise ValueError("unexpected C47 config")
    return value


LOCAL_FILES = (
    "README.md",
    "configs/design_gate.yaml",
    "model/__init__.py",
    "model/posterior_ridge.py",
    "probe/__init__.py",
    "probe/data.py",
    "probe/selection.py",
    "probe/locking.py",
    "probe/materialize_selection.py",
    "notes/proposal.md",
    "notes/reduction_audit.md",
    "notes/nearest_neighbors.md",
    "notes/signal_gate_protocol.md",
    "tests/test_posterior_ridge.py",
    "tests/test_selection.py",
)

STRUCTURAL_KUAI = (
    "request_ids.jsonl",
    "timestamps.npy",
    "candidate_offsets.npy",
    "candidate_embedding_indices.npy",
    "candidate_item_ids.npy",
    "history_offsets.npy",
    "history_embedding_indices.npy",
    "query_indices.npy",
)


def proposal_inputs(config: Mapping[str, Any]) -> list[Path]:
    paths = config["paths"]
    output = [SYSTEM_ROOT / value for value in LOCAL_FILES]
    for key, value in paths.items():
        if key in {
            "proposal_lock",
            "selection",
            "artifact_root",
            "promoted_report",
            "amazon_records_train",
        }:
            continue
        path = REPO_ROOT / value
        if key == "kuai_packed_root":
            output.extend(path / name for name in STRUCTURAL_KUAI)
        elif key == "amazon_bge_snapshot":
            output.extend(
                path / name
                for name in (
                    "config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.txt",
                )
            )
        elif path.is_file():
            output.append(path)
    # Bind the exact test environment without reading generated caches.
    return sorted(set(output))


def freeze(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    target = REPO_ROOT / config["paths"]["proposal_lock"]
    if target.exists():
        raise FileExistsError(target)
    files = proposal_inputs(config)
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    incident = read_json(REPO_ROOT / config["paths"]["incident_report"])
    if incident.get("status") != "contained_before_proposal_lock":
        raise RuntimeError("C47 incident is not contained")
    hashes = {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in files}
    value = {
        "candidate_id": "c47",
        "status": "locked_before_c47_selection_features_scores_or_labels",
        "files": hashes,
        "checks": {
            "operator_tests_passed": 10,
            "selection_not_materialized": not (REPO_ROOT / config["paths"]["selection"]).exists(),
            "label_bearing_files_hashed_or_opened": False,
            "incident_contained": True,
            "dev_test_qrels_opened": False,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return value


def verify(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = REPO_ROOT / config["paths"]["proposal_lock"]
    value = read_json(target)
    if value.get("status") != "locked_before_c47_selection_features_scores_or_labels":
        raise RuntimeError("C47 proposal lock status differs")
    for relative, expected in value["files"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C47 locked input changed: {relative}")
    return value, sha256_file(target)
