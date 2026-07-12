"""Hash and write-once guards for C62."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    design_paths = {
        "architecture_coverage_audit": REPO_ROOT
        / "doc/dev_log/20260712_architecture_intervention_coverage_after_c61.md",
        "config": REPO_ROOT / "systems/62_write_once_preference_memory_transformer/configs/train_gate.yaml",
        "nearest_neighbors": REPO_ROOT / config["paths"]["nearest_neighbors"],
        "preimplementation_review": REPO_ROOT / config["paths"]["preimplementation_review"],
        "proposal": REPO_ROOT / config["paths"]["proposal"],
        "readme": SYSTEM_ROOT / "README.md",
    }
    for name, source in design_paths.items():
        expected = lock["design_sha256"][name]
        actual = sha256_file(source)
        if actual != expected:
            raise RuntimeError(f"C62 proposal source changed: {name}")
    predecessor = {
        "c60_report": REPO_ROOT / "reports/pps_c60_base_order_edge_transport_gate.json",
        "c61_report": REPO_ROOT / "reports/pps_c61_counterfactual_edge_likelihood_gate.json",
    }
    for name, source in predecessor.items():
        if sha256_file(source) != lock["predecessor_sha256"][name]:
            raise RuntimeError(f"C62 predecessor changed: {name}")
    selections = {
        "c26": REPO_ROOT / config["paths"]["c26_selection"],
        "c38": REPO_ROOT / config["paths"]["c38_selection"],
        "c39": REPO_ROOT / config["paths"]["c39_selection"],
    }
    for name, source in selections.items():
        if sha256_file(source) != lock["selection_sha256"][name]:
            raise RuntimeError(f"C62 predecessor selection changed: {name}")
    return lock, sha256_file(path)


def verify_g0_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    _, proposal_hash = verify_proposal_lock(config)
    path = REPO_ROOT / config["paths"]["g0_lock"]
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock["proposal_lock_sha256"] != proposal_hash:
        raise RuntimeError("C62 G0 lock points to another proposal")
    for relative, expected in lock["source_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"C62 G0 source changed: {relative}")
    return lock, sha256_file(path)
