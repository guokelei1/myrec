from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

from probe.data import sha256_file


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if value.get("candidate_id") != "c46":
        raise ValueError("C46 config identity differs")
    return value


def verify_proposal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_c46_source_labels_features_scores_or_outcome":
        raise PermissionError("C46 proposal lock status differs")
    expected = {
        "A_features_scores_opened": False,
        "A_labels_opened": False,
        "dev_test_qrels_read": False,
        "source_labels_opened": False,
    }
    if value.get("declarations") != expected:
        raise PermissionError("C46 proposal declarations differ")
    for relative, digest in value["candidate_files"].items():
        if sha256_file(SYSTEM_ROOT / relative) != digest:
            raise RuntimeError(f"locked C46 candidate file changed: {relative}")
    for relative, digest in value["external_inputs"].items():
        if sha256_file(REPO_ROOT / relative) != digest:
            raise RuntimeError(f"locked C46 external input changed: {relative}")
    return value, sha256_file(path)


def state_sha256(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
