from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.data import atomic_json, sha256_file  # noqa: E402
from probe.protocol import load_config  # noqa: E402


def main() -> None:
    config_path = SYSTEM_ROOT / "configs/signal_gate.yaml"
    config = load_config(config_path)
    lock_path = REPO_ROOT / config["paths"]["proposal_lock"]
    if lock_path.exists():
        raise FileExistsError(lock_path)
    candidate_files = [
        "README.md",
        "environment.txt",
        "configs/signal_gate.yaml",
        "model/__init__.py",
        "model/behavioral_semantic.py",
        "probe/__init__.py",
        "probe/data.py",
        "probe/materialize_selection.py",
        "probe/protocol.py",
        "probe/materialize_g0.py",
        "probe/metrics.py",
        "probe/run_signal_gate.py",
        "probe/freeze_lock.py",
        "tests/test_model.py",
        "tests/test_protocol.py",
        "notes/proposal.md",
        "notes/mechanism_fingerprint.md",
        "notes/nearest_neighbors.md",
        "notes/signal_gate_protocol.md",
    ]
    selection = json.loads((REPO_ROOT / config["paths"]["selection"]).read_text(encoding="utf-8"))
    external = {
        config["paths"]["selection"]: sha256_file(REPO_ROOT / config["paths"]["selection"]),
        config["paths"]["packed_manifest"]: sha256_file(REPO_ROOT / config["paths"]["packed_manifest"]),
        config["paths"]["label_free_request_metadata"]: sha256_file(REPO_ROOT / config["paths"]["label_free_request_metadata"]),
        config["paths"]["raw_item_embeddings"]: sha256_file(REPO_ROOT / config["paths"]["raw_item_embeddings"]),
        config["paths"]["train_candidate_labels"]: sha256_file(REPO_ROOT / config["paths"]["train_candidate_labels"]),
        config["paths"]["candidate_manifest"]: sha256_file(REPO_ROOT / config["paths"]["candidate_manifest"]),
    }
    for relative, digest in selection["provenance"]["selection_file_sha256"].items():
        if sha256_file(REPO_ROOT / relative) != digest:
            raise RuntimeError(f"C46 selection provenance changed: {relative}")
        external[relative] = digest
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, check=True, capture_output=True, text=True).stdout)
    value = {
        "candidate_id": "c46",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_c46_source_labels_features_scores_or_outcome",
        "git_commit": commit,
        "git_dirty": dirty,
        "environment": "/data/gkl/conda_envs/myrec-c37",
        "candidate_files": {name: sha256_file(SYSTEM_ROOT / name) for name in candidate_files},
        "external_inputs": external,
        "declarations": {
            "source_labels_opened": False,
            "A_features_scores_opened": False,
            "A_labels_opened": False,
            "dev_test_qrels_read": False,
        },
    }
    atomic_json(lock_path, value)
    print(json.dumps({"candidate_id": "c46", "status": value["status"], "lock": str(lock_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
