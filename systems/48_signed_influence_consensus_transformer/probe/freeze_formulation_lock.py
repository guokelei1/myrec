"""Freeze C48 before computing its exposed-cohort formulation scores."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml

SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
sys.path.insert(0, str(C47_ROOT))

from probe.freeze_signal_lock import load_config as load_c47_config, verify_signal_lock  # noqa: E402
from probe.locking import sha256_file  # noqa: E402


LOCAL_FILES = (
    "README.md",
    "configs/formulation_gate.yaml",
    "model/__init__.py",
    "model/signed_consensus.py",
    "notes/nearest_neighbors.md",
    "notes/proposal.md",
    "notes/reduction_audit.md",
    "probe/__init__.py",
    "probe/freeze_formulation_lock.py",
    "probe/run_formulation_gate.py",
    "tests/test_signed_consensus.py",
)

C47_GENERATED_INPUTS = (
    "amazon_feature_selection.json",
    "amazon_fixed_score_report.json",
    "amazon_fixed_scores.npz",
    "kuai_fixed_score_report.json",
    "kuai_fixed_scores.npz",
    "a0_report.json",
)

C47_AMAZON_FEATURE_INPUTS = (
    "base_scores.npy",
    "candidate_item_positions.npy",
    "candidate_offsets.npy",
    "embedding_manifest.json",
    "feature_manifest.json",
    "feature_request_indices.npy",
    "item_embeddings.npy",
    "items.jsonl",
    "query_embeddings.npy",
    "requests.jsonl",
    "true_history_item_positions.npy",
    "true_history_offsets.npy",
    "wrong_history_item_positions.npy",
    "wrong_history_offsets.npy",
)


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("candidate_id") != "c48":
        raise ValueError("unexpected C48 config")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths = config["paths"]
    c47_config = load_c47_config(REPO_ROOT / paths["c47_config"])
    _, c47_lock_hash = verify_signal_lock(c47_config)
    c47_report = json.loads((REPO_ROOT / paths["c47_signal_report"]).read_text(encoding="utf-8"))
    if c47_report.get("status") != "failed_S0_terminal":
        raise RuntimeError("C48 requires the terminal C47 report")
    if c47_report.get("A_labels_opened_after_A0") is not True:
        raise PermissionError("C48 formulation cohort is not lawfully exposed")
    target = REPO_ROOT / paths["formulation_lock"]
    output_root = REPO_ROOT / paths["artifact_root"]
    promoted = REPO_ROOT / paths["promoted_report"]
    if target.exists() or output_root.exists() or promoted.exists():
        raise RuntimeError("C48 formulation output exists before lock")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "systems/48_signed_influence_consensus_transformer/tests/test_signed_consensus.py",
    ]
    test = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if test.returncode != 0 or "5 passed" not in test.stdout:
        raise RuntimeError(f"C48 tests failed: {test.stdout}\n{test.stderr}")
    local = [SYSTEM_ROOT / relative for relative in LOCAL_FILES]
    c47_artifact = REPO_ROOT / c47_config["paths"]["artifact_root"]
    external = [
        REPO_ROOT / paths["c47_config"],
        REPO_ROOT / paths["c47_signal_lock"],
        REPO_ROOT / paths["c47_signal_report"],
        *[c47_artifact / name for name in C47_GENERATED_INPUTS],
        *[
            c47_artifact / "amazon_features" / name
            for name in C47_AMAZON_FEATURE_INPUTS
        ],
    ]
    missing = [str(path) for path in [*local, *external] if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    value = {
        "candidate_id": "c48",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_c48_formulation_scores_on_exposed_c47_A",
        "c47_signal_execution_lock_sha256": c47_lock_hash,
        "local_files_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in local
        },
        "external_inputs_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in external
        },
        "declarations": {
            "operator_scores_opened": False,
            "c47_A_already_open_before_C48": True,
            "fresh_reserve_opened": False,
            "dev_test_records_labels_qrels_opened": False,
            "confirmatory_claim_allowed": False,
            "tests_passed": 5,
        },
        "test_stdout": test.stdout.strip(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    write_once(target, value)
    return value


def verify_formulation_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["formulation_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_c48_formulation_scores_on_exposed_c47_A":
        raise RuntimeError("C48 formulation lock status differs")
    for group in ("local_files_sha256", "external_inputs_sha256"):
        for relative, expected in value[group].items():
            source = REPO_ROOT / relative
            if not source.is_file() or sha256_file(source) != expected:
                raise RuntimeError(f"C48 locked input changed: {relative}")
    return value, sha256_file(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = freeze(args.config)
    print(json.dumps({"candidate_id": "c48", "status": result["status"]}, sort_keys=True))
