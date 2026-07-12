"""Freeze the C47 S0 implementation and every label-free input."""

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
sys.path.insert(0, str(SYSTEM_ROOT))

from probe.locking import sha256_file, verify  # noqa: E402
from probe.locking_v2 import verify_v2  # noqa: E402
from probe.locking_v3 import verify_v3  # noqa: E402


LOCAL_FILES = (
    "configs/signal_gate_v1.yaml",
    "probe/freeze_signal_lock.py",
    "probe/run_signal_gate.py",
    "probe/signal_features.py",
    "probe/signal_scoring.py",
    "tests/test_signal_scoring.py",
)

REUSED_LOCAL_FILES = (
    "systems/38_cross_domain_global_tangent_transfer/train/features.py",
    "systems/38_cross_domain_global_tangent_transfer/train/gate_metrics.py",
    "systems/38_cross_domain_global_tangent_transfer/train/selection.py",
    "systems/38_cross_domain_global_tangent_transfer/train/store.py",
    "src/myrec/eval/metrics.py",
)

KUAI_STRUCTURAL_FILES = (
    "request_ids.jsonl",
    "candidate_offsets.npy",
    "candidate_embedding_indices.npy",
    "candidate_item_ids.npy",
    "history_offsets.npy",
    "history_embedding_indices.npy",
    "query_indices.npy",
)

SNAPSHOT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("candidate_id") != "c47":
        raise ValueError("unexpected C47 signal config")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths = config["paths"]
    design_config = REPO_ROOT / paths["design_config"]
    design = yaml.safe_load(design_config.read_text(encoding="utf-8"))
    _, proposal_hash = verify(design)
    _, v2_hash = verify_v2(design)
    _, v3_hash = verify_v3(design)
    selection_path = REPO_ROOT / paths["selection"]
    if sha256_file(selection_path) != config["integrity"]["selection_sha256"]:
        raise RuntimeError("C47 selection hash differs before signal lock")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not all(selection.get("checks", {}).values()):
        raise PermissionError("C47 selection checks did not pass")
    if selection["candidate_key_sha256"]["kuai_internal_A"] != config["integrity"]["kuai_candidate_key_sha256"]:
        raise RuntimeError("C47 Kuai candidate key differs")
    if selection["candidate_key_sha256"]["amazon_internal_A"] != config["integrity"]["amazon_candidate_key_sha256"]:
        raise RuntimeError("C47 Amazon candidate key differs")

    forbidden_outputs = (
        paths["amazon_adapter_selection"],
        paths["amazon_feature_root"],
        str(Path(paths["artifact_root"]) / "kuai_fixed_scores.npz"),
        str(Path(paths["artifact_root"]) / "amazon_fixed_scores.npz"),
        str(Path(paths["artifact_root"]) / "a0_report.json"),
        paths["promoted_report"],
    )
    present = [value for value in forbidden_outputs if (REPO_ROOT / value).exists()]
    if present:
        raise RuntimeError(f"C47 feature/score output exists before signal lock: {present}")

    test_command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "systems/47_posterior_supported_ridge_transformer/tests/test_signal_scoring.py",
        "systems/47_posterior_supported_ridge_transformer/tests/test_posterior_ridge.py",
        "systems/47_posterior_supported_ridge_transformer/tests/test_selection.py",
    ]
    test = subprocess.run(test_command, cwd=REPO_ROOT, text=True, capture_output=True)
    if test.returncode != 0 or "14 passed" not in test.stdout:
        raise RuntimeError(f"C47 prelock tests failed: {test.stdout}\n{test.stderr}")

    local_paths = [SYSTEM_ROOT / relative for relative in LOCAL_FILES]
    external_paths = [REPO_ROOT / relative for relative in REUSED_LOCAL_FILES]
    external_paths.extend(
        REPO_ROOT / paths[key]
        for key in (
            "design_config",
            "proposal_lock",
            "proposal_lock_v2",
            "proposal_lock_v3",
            "selection",
            "amazon_records_train_blind",
            "kuai_item_embeddings",
            "kuai_query_embeddings",
        )
    )
    packed_root = REPO_ROOT / paths["kuai_packed_root"]
    external_paths.extend(packed_root / name for name in KUAI_STRUCTURAL_FILES)
    snapshot = REPO_ROOT / paths["amazon_bge_snapshot"]
    external_paths.extend(snapshot / name for name in SNAPSHOT_FILES)
    missing = [str(path) for path in [*local_paths, *external_paths] if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    value = {
        "candidate_id": "c47",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_after_fresh_selection_before_features_scores_or_A_labels",
        "proposal_lock_sha256": proposal_hash,
        "proposal_lock_v2_sha256": v2_hash,
        "proposal_lock_v3_sha256": v3_hash,
        "selection_sha256": sha256_file(selection_path),
        "local_files_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in local_paths
        },
        "label_free_inputs_sha256": {
            str(path.relative_to(REPO_ROOT)): sha256_file(path)
            for path in external_paths
        },
        "declarations": {
            "prelock_tests_passed": 14,
            "features_scores_opened": False,
            "A_labels_opened": False,
            "label_bearing_files_hashed_or_opened": False,
            "dev_test_records_labels_qrels_opened": False,
            "scientific_settings_frozen": True,
        },
        "test_command": " ".join(test_command),
        "test_stdout": test.stdout.strip(),
    }
    target = REPO_ROOT / paths["signal_execution_lock"]
    _write_once(target, value)
    return value


def verify_signal_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    target = REPO_ROOT / config["paths"]["signal_execution_lock"]
    value = json.loads(target.read_text(encoding="utf-8"))
    if value.get("status") != "locked_after_fresh_selection_before_features_scores_or_A_labels":
        raise RuntimeError("C47 signal execution lock status differs")
    for group in ("local_files_sha256", "label_free_inputs_sha256"):
        for relative, expected in value[group].items():
            path = REPO_ROOT / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"C47 signal locked input changed: {relative}")
    if value["declarations"].get("A_labels_opened") is not False:
        raise PermissionError("C47 signal lock is not pre-A-label")
    return value, sha256_file(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = freeze(args.config)
    print(json.dumps({"candidate_id": "c47", "status": result["status"]}, sort_keys=True))
