"""Freeze C50 before any dual-memory score is computed."""

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
C49_PROBE = REPO_ROOT / "systems/49_prequential_innovation_memory_transformer/probe"
sys.path.insert(0, str(C49_PROBE))

from freeze_lock import load_config as load_c49_config, verify_lock as verify_c49_lock  # noqa: E402

C47_ROOT = REPO_ROOT / "systems/47_posterior_supported_ridge_transformer"
if str(C47_ROOT) not in sys.path:
    sys.path.insert(0, str(C47_ROOT))
from probe.locking import sha256_file  # noqa: E402


LOCAL_FILES = (
    "README.md",
    "configs/formulation_gate.yaml",
    "model/__init__.py",
    "model/dual_memory.py",
    "notes/nearest_neighbors.md",
    "notes/proposal.md",
    "notes/reduction_audit.md",
    "probe/__init__.py",
    "probe/freeze_protocol.py",
    "probe/run_formulation_gate.py",
    "tests/test_dual_memory.py",
    "tests/test_runner.py",
)


def load_config(path: str | Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("candidate_id") != "c50":
        raise ValueError("unexpected C50 config")
    return value


def write_once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def freeze(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    paths = config["paths"]
    c49 = load_c49_config(REPO_ROOT / paths["c49_config"])
    _, c49_lock_hash = verify_c49_lock(c49)
    report = json.loads((REPO_ROOT / paths["c49_report"]).read_text(encoding="utf-8"))
    if report.get("status") != "failed_exposed_learnability_terminal" or report.get("A_labels_read_after_A0") is not True:
        raise PermissionError("C50 requires terminal exposed C49")
    target = REPO_ROOT / paths["proposal_lock"]
    artifact = REPO_ROOT / paths["artifact_root"]
    promoted = REPO_ROOT / paths["promoted_report"]
    if target.exists() or artifact.exists() or promoted.exists():
        raise RuntimeError("C50 output exists before lock")
    command = [sys.executable, "-m", "pytest", "-q", "systems/50_semantic_protected_dual_memory_transformer/tests"]
    test = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    if test.returncode != 0 or "passed" not in test.stdout:
        raise RuntimeError(f"C50 tests failed: {test.stdout}\n{test.stderr}")
    local = [SYSTEM_ROOT / relative for relative in LOCAL_FILES]
    c49_artifact = REPO_ROOT / c49["paths"]["artifact_root"]
    c49_checkpoint = REPO_ROOT / c49["paths"]["checkpoint_root"]
    generated = [c49_artifact / "a0_report.json", REPO_ROOT / paths["c49_report"]]
    for domain in ("kuai", "amazon"):
        for seed in c49["training"][f"{domain}_seeds"]:
            generated.extend(
                (
                    c49_artifact / f"{domain}_seed_{seed}_report.json",
                    c49_artifact / f"{domain}_seed_{seed}_scores.npz",
                    c49_checkpoint / f"{domain}_seed_{seed}.pt",
                )
            )
    external = [REPO_ROOT / paths["c49_config"], REPO_ROOT / paths["c49_proposal_lock"], *generated]
    missing = [str(path) for path in [*local, *external] if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    value = {
        "candidate_id": "c50",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "locked_before_C50_dual_memory_scores",
        "c49_proposal_lock_sha256": c49_lock_hash,
        "local_files_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in local},
        "external_inputs_sha256": {str(path.relative_to(REPO_ROOT)): sha256_file(path) for path in external},
        "declarations": {
            "C50_scores_opened": False,
            "C47_A_already_exposed": True,
            "optimizer_steps_authorized": 0,
            "fresh_reserve_opened": False,
            "dev_test_qrels_opened": False,
        },
        "test_stdout": test.stdout.strip(),
    }
    write_once(target, value)
    return value


def verify_lock(config: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = REPO_ROOT / config["paths"]["proposal_lock"]
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "locked_before_C50_dual_memory_scores":
        raise RuntimeError("C50 lock status differs")
    for group in ("local_files_sha256", "external_inputs_sha256"):
        for relative, expected in value[group].items():
            source = REPO_ROOT / relative
            if not source.is_file() or sha256_file(source) != expected:
                raise RuntimeError(f"C50 locked input changed: {relative}")
    return value, sha256_file(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = freeze(args.config)
    print(json.dumps({"candidate_id": "c50", "status": result["status"]}, sort_keys=True))
