"""Freeze and run the inherited data-free C41 design gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
C40_ROOT = REPO_ROOT / "systems" / "40_metric_coupled_transport_transformer"
sys.path.insert(0, str(SYSTEM_ROOT))

from model.semantic_routing import (  # noqa: E402
    MODES,
    SEMANTIC_ROUTING,
    SemanticCarrierRoutingTransformer,
)

# Load C40 under a non-conflicting module name.
import importlib.util  # noqa: E402


spec = importlib.util.spec_from_file_location(
    "c40_metric_coupled",
    C40_ROOT / "model" / "metric_coupled.py",
)
c40_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(c40_module)


LOCK_PATH = SYSTEM_ROOT / "notes" / "design_lock.json"
REPORT_PATH = REPO_ROOT / "reports" / "pps_c41_design_gate.json"
LOCKED_FILES = (
    "README.md",
    "environment.txt",
    "configs/design_gate.yaml",
    "model/__init__.py",
    "model/semantic_routing.py",
    "notes/proposal.md",
    "notes/mechanism_fingerprint.md",
    "notes/nearest_neighbors.md",
    "notes/design_gate_protocol.md",
    "probe/run_design_gate.py",
    "tests/test_model.py",
)
EXTERNAL_FILES = (
    "reports/pps_c40_design_gate.json",
    "systems/40_metric_coupled_transport_transformer/model/metric_coupled.py",
    "systems/40_metric_coupled_transport_transformer/notes/design_lock.json",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def load_config() -> dict[str, Any]:
    with (SYSTEM_ROOT / "configs" / "design_gate.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("C41 config must be an object")
    return value


def freeze() -> dict[str, Any]:
    if REPORT_PATH.exists():
        raise RuntimeError("C41 outcome already exists")
    files = {name: sha256_file(SYSTEM_ROOT / name) for name in LOCKED_FILES}
    external = {name: sha256_file(REPO_ROOT / name) for name in EXTERNAL_FILES}
    payload = {
        "candidate_id": "c41",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "locked_files": files,
        "external_files": external,
        "repository_data_authorized": False,
        "dev_test_authorized": False,
    }
    payload["content_sha256"] = hashlib.sha256(
        json.dumps(
            {"files": files, "external": external},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    write_json(LOCK_PATH, payload)
    return payload


def verify_lock() -> tuple[dict[str, Any], str]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    files = {name: sha256_file(SYSTEM_ROOT / name) for name in LOCKED_FILES}
    external = {name: sha256_file(REPO_ROOT / name) for name in EXTERNAL_FILES}
    if files != lock["locked_files"] or external != lock["external_files"]:
        raise RuntimeError("C41 design lock differs")
    return lock, sha256_file(LOCK_PATH)


def make_model(config: Mapping[str, Any], seed: int, mode: str):
    row = config["model"]
    return SemanticCarrierRoutingTransformer(
        dim=int(row["dim"]),
        heads=int(row["heads"]),
        rank=int(row["rank"]),
        temperature=float(row["temperature"]),
        profile_scale=float(row["profile_scale"]),
        correction_scale=float(row["correction_scale"]),
        seed=seed,
        mode=mode,
        init_std=float(row["init_std"]),
    )


def state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run() -> dict[str, Any]:
    lock, lock_sha = verify_lock()
    config = load_config()
    if REPORT_PATH.exists():
        raise FileExistsError(REPORT_PATH)
    if not torch.cuda.is_available():
        raise RuntimeError("C41 design gate requires CUDA")
    device = torch.device(str(config["device"]))
    c40_path = REPO_ROOT / config["inputs"]["c40_report"]
    if sha256_file(c40_path) != config["inputs"]["c40_report_sha256"]:
        raise RuntimeError("C40 report hash differs")
    c40 = json.loads(c40_path.read_text(encoding="utf-8"))
    threshold = config["thresholds"]
    inherited_rows = {}
    inherited_checks = []
    for seed, row in c40["D1"]["seed_reports"].items():
        selection = row["modes"]["selection_only"]
        coupled = row["modes"]["multihead_coupled"]
        difference = selection["clean_ndcg10"] - coupled["clean_ndcg10"]
        clean_wrong = selection["clean_minus_wrong"]
        checks = {
            "selection_over_coupled": difference
            >= float(threshold["inherited_selection_minus_coupled_min"]),
            "selection_clean_over_wrong": clean_wrong
            >= float(threshold["inherited_clean_minus_wrong_min"]),
            "selection_finite": bool(selection["training"]["finite"]),
            "selection_exact_fallback": selection["exact_fallback_max_abs"] == 0.0,
        }
        inherited_checks.extend(checks.values())
        inherited_rows[seed] = {
            "selection_minus_coupled": difference,
            "selection_clean_minus_wrong": clean_wrong,
            "checks": checks,
        }

    generator = torch.Generator().manual_seed(20262001)
    dim = int(config["model"]["dim"])
    query = torch.randn(dim, generator=generator).to(device)
    history = torch.randn(7, dim, generator=generator).to(device)
    candidates = torch.randn(9, dim, generator=generator).to(device)
    permutation = torch.tensor([4, 0, 8, 1, 6, 2, 7, 3, 5], device=device)
    seed = 20262001
    reports = {}
    hashes = []
    counts = []
    for mode in MODES:
        model = make_model(config, seed, mode).to(device)
        hashes.append(state_sha256(model))
        counts.append(model.trainable_parameter_count())
        first = model(query, history, candidates)
        second = model(query, history, candidates)
        permuted = model(query, history, candidates[permutation])
        loss = (first * torch.linspace(-1, 1, len(first), device=device)).sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        reports[mode] = {
            "parameters": model.trainable_parameter_count(),
            "finite": bool(torch.isfinite(first).all()),
            "deterministic_max_abs": float((first - second).abs().max().cpu()),
            "permutation_max_abs": float(
                (first[permutation] - permuted).abs().max().cpu()
            ),
            "nohistory_max_abs": float(
                model(query, history[:0], candidates).abs().max().cpu()
            ),
            "query_absent_max_abs": float(
                model(query, history, candidates, query_present=False)
                .abs()
                .max()
                .cpu()
            ),
            "repeat_max_abs": float(
                model(query, history, candidates, repeat_present=True)
                .abs()
                .max()
                .cpu()
            ),
            "down_gradient_nonzero": bool(model.down.grad.ne(0).any()),
            "up_gradient_nonzero": bool(model.up.grad.ne(0).any()),
        }

    c41_primary = make_model(config, seed, SEMANTIC_ROUTING).to(device)
    c40_primary = c40_module.MetricCoupledTransportTransformer(
        dim=dim,
        heads=int(config["model"]["heads"]),
        rank=int(config["model"]["rank"]),
        temperature=float(config["model"]["temperature"]),
        profile_scale=float(config["model"]["profile_scale"]),
        correction_scale=float(config["model"]["correction_scale"]),
        seed=seed,
        mode=c40_module.SELECTION_ONLY,
        init_std=float(config["model"]["init_std"]),
    ).to(device)
    c40_primary.load_state_dict(c41_primary.state_dict())
    equivalence = float(
        (
            c41_primary(query, history, candidates)
            - c40_primary(query, history, candidates)
        )
        .abs()
        .max()
        .cpu()
    )
    state = c41_primary.components(query, history, candidates)
    raw_history = torch.nn.functional.normalize(history, dim=-1)
    reproduced = torch.einsum("hj,jd->hd", state["attention"], raw_history)
    profile_error = float((state["profile"] - reproduced).abs().max().cpu())
    attention_sum_error = float(
        (state["attention"].sum(dim=-1) - 1).abs().max().cpu()
    )
    limit = float(threshold["permutation_max_abs"])
    checks = {
        "c40_hash_and_isolation": (
            c40["repository_data_read"] is False
            and c40["dev_test_read"] is False
            and all(c40["D0"]["checks"].values())
        ),
        "inherited_conditional_evidence": all(inherited_checks),
        "equal_parameters": len(set(counts)) == 1,
        "paired_initialization": len(set(hashes)) == 1,
        "finite": all(row["finite"] for row in reports.values()),
        "deterministic": all(
            row["deterministic_max_abs"] == 0.0 for row in reports.values()
        ),
        "candidate_permutation": all(
            row["permutation_max_abs"] <= limit for row in reports.values()
        ),
        "exact_fallbacks": all(
            row["nohistory_max_abs"] == 0.0
            and row["query_absent_max_abs"] == 0.0
            and row["repeat_max_abs"] == 0.0
            for row in reports.values()
        ),
        "both_factors_receive_gradient": all(
            row["down_gradient_nonzero"] and row["up_gradient_nonzero"]
            for row in reports.values()
        ),
        "exact_c40_selection_equivalence": equivalence
        <= float(threshold["equivalence_max_abs"]),
        "raw_semantic_profile": profile_error <= 1e-7,
        "simplex_attention": (
            bool((state["attention"] >= 0).all()) and attention_sum_error <= 2e-7
        ),
    }
    report = {
        "candidate_id": "c41",
        "gate_id": config["gate_id"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design_lock_sha256": lock_sha,
        "design_lock_content_sha256": lock["content_sha256"],
        "checks": checks,
        "inherited_rows": inherited_rows,
        "mode_reports": reports,
        "c40_equivalence_max_abs": equivalence,
        "raw_profile_max_abs_error": profile_error,
        "attention_sum_max_abs_error": attention_sum_error,
        "repository_data_read": False,
        "dev_test_read": False,
        "novelty_status": "boundary_only",
        "status": "passed_design_gate" if all(checks.values()) else "failed_terminal",
    }
    write_json(REPORT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("lock", "run"), required=True)
    args = parser.parse_args()
    value = freeze() if args.stage == "lock" else run()
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
