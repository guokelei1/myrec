"""Canonical-order C30 rescore and staged A0/A1 aggregation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT / "train"))

from c30_protocol import atomic_json, load_config, read_json, sha256_file, verify_lock  # noqa: E402

C29_ROOT = REPO_ROOT / "systems" / "29_causally_authenticated_mediation_transformer"
sys.path.insert(0, str(C29_ROOT))

from model.authenticated_mediation import PRIMARY  # noqa: E402
from train.locking import verify_execution_lock, verify_proposal_lock  # noqa: E402
from train.real_data import open_original_labels  # noqa: E402
from train.run_train_gate import (  # noqa: E402
    average_rows,
    candidate_hashes,
    change_fraction,
    make_builder,
    make_model,
    max_difference,
    order_changes,
    utility_gate,
)
from train.structure import load_config as load_c29_config  # noqa: E402
from train.real_data import FrozenMediationStore  # noqa: E402


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def assert_cuda(config: Mapping[str, Any], seed: int, device: torch.device) -> int:
    row = config["source_seeds"][str(seed)]
    physical = int(row["physical_gpu"])
    if str(device) != "cuda:0" or os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical):
        raise RuntimeError("C30 seed/GPU registration differs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("C30 deterministic CUBLAS setting absent")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C30 requires one visible GPU")
    return physical


def canonical_positions(item_ids: Sequence[Any], external_order: Sequence[int]) -> list[int]:
    count = len(item_ids)
    order = [int(value) for value in external_order]
    if sorted(order) != list(range(count)):
        raise ValueError("C30 external candidate order is not a permutation")
    keys = [str(value) for value in item_ids]
    if len(keys) != len(set(keys)):
        raise ValueError("C30 canonical item IDs must be unique within request")
    return sorted(order, key=lambda position: keys[position])


def canonical_score(
    model: torch.nn.Module,
    builder: Any,
    store: FrozenMediationStore,
    indices: Sequence[int],
    c29_config: Mapping[str, Any],
    device: torch.device,
    *,
    history_source: str = "true",
    caller_order: str = "identity",
) -> dict[str, Any]:
    if caller_order not in {"identity", "reverse"}:
        raise ValueError("unexpected C30 caller order")
    model.to(device).eval()
    request_indices = [int(value) for value in indices]
    base_rows = [store.base_row(index) for index in request_indices]
    item_ids = [store.candidate_item_ids(index) for index in request_indices]
    corrections = [np.zeros(len(row), dtype=np.float32) for row in base_rows]
    active: list[bool] = []
    canonical_by_row: list[list[int]] = []
    examples: list[tuple[int, int]] = []
    locations: list[tuple[int, int]] = []
    for row, index in enumerate(request_indices):
        history = store.authenticated_history(index, history_source)
        is_active = (
            bool(store.query_tokens(index, int(c29_config["sequence"]["max_query_content"])))
            and not store.has_repeat(index)
            and bool(history.size)
        )
        active.append(is_active)
        external = list(range(len(base_rows[row])))
        if caller_order == "reverse":
            external.reverse()
        canonical = canonical_positions(item_ids[row], external)
        canonical_by_row.append(canonical)
        if is_active:
            for position in canonical:
                examples.append((index, position))
                locations.append((row, position))
    batch_size = int(c29_config["training"]["batch_size"])
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(examples), batch_size):
            ids, attention = builder.batch(
                examples[start : start + batch_size],
                history_source=history_source,
                authenticated=True,
                device=device,
            )
            chunks.append(model.correction_from_paired_logits(model(ids, attention)).cpu().numpy())
    flat = np.concatenate(chunks).astype(np.float32, copy=False) if chunks else np.empty(0, np.float32)
    for value, (row, position) in zip(flat, locations):
        corrections[row][position] = value
    scores: list[np.ndarray] = []
    for row in range(len(request_indices)):
        if not active[row]:
            corrections[row].fill(0.0)
            scores.append(base_rows[row].copy())
            continue
        ordered = np.asarray(
            [corrections[row][position] for position in canonical_by_row[row]], dtype=np.float64
        )
        corrections[row] -= float(ordered.mean())
        scores.append((base_rows[row] + corrections[row]).astype(np.float32))
    return {
        "request_indices": request_indices,
        "request_ids": [store.data.request_ids[index] for index in request_indices],
        "item_ids": item_ids,
        "scores": scores,
        "base_scores": base_rows,
        "corrections": corrections,
        "active": active,
    }


def flatten_rows(rows: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    offsets = [0]
    for row in rows:
        offsets.append(offsets[-1] + len(row))
    return np.asarray(offsets, dtype=np.int64), np.concatenate(rows).astype(np.float32, copy=False)


def unflatten_rows(offsets: np.ndarray, values: np.ndarray) -> list[np.ndarray]:
    return [
        np.asarray(values[int(offsets[row]) : int(offsets[row + 1])], dtype=np.float32).copy()
        for row in range(len(offsets) - 1)
    ]


def write_npz_once(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise FileExistsError(f"immutable C30 score artifact exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def load_c29() -> tuple[dict[str, Any], FrozenMediationStore]:
    config = load_c29_config(
        C29_ROOT / "configs" / "train_gate.yaml", require_selection=True
    )
    _, proposal_hash = verify_proposal_lock(config)
    verify_execution_lock(config, proposal_hash)
    return config, FrozenMediationStore(config)


def run_seed(
    config: Mapping[str, Any], seed: int, device: torch.device
) -> dict[str, Any]:
    physical = assert_cuda(config, seed, device)
    seed_all(seed)
    _, lock_hash = verify_lock(config)
    source = config["source_seeds"][str(seed)]
    for name in ("report", "scores", "checkpoint"):
        if sha256_file(source[name]) != source[f"{name}_sha256"]:
            raise RuntimeError(f"C30 source seed {name} changed")
    root = Path(config["paths"]["artifact_root"])
    report_path = root / f"seed_{seed}_report.json"
    score_path = root / f"seed_{seed}_canonical_A_scores.npz"
    if report_path.exists() or score_path.exists():
        raise FileExistsError(f"C30 seed output exists: {seed}")
    started = time.time()
    c29_config, store = load_c29()
    hashes = candidate_hashes(store)
    checkpoint = torch.load(source["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("candidate_id") != "c29" or int(checkpoint.get("seed", -1)) != seed:
        raise ValueError("C30 checkpoint identity differs")
    model = make_model(c29_config, PRIMARY)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device)
    builder = make_builder(store, c29_config)
    indices = store.role_indices("internal_A")
    clean = canonical_score(model, builder, store, indices, c29_config, device)
    repeated = canonical_score(model, builder, store, indices, c29_config, device)
    permuted = canonical_score(
        model, builder, store, indices, c29_config, device, caller_order="reverse"
    )
    wrong = canonical_score(
        model,
        builder,
        store,
        indices,
        c29_config,
        device,
        history_source="wrong",
    )
    with np.load(source["scores"], allow_pickle=False) as old:
        old_offsets = np.asarray(old["candidate_offsets"])
        old_clean = unflatten_rows(old_offsets, np.asarray(old["clean_scores"]))
    offsets, clean_scores = flatten_rows(clean["scores"])
    _, clean_corrections = flatten_rows(clean["corrections"])
    _, wrong_scores = flatten_rows(wrong["scores"])
    _, wrong_corrections = flatten_rows(wrong["corrections"])
    _, base_scores = flatten_rows(clean["base_scores"])
    write_npz_once(
        score_path,
        request_indices=np.asarray(indices, dtype=np.int64),
        candidate_offsets=offsets,
        clean_scores=clean_scores,
        clean_corrections=clean_corrections,
        wrong_scores=wrong_scores,
        wrong_corrections=wrong_corrections,
        base_scores=base_scores,
    )
    old_common = {
        "request_ids": clean["request_ids"],
        "item_ids": clean["item_ids"],
        "scores": old_clean,
    }
    output = {
        "candidate_id": "c30",
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.time() - started,
        "physical_gpu": physical,
        "continuation_lock_sha256": lock_hash,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "weights_changed": False,
        "optimizer_steps": 0,
        "candidate_hashes": hashes,
        "score_artifact": {"path": str(score_path), "sha256": sha256_file(score_path)},
        "internal_A_labels_opened": False,
        "delayed_B_escrow_dev_test_opened": False,
        "canonical_deterministic_max_abs_difference": max_difference(
            clean["scores"], repeated["scores"]
        ),
        "canonical_permutation_max_abs_difference": max_difference(
            clean["scores"], permuted["scores"]
        ),
        "canonical_vs_c29_max_abs_difference": max_difference(
            clean["scores"], old_clean
        ),
        "canonical_vs_c29_order_changes": order_changes(old_common, clean),
        "wrong_correction_change_fraction": change_fraction(
            clean["corrections"], wrong["corrections"]
        ),
        "wrong_order_changes": order_changes(wrong, clean),
    }
    atomic_json(report_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    return output


def load_scores(report: Mapping[str, Any], store: FrozenMediationStore) -> dict[str, Any]:
    path = Path(report["score_artifact"]["path"])
    if sha256_file(path) != report["score_artifact"]["sha256"]:
        raise RuntimeError("C30 score artifact changed")
    with np.load(path, allow_pickle=False) as data:
        indices = [int(value) for value in data["request_indices"]]
        offsets = np.asarray(data["candidate_offsets"])
        clean = unflatten_rows(offsets, np.asarray(data["clean_scores"]))
        correction = unflatten_rows(offsets, np.asarray(data["clean_corrections"]))
        wrong = unflatten_rows(offsets, np.asarray(data["wrong_scores"]))
        wrong_correction = unflatten_rows(offsets, np.asarray(data["wrong_corrections"]))
        base = unflatten_rows(offsets, np.asarray(data["base_scores"]))
    if indices != store.role_indices("internal_A"):
        raise RuntimeError("C30 score role differs")
    common = {
        "request_ids": [store.data.request_ids[index] for index in indices],
        "item_ids": [store.candidate_item_ids(index) for index in indices],
    }
    return {
        "clean": {**common, "scores": clean, "corrections": correction, "base_scores": base},
        "wrong": {
            **common,
            "scores": wrong,
            "corrections": wrong_correction,
            "base_scores": base,
        },
    }


def aggregate(config: Mapping[str, Any]) -> dict[str, Any]:
    _, lock_hash = verify_lock(config)
    root = Path(config["paths"]["artifact_root"])
    target = root / "continuation_report.json"
    if target.exists():
        raise FileExistsError("immutable C30 report exists")
    c29_config, store = load_c29()
    c29_report = read_json(config["paths"]["c29_train_report"])
    source_failures = [
        name for name, passed in c29_report["A0"]["checks"].items() if not passed
    ]
    seeds = [int(value) for value in config["training"]["seeds"]]
    seed_reports: dict[int, dict[str, Any]] = {}
    outputs: dict[int, dict[str, Any]] = {}
    for seed in seeds:
        row = read_json(root / f"seed_{seed}_report.json")
        if int(row.get("seed", -1)) != seed or row.get("internal_A_labels_opened") is not False:
            raise ValueError("C30 seed boundary differs")
        seed_reports[seed] = row
        outputs[seed] = load_scores(row, store)
    clean_average = average_rows([outputs[seed]["clean"]["scores"] for seed in seeds])
    base = outputs[seeds[0]]["clean"]["base_scores"]
    common = {
        "request_ids": outputs[seeds[0]]["clean"]["request_ids"],
        "item_ids": outputs[seeds[0]]["clean"]["item_ids"],
    }
    activity = order_changes({**common, "scores": base}, {**common, "scores": clean_average})
    gate = config["gate"]
    inherited = dict(c29_report["A0"]["checks"])
    inherited.pop("candidate_permutation")
    checks = {
        "source_only_permutation_failed": source_failures == ["candidate_permutation"],
        "source_other_A0_checks_passed": all(inherited.values()),
        "weights_unchanged_no_optimizer": all(
            row["weights_changed"] is False and int(row["optimizer_steps"]) == 0
            for row in seed_reports.values()
        ),
        "canonical_deterministic": all(
            float(row["canonical_deterministic_max_abs_difference"])
            <= float(gate["deterministic_max_abs_difference"])
            for row in seed_reports.values()
        ),
        "canonical_candidate_permutation": all(
            float(row["canonical_permutation_max_abs_difference"])
            <= float(gate["candidate_permutation_max_abs_difference"])
            for row in seed_reports.values()
        ),
        "order_active": activity["any_fraction"] >= float(gate["order_change_fraction_min"]),
        "top10_active": activity["top10_fraction"] >= float(gate["top10_change_fraction_min"]),
        "wrong_changes_correction": all(
            float(row["wrong_correction_change_fraction"])
            >= float(gate["wrong_correction_change_fraction_min"])
            for row in seed_reports.values()
        ),
        "wrong_changes_order": all(
            float(row["wrong_order_changes"]["any_fraction"])
            >= float(gate["wrong_order_change_fraction_min"])
            for row in seed_reports.values()
        ),
        "wrong_changes_top10": all(
            float(row["wrong_order_changes"]["top10_fraction"])
            >= float(gate["wrong_top10_change_fraction_min"])
            for row in seed_reports.values()
        ),
        "source_A_labels_closed": c29_report["internal_A_labels_opened"] is False,
        "delayed_B_closed": c29_report["delayed_B_features_labels_scores_opened"] is False,
    }
    a0 = {
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "order_changes_vs_d2p": activity,
        "seed_permutation_max_abs_difference": {
            str(seed): seed_reports[seed]["canonical_permutation_max_abs_difference"]
            for seed in seeds
        },
        "seed_wrong_order_changes": {
            str(seed): seed_reports[seed]["wrong_order_changes"] for seed in seeds
        },
    }
    common_report = {
        "candidate_id": "c30",
        "continuation_lock_sha256": lock_hash,
        "source_c29_report_sha256": config["paths"]["c29_train_report_sha256"],
        "A0": a0,
        "seed_reports": {str(seed): seed_reports[seed] for seed in seeds},
        "weights_changed": False,
        "optimizer_steps": 0,
        "delayed_B_escrow_dev_test_opened": False,
    }
    if a0["status"] != "passed":
        report = {
            **common_report,
            "status": "failed_A0_terminal",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "internal_A_labels_opened": False,
        }
        atomic_json(target, report)
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return report
    hashes = candidate_hashes(store)
    if hashes["internal_A"] != c29_report["candidate_hashes"]["internal_A"]:
        raise RuntimeError("C30 candidate hash changed before A label access")
    indices = store.role_indices("internal_A")
    labels = open_original_labels(
        data=store.data,
        indices=indices,
        path=config["paths"]["train_candidate_labels"],
        selection_path=config["paths"]["c29_selection"],
        selection_sha256=config["paths"]["c29_selection_sha256"],
    )
    # The C29 utility function is architecture-agnostic and uses the shared evaluator.
    a1 = utility_gate(store=store, outputs=outputs, labels=labels, config=config)
    status = "passed_A1_review_authorized" if a1["status"] == "passed" else "failed_A1_terminal"
    report = {
        **common_report,
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "A1": a1,
        "internal_A_labels_opened": True,
    }
    atomic_json(target, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("seed", "aggregate"), required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "seed":
        if args.seed is None:
            parser.error("--seed is required for seed stage")
        run_seed(config, args.seed, torch.device(args.device))
    else:
        aggregate(config)


if __name__ == "__main__":
    main()
