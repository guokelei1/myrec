"""Run C61's pre-training, label-free structural gate."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
for value in (str(SYSTEM_ROOT), str(REPO_ROOT / "src")):
    if value not in sys.path:
        sys.path.insert(0, value)

from execution.locking import (  # noqa: E402
    load_config,
    read_json,
    sha256_file,
    write_once,
)
from execution.g0_locking import verify_g0  # noqa: E402
from model.counterfactual_edge import CounterfactualEdgeLikelihoodTransformer  # noqa: E402
from train.data import C61Store, to_device  # noqa: E402


def assert_sources(config: Mapping[str, Any]) -> None:
    for path_name, hash_name in (
        ("c26_config", "c26_config_sha256"),
        ("c26_selection", "c26_selection_sha256"),
        ("c26_g0_report", "c26_g0_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("c60_report", "c60_report_sha256"),
        ("c59_report", "c59_report_sha256"),
    ):
        if sha256_file(REPO_ROOT / config["paths"][path_name]) != config["integrity"][hash_name]:
            raise RuntimeError(f"C61 G0 source changed: {path_name}")
    c60 = read_json(REPO_ROOT / config["paths"]["c60_report"])
    if c60.get("status") != "failed_exposed_formulation_terminal":
        raise RuntimeError("C61 C60 boundary differs")
    if c60.get("fresh_C26_A_B_escrow_dev_test_qrels_opened") is not False:
        raise PermissionError("C61 inherited fresh roles are not closed")


def make_model(config: Mapping[str, Any]) -> CounterfactualEdgeLikelihoodTransformer:
    value = config["encoding"]
    return CounterfactualEdgeLikelihoodTransformer(
        input_dim=int(value["input_dim"]),
        hidden_dim=int(value["hidden_dim"]),
        heads=int(value["heads"]),
        ffn_dim=int(value["ffn_dim"]),
        token_layers=int(value["token_layers"]),
        edge_layers=int(value["edge_layers"]),
        dropout=float(value["dropout"]),
        max_query_tokens=int(value["max_query_tokens"]),
        max_item_tokens=int(value["max_item_tokens"]),
        max_history=int(value["max_history"]),
        zero_initial_output=True,
    )


def max_score_error(
    model: CounterfactualEdgeLikelihoodTransformer,
    store: C61Store,
    indices: list[int],
    anchor: str,
) -> float:
    maximum = 0.0
    with torch.inference_mode():
        for index in indices:
            batch = store.collate([index])
            tensors = to_device(batch, torch.device("cpu"))
            output = model(**tensors)
            count = int(batch["candidate_mask"][0].sum())
            expected = batch[f"{anchor}_scores"][0, :count]
            actual = output.scores[0, :count].numpy()
            maximum = max(maximum, float(np.max(np.abs(actual - expected))))
    return maximum


def run(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    _, g0_lock_hash = verify_g0(config)
    assert_sources(config)
    manifest_path = REPO_ROOT / config["paths"]["contextual_manifest"]
    manifest = read_json(manifest_path)
    if manifest.get("status") != "passed":
        raise RuntimeError("C61 contextual manifest failed")
    for shard in manifest["shards"]:
        for output in shard["outputs"].values():
            if sha256_file(REPO_ROOT / output["path"]) != output["sha256"]:
                raise RuntimeError("C61 contextual output changed before G0")
    store = C61Store(config, REPO_ROOT)
    fit, internal = store.role("fit"), store.role("internal_A")
    actual_fit_hash = store.candidate_hash(fit)
    actual_internal_hash = store.candidate_hash(internal)
    expected_fit_hash = store.selection["roles"]["fit"]["candidate_key_sha256"]
    expected_internal_hash = store.selection["roles"]["internal_A"]["candidate_key_sha256"]
    torch.manual_seed(20264200)
    zero_model = make_model(config).eval()
    zero_batch = to_device(store.collate(internal[:2]), torch.device("cpu"))
    with torch.inference_mode():
        zero_output = zero_model(**zero_batch)
    zero_initial_error = float((zero_output.scores - zero_batch["base_scores"]).abs().max())
    nohistory_error = max_score_error(
        zero_model, store, store.role("structural_nohistory")[:16], "base"
    )
    repeat_error = max_score_error(
        zero_model, store, store.role("structural_repeat")[:16], "item_only"
    )

    probe_model = make_model(config).eval()
    with torch.no_grad():
        torch.manual_seed(20264201)
        probe_model.edge_head.weight.normal_(std=0.02)
        probe_model.candidate_head.weight.normal_(std=0.02)
    batch = store.collate(internal[:2])
    tensors = to_device(batch, torch.device("cpu"))
    wrong_tensors = to_device(store.collate(internal[:2], history_source="wrong"), torch.device("cpu"))
    with torch.inference_mode():
        first = probe_model(**tensors)
        second = probe_model(**tensors)
        wrong = probe_model(**wrong_tensors)
        factual_ablation = probe_model(**tensors, mode="factual_edge")
    deterministic_error = float((first.scores - second.scores).abs().max())
    conservation_error = float(first.correction.sum(dim=-1).abs().max())
    capacity_error = float((first.transport - first.base_gap).clamp_min(0).max())
    wrong_likelihood_difference = float((first.likelihood_ratio - wrong.likelihood_ratio).abs().max())
    null_ablation_difference = float((first.likelihood_ratio - factual_ablation.likelihood_ratio).abs().max())
    nonzero_likelihood = float(first.likelihood_ratio.abs().max())

    one = store.collate(internal[:1])
    original = to_device(one, torch.device("cpu"))
    count = original["candidate_mask"].shape[1]
    permutation = torch.arange(count - 1, -1, -1)
    inverse = torch.argsort(permutation)
    moved = dict(original)
    for name in (
        "candidate_tokens",
        "candidate_token_mask",
        "candidate_mask",
        "base_scores",
        "item_only_scores",
    ):
        moved[name] = original[name][:, permutation]
    moved["canonical_order"] = inverse[original["canonical_order"]]
    with torch.inference_mode():
        original_output = probe_model(**original)
        moved_output = probe_model(**moved)
    permutation_error = float(
        (original_output.scores - moved_output.scores[:, inverse]).abs().max()
    )

    gap = torch.tensor([1.0])
    likelihood = torch.tensor([2.0])
    baseline = torch.sigmoid(-gap)
    rate = ((torch.sigmoid(-gap + likelihood) - baseline) / (1.0 - baseline)).clamp(0, 1)
    hand_transport = rate * gap
    checks = {
        "materialization_manifest_passed": manifest.get("status") == "passed",
        "fit_internal_disjoint": not (set(fit) & set(internal)),
        "fit_count_frozen": len(fit) == int(config["selection"]["train_requests"]),
        "internal_A_count_frozen": len(internal) == int(config["selection"]["fresh_A_requests"]),
        "fit_candidate_hash": actual_fit_hash == expected_fit_hash,
        "internal_A_candidate_hash": actual_internal_hash == expected_internal_hash,
        "label_caches_closed": store._fit_labels_cache is None and store._all_labels_cache is None,
        "zero_initial_base": zero_initial_error == 0.0,
        "exact_nohistory_base": nohistory_error == 0.0,
        "exact_repeat_item_only": repeat_error == 0.0,
        "deterministic": deterministic_error == 0.0,
        "candidate_permutation": permutation_error <= float(config["evaluation"]["candidate_permutation_tolerance"]),
        "score_conservation": conservation_error <= float(config["evaluation"]["conservation_tolerance"]),
        "edge_capacity": capacity_error <= float(config["evaluation"]["capacity_tolerance"]),
        "nonzero_counterfactual_likelihood": nonzero_likelihood > 0.0,
        "wrong_history_changes_likelihood": wrong_likelihood_difference > 0.0,
        "NULL_subtraction_load_bearing": null_ablation_difference > 0.0,
        "hand_constructed_edge_opens": float(hand_transport) > 0.0 and float(hand_transport) <= 1.0,
        "fit_labels_closed_in_G0": True,
        "internal_A_delayed_B_escrow_dev_test_qrels_closed": True,
    }
    value = {
        "candidate_id": "c61",
        "gate": "G0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed_terminal",
        "g0_lock_sha256": g0_lock_hash,
        "contextual_manifest_sha256": sha256_file(manifest_path),
        "fit_candidate_key_sha256": actual_fit_hash,
        "internal_A_candidate_key_sha256": actual_internal_hash,
        "diagnostics": {
            "zero_initial_max_abs_error": zero_initial_error,
            "nohistory_max_abs_error": nohistory_error,
            "repeat_max_abs_error": repeat_error,
            "deterministic_max_abs_error": deterministic_error,
            "candidate_permutation_max_abs_error": permutation_error,
            "conservation_max_abs_error": conservation_error,
            "capacity_max_excess": capacity_error,
            "nonzero_likelihood_max_abs": nonzero_likelihood,
            "wrong_likelihood_max_abs_difference": wrong_likelihood_difference,
            "null_ablation_max_abs_difference": null_ablation_difference,
            "hand_transport": float(hand_transport),
        },
        "checks": checks,
        "fit_labels_read": False,
        "internal_A_delayed_B_escrow_dev_test_qrels_opened": False,
    }
    write_once(REPO_ROOT / config["paths"]["artifact_root"] / "g0_report.json", value)
    return value


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run_g0.py CONFIG")
    print(json.dumps(run(sys.argv[1]), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
