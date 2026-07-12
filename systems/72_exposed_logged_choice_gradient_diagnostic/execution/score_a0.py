from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.c71_helpers import load_helpers  # noqa: E402
from execution.locking import load_config, sha256_file, timestamp, verify_execution_lock  # noqa: E402
from execution.selection import candidate_key_sha256, load_json_map  # noqa: E402


def main() -> None:
    config = load_config(SYSTEM_ROOT / "configs/diagnostic.yaml")
    lock, lock_hash = verify_execution_lock(config)
    expected_gpu = str(config["resources"]["physical_gpu"])
    if os.environ.get("CUDA_VISIBLE_DEVICES") != expected_gpu:
        raise RuntimeError(f"C72 requires physical GPU {expected_gpu}")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("C72 expects one visible GPU")
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    helpers = load_helpers()
    paths = config["paths"]
    selection_path = REPO_ROOT / paths["selection"]
    selection = json.loads(selection_path.read_text())
    if selection["status"] != "passed" or selection["proposal_lock_sha256"] != lock["proposal_lock_sha256"]:
        raise RuntimeError("C72 selection identity differs")
    records = helpers.load_required_records(REPO_ROOT / paths["records_train"], selection)
    target_rows = [records[row["request_id"]] for row in selection["targets"]]
    if candidate_key_sha256(target_rows) != selection["candidate_key_sha256"]:
        raise RuntimeError("C72 candidate key changed before scoring")
    item_map = load_json_map(REPO_ROOT / paths["item_id_map"]); query_map = load_json_map(REPO_ROOT / paths["request_query_map"])
    item_embeddings = np.load(REPO_ROOT / paths["item_embeddings"], mmap_mode="r"); query_embeddings = np.load(REPO_ROOT / paths["query_embeddings"], mmap_mode="r")
    op = config["operator"]
    cache = helpers.EpisodeCache(
        records=records, item_map=item_map, item_embeddings=item_embeddings, query_map=query_map,
        query_embeddings=query_embeddings, device=device,
        temperature=float(op["historical_query_slate_temperature"]), epsilon=float(op["normalization_epsilon"]),
    )
    donors = {row["request_id"]: row for row in selection["selected_donors"]}
    donor_by_target = {row["target_request_id"]: donors[row["donor_request_id"]] for row in selection["wrong_donors"]}
    scores = {name: [] for name in helpers.SCORE_NAMES}
    deterministic = permutation = nohistory = 0.0; nonzero = total = order_changes = top10_changes = 0
    started = time.monotonic()
    for position, target in enumerate(selection["targets"]):
        donor = donor_by_target[target["request_id"]]
        output, activity = helpers.score_one(
            target, donor, item_map=item_map, item_embeddings=item_embeddings, query_embeddings=query_embeddings,
            cache=cache, config=config, device=device,
        )
        for name in helpers.SCORE_NAMES: scores[name].append(output[name])
        nonzero += activity[0]; total += activity[1]
        base_rank = helpers.rankings(target["candidate_ids"], output["base"]); primary_rank = helpers.rankings(target["candidate_ids"], output["primary_true"])
        order_changes += int(base_rank != primary_rank); top10_changes += int(base_rank[:10] != primary_rank[:10])
        if position < 64:
            repeat, _ = helpers.score_one(target, donor, item_map=item_map, item_embeddings=item_embeddings, query_embeddings=query_embeddings, cache=cache, config=config, device=device)
            reverse, _ = helpers.score_one(target, donor, item_map=item_map, item_embeddings=item_embeddings, query_embeddings=query_embeddings, cache=cache, config=config, device=device, candidate_ids=list(reversed(target["candidate_ids"])))
            deterministic = max(deterministic, max(float(np.max(np.abs(output[name] - repeat[name]))) for name in helpers.SCORE_NAMES))
            permutation = max(permutation, max(float(np.max(np.abs(output[name] - reverse[name][::-1]))) for name in helpers.SCORE_NAMES))
            query = torch.from_numpy(np.asarray(query_embeddings[int(target["query_embedding_index"])], dtype=np.float32).copy()).to(device)
            candidates = helpers.embedding_tensor(target["candidate_ids"], item_map, item_embeddings, device)
            empty_scores, empty_correction = helpers.score_memory(query, candidates, torch.zeros_like(query), correction_scale=float(op["correction_scale"]), epsilon=float(op["normalization_epsilon"]))
            nohistory = max(nohistory, float((empty_scores - torch.from_numpy(output["base"]).to(device)).abs().max().cpu()), float(empty_correction.abs().max().cpu()))
    root = REPO_ROOT / paths["artifact_root"]; root.mkdir(parents=True, exist_ok=True)
    score_path = root / "scores.npz"; report_path = root / "a0_report.json"
    if score_path.exists() or report_path.exists(): raise FileExistsError(score_path if score_path.exists() else report_path)
    offsets, _ = helpers.flatten(scores["base"])
    with score_path.open("wb") as handle: np.savez(handle, offsets=offsets, **{name: helpers.flatten(rows)[1] for name, rows in scores.items()})
    primary = helpers.flatten(scores["primary_correction"])[1].astype(np.float64); wrong = helpers.flatten(scores["wrong_correction"])[1].astype(np.float64)
    primary_rms = float(np.sqrt(np.mean(primary**2))); wrong_rms = float(np.sqrt(np.mean((primary-wrong)**2))); count = len(selection["targets"])
    diagnostics = {"deterministic_max_abs":deterministic,"candidate_permutation_max_abs":permutation,"nohistory_max_abs":nohistory,"nonzero_episode_gradient_fraction":nonzero/max(1,total),"primary_correction_rms":primary_rms,"true_wrong_correction_rms":wrong_rms,"order_change_fraction_vs_base":order_changes/count,"top10_change_fraction_vs_base":top10_changes/count,"requests":count,"episode_values_cached":len(cache.cache)}
    gate = config["mechanical_gate"]
    checks = {
        "selection_passed":selection["status"]=="passed", "candidate_key_matches":candidate_key_sha256(target_rows)==selection["candidate_key_sha256"], "request_count_exact":count==int(config["selection"]["target_requests"]),
        "scores_finite":all(np.isfinite(row).all() for values in scores.values() for row in values), "deterministic":deterministic<=float(gate["deterministic_tolerance"]), "candidate_permutation":permutation<=float(gate["candidate_permutation_tolerance"]), "nohistory_exact":nohistory<=float(gate["nohistory_tolerance"]),
        "episode_gradients_active":diagnostics["nonzero_episode_gradient_fraction"]>=float(gate["minimum_nonzero_episode_gradient_fraction"]), "primary_correction_active":primary_rms>=float(gate["minimum_primary_correction_rms"]), "true_wrong_active":wrong_rms>=float(gate["minimum_true_wrong_correction_rms"]), "order_active":diagnostics["order_change_fraction_vs_base"]>=float(gate["minimum_order_change_fraction_vs_base"]), "top10_active":diagnostics["top10_change_fraction_vs_base"]>=float(gate["minimum_top10_change_fraction_vs_base"]),
        "labels_not_read_during_scoring":True, "dev_test_qrels_closed":True,
    }
    report = {"schema":"myrec.c72.a0.v1","candidate_id":"c72","created_at":timestamp(),"execution_lock_sha256":lock_hash,"proposal_lock_sha256":lock["proposal_lock_sha256"],"selection_sha256":sha256_file(selection_path),"candidate_key_sha256":selection["candidate_key_sha256"],"checks":checks,"diagnostics":diagnostics,"passed_A0":all(checks.values()),"failed_checks":sorted(name for name,value in checks.items() if not value),"score_artifact":{"path":str(score_path.relative_to(REPO_ROOT)),"sha256":sha256_file(score_path)},"elapsed_seconds":time.monotonic()-started,"claim_boundary":{"fresh":False,"formulation_only":True,"dev_test_qrels":False}}
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps({"passed_A0":report["passed_A0"],"failed_checks":report["failed_checks"],"diagnostics":diagnostics},sort_keys=True))


if __name__ == "__main__": main()
