from __future__ import annotations

import json
from pathlib import Path
import sys


SYSTEM_ROOT=Path(__file__).resolve().parents[1]; REPO_ROOT=SYSTEM_ROOT.parents[1]; sys.path.insert(0,str(SYSTEM_ROOT))
from execution.locking import atomic_json, load_config, sha256_file, timestamp, verify_inputs, verify_proposal_lock  # noqa: E402


SOURCES=(
"systems/72_exposed_logged_choice_gradient_diagnostic/configs/diagnostic.yaml",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/__init__.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/locking.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/selection.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/c71_helpers.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/freeze_proposal_lock.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/materialize_selection.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/score_a0.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/aggregate.py",
"systems/72_exposed_logged_choice_gradient_diagnostic/execution/freeze_execution_lock.py",
"systems/71_logged_choice_gradient_signal_probe/model/choice_gradient.py",
"systems/71_logged_choice_gradient_signal_probe/execution/score_gate.py",
"src/myrec/eval/metrics.py",
"systems/38_cross_domain_global_tangent_transfer/train/gate_metrics.py",
"artifacts/c72_exposed_logged_choice_gradient_diagnostic/diagnostic_v1/selection.json",
"data/standardized/kuaisearch/v0_lite/records_train.jsonl",
"artifacts/batch2b/b5o_stageb_standardized/data/item_id2idx.json",
"artifacts/batch2b/b5o_stageb_standardized/data/item_title_emb.npy",
"artifacts/batch2b/b5o_stageb_standardized/data/session_id2idx.json",
"artifacts/batch2b/b5o_stageb_standardized/data/query_emb.npy",
)


def main()->None:
    config=load_config(SYSTEM_ROOT/"configs/diagnostic.yaml"); verify_inputs(config); _,proposal_hash=verify_proposal_lock(config); selection_path=REPO_ROOT/config["paths"]["selection"]; selection=json.loads(selection_path.read_text())
    if selection["status"]!="passed" or selection["proposal_lock_sha256"]!=proposal_hash: raise RuntimeError("C72 selection not bound")
    target=REPO_ROOT/config["paths"]["execution_lock"]; value={"candidate_id":"c72","created_at":timestamp(),"decision":"authorize_one_exposed_fit_diagnostic","proposal_lock_sha256":proposal_hash,"selection_sha256":sha256_file(selection_path),"source_sha256":{relative:sha256_file(REPO_ROOT/relative) for relative in SOURCES},"claim_boundary":{"fresh":False,"formulation_only":True,"attempts":1,"dev_test_qrels":False}}
    atomic_json(target,value); print(target.relative_to(REPO_ROOT)); print(sha256_file(target))


if __name__=="__main__": main()
