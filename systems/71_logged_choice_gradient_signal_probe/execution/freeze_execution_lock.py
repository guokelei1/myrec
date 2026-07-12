from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from execution.locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
    verify_proposal_lock,
    verify_registered_inputs,
)


SOURCES = (
    "systems/71_logged_choice_gradient_signal_probe/configs/signal_gate.yaml",
    "systems/71_logged_choice_gradient_signal_probe/model/__init__.py",
    "systems/71_logged_choice_gradient_signal_probe/model/choice_gradient.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/__init__.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/locking.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/selection.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/freeze_proposal_lock.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/materialize_selection.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/freeze_execution_lock.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/score_gate.py",
    "systems/71_logged_choice_gradient_signal_probe/execution/aggregate_gate.py",
    "systems/71_logged_choice_gradient_signal_probe/tests/__init__.py",
    "systems/71_logged_choice_gradient_signal_probe/tests/test_choice_gradient.py",
    "src/myrec/eval/metrics.py",
    "systems/38_cross_domain_global_tangent_transfer/train/gate_metrics.py",
    "artifacts/c71_logged_choice_gradient_signal_probe/signal_v1/selection.json",
    "data/standardized/kuaisearch/v0_lite/records_train.jsonl",
    "data/standardized/kuaisearch/v0_lite/candidate_manifest.json",
    "artifacts/batch2b/b5o_stageb_standardized/data/item_id2idx.json",
    "artifacts/batch2b/b5o_stageb_standardized/data/item_title_emb.npy",
    "artifacts/batch2b/b5o_stageb_standardized/data/session_id2idx.json",
    "artifacts/batch2b/b5o_stageb_standardized/data/query_emb.npy",
    "artifacts/analysis/supervised_motivation_diagnostics/data/train/request_ids.jsonl",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "configs/signal_gate.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    verify_registered_inputs(config)
    _, proposal_hash = verify_proposal_lock(config)
    selection_path = REPO_ROOT / config["paths"]["selection"]
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["status"] != "passed" or selection["proposal_lock_sha256"] != proposal_hash:
        raise RuntimeError("C71 selection is not proposal-bound and passed")
    target = REPO_ROOT / config["paths"]["execution_lock"]
    value = {
        "candidate_id": "c71",
        "created_at": timestamp(),
        "decision": "authorize_one_label_free_gpu_score_and_conditional_label_open",
        "proposal_lock_sha256": proposal_hash,
        "selection_sha256": sha256_file(selection_path),
        "source_sha256": {relative: sha256_file(REPO_ROOT / relative) for relative in SOURCES},
        "outcome_boundary": {
            "target_labels_open_only_after_A0": True,
            "source_episode_labels": False,
            "dev_test_qrels": False,
            "attempts": 1,
        },
    }
    atomic_json(target, value)
    print(target.relative_to(REPO_ROOT))
    print(sha256_file(target))


if __name__ == "__main__":
    main()
