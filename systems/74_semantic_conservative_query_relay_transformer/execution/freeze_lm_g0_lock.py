"""Freeze C74 pretrained-LM mechanics before G0."""

from __future__ import annotations

import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SYSTEM_ROOT.parents[1]
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from execution.lm_locking import (  # noqa: E402
    atomic_json,
    load_config,
    sha256_file,
    timestamp,
)


SOURCES = (
    "systems/74_semantic_conservative_query_relay_transformer/configs/kuai_lm_probe.yaml",
    "systems/74_semantic_conservative_query_relay_transformer/model/adaptive_semantic_relay.py",
    "systems/74_semantic_conservative_query_relay_transformer/train/data_bridge.py",
    "systems/74_semantic_conservative_query_relay_transformer/execution/lm_locking.py",
    "systems/74_semantic_conservative_query_relay_transformer/execution/freeze_lm_g0_lock.py",
    "systems/74_semantic_conservative_query_relay_transformer/execution/run_lm_g0.py",
    "systems/74_semantic_conservative_query_relay_transformer/tests/test_adaptive_model.py",
    "systems/74_semantic_conservative_query_relay_transformer/tests/test_lm_protocol.py",
    "systems/74_semantic_conservative_query_relay_transformer/notes/lm_probe_protocol.md",
    "systems/74_semantic_conservative_query_relay_transformer/notes/lm_probe_preimplementation_review.md",
)


def main() -> None:
    config = load_config()
    design_report = REPO_ROOT / config["paths"]["design_gate_report"]
    report = json.loads(design_report.read_text(encoding="utf-8"))
    if report["decision"] != "pass_authorize_pretrained_probe" or not report["passed"]:
        raise PermissionError("C74 design gate did not authorize LM G0")
    target = REPO_ROOT / config["paths"]["lm_probe_g0_lock"]
    value = {
        "candidate_id": "c74",
        "created_at": timestamp(),
        "decision": "authorize_one_registered_label_free_pretrained_lm_G0",
        "source_sha256": {
            relative: sha256_file(REPO_ROOT / relative) for relative in SOURCES
        },
        "authority_sha256": {
            config["paths"]["design_gate_report"]: sha256_file(design_report),
            config["paths"]["design_proposal_lock"]: sha256_file(
                REPO_ROOT / config["paths"]["design_proposal_lock"]
            ),
        },
        "outcome_boundary": {
            "fit_labels_opened": False,
            "validation_labels_opened": False,
            "fresh_features_scores_labels_opened": False,
            "dev_test_qrels_opened": False,
        },
    }
    atomic_json(target, value)
    print(json.dumps({"path": str(target), "sha256": sha256_file(target)}))


if __name__ == "__main__":
    main()
