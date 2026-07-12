"""Copy C25's untouched A/B partition into a C26-owned selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.structure import load_config, read_json, sha256_file, write_json_once  # noqa: E402


def materialize(config_path: str | Path) -> dict:
    config = load_config(config_path)
    paths = config["paths"]
    for name, expected_name in (
        ("c25_selection", "c25_selection_sha256"),
        ("c25_g0_report", "c25_g0_report_sha256"),
        ("c25_train_report", "c25_train_report_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("query_token_manifest", "query_token_manifest_sha256"),
    ):
        if sha256_file(paths[name]) != paths[expected_name]:
            raise ValueError(f"C26 registered source changed: {name}")
    source = read_json(paths["c25_selection"])
    g0 = read_json(paths["c25_g0_report"])
    outcome = read_json(paths["c25_train_report"])
    if g0.get("internal_A_labels_opened") is not False or g0.get(
        "delayed_B_labels_opened"
    ) is not False:
        raise ValueError("C25 G0 label boundary differs")
    if outcome.get("internal_A_labels_opened") is not False or outcome.get(
        "delayed_B_labels_opened"
    ) is not False:
        raise ValueError("C25 terminal label boundary differs")
    result = {
        "candidate_id": "c26",
        "selection_id": "c26_token_bridge_selection_v1",
        "status": "frozen_before_any_c26_label_or_outcome",
        "roles": source["roles"],
        "wrong_history_donors": source["wrong_history_donors"],
        "donor_matching": source["donor_matching"],
        "sources": {
            "c25_selection_sha256": paths["c25_selection_sha256"],
            "c25_g0_report_sha256": paths["c25_g0_report_sha256"],
            "c25_train_report_sha256": paths["c25_train_report_sha256"],
        },
        "checks": {
            "c25_fit_labels_previously_opened": True,
            "c25_internal_A_labels_opened": False,
            "c25_delayed_B_labels_opened": False,
            "c26_internal_A_labels_opened": False,
            "c26_delayed_B_labels_opened": False,
            "roles_pairwise_disjoint": True,
            "donor_candidate_overlap_zero": True,
            "dev_test_qrels_metrics_read": False,
        },
    }
    write_json_once(paths["selection"], result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = materialize(args.config)
    print(
        json.dumps(
            {
                "selection": result["selection_id"],
                "role_counts": {name: len(row["indices"]) for name, row in result["roles"].items()},
                "checks": result["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
