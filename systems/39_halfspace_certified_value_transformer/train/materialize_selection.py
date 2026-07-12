"""Create the frozen label-free C39 fit/A cohort."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


SYSTEM_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SYSTEM_ROOT))

from train.selection import materialize_selection  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    paths = config["paths"]
    selection = config["selection"]
    value = materialize_selection(
        records_path=paths["records_train_blind"],
        standardized_manifest_path=paths["standardized_manifest"],
        candidate_manifest_path=paths["candidate_manifest"],
        c0_report_path=paths["c0_report"],
        predecessor_selection_path=paths["c38_selection"],
        predecessor_selection_sha256=paths["c38_selection_sha256"],
        output_path=paths["selection"],
        seed=int(selection["seed"]),
        internal_a_requests=int(selection["internal_A_requests"]),
        length_bins=[int(value) for value in selection["history_length_bins"]],
    )
    print(
        json.dumps(
            {
                "roles": {
                    role: len(row["indices"])
                    for role, row in value["roles"].items()
                },
                "reserve": len(value["reserve_indices"]),
                "wrong_donor_audit": value["wrong_donor_audit"],
                "outcome_isolation": value["outcome_isolation"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
