"""Create the frozen label-free C38 train-internal cohort."""

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
    selection = config["selection"]
    paths = config["paths"]
    value = materialize_selection(
        records_path=paths["records_train_blind"],
        standardized_manifest_path=paths["standardized_manifest"],
        candidate_manifest_path=paths["candidate_manifest"],
        c0_report_path=paths["c0_report"],
        output_path=paths["selection"],
        seed=int(selection["seed"]),
        role_counts={
            role: int(selection[f"{role}_requests"])
            for role in ("fit", "internal_A", "delayed_B", "escrow")
        },
        length_bins=[int(value) for value in selection["history_length_bins"]],
    )
    print(json.dumps(value["wrong_donor_audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
