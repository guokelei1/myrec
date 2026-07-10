#!/usr/bin/env python
"""Audit all D2 dev score files without reading qrels."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from myrec.utils.hashing import sha256_file
from myrec.utils.jsonl import iter_jsonl, write_json


SEEDS = [20260708, 20260709, 20260710]
RUN_NAMES = {
    "d2t": "d2t_finetuned_text_dev",
    "d2p": "d2p_text_pop_dev",
}


def main() -> int:
    manifest_path = Path(
        "data/standardized/kuaisearch/v0_lite/candidate_manifest.json"
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected = [
        (str(entry["request_id"]), str(item_id))
        for entry in manifest["entries"]
        if entry["split"] == "dev"
        for item_id in entry["candidate_item_ids"]
    ]
    manifest_sha = sha256_file(manifest_path)
    final_config = "configs/analysis/finetuned_nonpersonalized_control_final.yaml"
    final_sha = sha256_file(final_config)
    runs = {}
    for seed in SEEDS:
        for variant, name in RUN_NAMES.items():
            run_id = f"20260710_kuaisearch_{name}_s{seed}"
            run_dir = Path("runs") / run_id
            with (run_dir / "metadata.json").open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
            if metadata.get("qrels_read") is not False or metadata.get("test_read") is not False:
                raise AssertionError(f"label-isolation metadata failed: {run_id}")
            if metadata.get("candidate_manifest_sha256") != manifest_sha:
                raise AssertionError(f"candidate hash mismatch: {run_id}")
            if metadata.get("final_config_sha256") != final_sha:
                raise AssertionError(f"final config hash mismatch: {run_id}")
            count = 0
            methods = set()
            scores_path = run_dir / "scores.jsonl"
            for count, (row, expected_key) in enumerate(
                zip(iter_jsonl(scores_path), expected, strict=True), start=1
            ):
                key = (str(row["request_id"]), str(row["candidate_item_id"]))
                if key != expected_key:
                    raise AssertionError(
                        f"candidate order mismatch at {run_id}:{count}"
                    )
                if not math.isfinite(float(row["score"])):
                    raise AssertionError(f"non-finite score at {run_id}:{count}")
                methods.add(str(row["method_id"]))
            if count != len(expected) or len(methods) != 1:
                raise AssertionError(f"score coverage/method mismatch: {run_id}")
            runs[run_id] = {
                "method_id": methods.pop(),
                "row_count": count,
                "scores_sha256": sha256_file(scores_path),
                "status": "passed",
            }
    report = {
        "analysis_id": "finetuned_nonpersonalized_control_v1",
        "candidate_manifest_sha256": manifest_sha,
        "dev_candidate_rows": len(expected),
        "final_config_path": final_config,
        "final_config_sha256": final_sha,
        "qrels_read": False,
        "runs": runs,
        "status": "passed",
        "test_read": False,
    }
    write_json("reports/pps_d2_score_audit.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
